"""MLB model daily pipeline. Usage: python pipeline.py [nightly]"""

import logging
import os
import time
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta

import pandas as pd
from sqlalchemy import text

from backend.data.bullpen_daily import update_bullpen_daily
from backend.data.fangraphs import fetch_bullpen_stats, fetch_pitcher_stats, fetch_team_batting
from backend.data.mlb_api import fetch_probable_starters, fetch_schedule
from backend.data.odds_api import (
    fetch_odds,
    latest_quota_state,
)
from backend.data.savant import fetch_park_factors
from backend.data.weather import update_weather_for_date
from backend.db import engine
from backend.log import setup_logging

log = logging.getLogger(__name__)


def _timed(label, fn):
    t0 = time.time()
    result = fn()
    log.info(f"[{time.time() - t0:.1f}s] {label}")
    return result


def _batch_upsert_games(conn, games_df):
    if games_df.empty:
        return
    for _, g in games_df.iterrows():
        conn.execute(
            text("""
                INSERT INTO games (game_pk, game_date, home_team, away_team,
                                   home_score, away_score, status, venue, start_time)
                VALUES (:game_pk, :game_date, :home_team, :away_team,
                        :home_score, :away_score, :status, :venue, :start_time)
                ON CONFLICT (game_pk) DO UPDATE SET
                    game_date = EXCLUDED.game_date,
                    start_time = EXCLUDED.start_time,
                    home_team = EXCLUDED.home_team,
                    away_team = EXCLUDED.away_team,
                    venue = EXCLUDED.venue,
                    home_score = COALESCE(EXCLUDED.home_score, games.home_score),
                    away_score = COALESCE(EXCLUDED.away_score, games.away_score),
                    status = EXCLUDED.status,
                    updated_at = now()
            """),
            {
                "game_pk": int(g["game_pk"]),
                "game_date": str(g["game_date"]),
                "home_team": g["home_team"],
                "away_team": g["away_team"],
                "home_score": int(g["home_score"]) if pd.notna(g.get("home_score")) else None,
                "away_score": int(g["away_score"]) if pd.notna(g.get("away_score")) else None,
                "status": g["status"],
                "venue": g.get("venue", ""),
                "start_time": g.get("start_time"),
            },
        )


def _truncate_and_load(table_name, df):
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {table_name} RESTART IDENTITY"))
        df.to_sql(table_name, conn, if_exists="append", index=False)


def update_scores_and_schedule():
    """Fetch last 3 days + today + tomorrow; finalize scores; refresh probable starters."""
    today = date.today()

    # Fetch all dates we care about in one pass: 3 days back + today + tomorrow
    dates = [today - timedelta(days=d) for d in range(3, -1, -1)] + [today + timedelta(days=1)]
    schedules = {}
    for d in dates:
        sched = fetch_schedule(d)
        if not sched.empty:
            schedules[d] = sched

    if not schedules:
        log.info("No schedule data returned for any date")
        return

    # Upsert all games and update scores in one transaction
    score_updates = 0
    with engine.begin() as conn:
        for d, sched in schedules.items():
            _batch_upsert_games(conn, sched)

            # Count score updates for final games
            final = sched[sched["status"] == "Final"]
            if not final.empty and d < today:
                for _, g in final.iterrows():
                    result = conn.execute(
                        text("""
                            UPDATE games SET
                                home_score = :hs, away_score = :as,
                                status = 'Final', updated_at = now()
                            WHERE game_pk = :pk AND status != 'Final'
                        """),
                        {
                            "pk": int(g["game_pk"]),
                            "hs": int(g["home_score"]) if pd.notna(g["home_score"]) else None,
                            "as": int(g["away_score"]) if pd.notna(g["away_score"]) else None,
                        },
                    )
                    score_updates += result.rowcount

    total_games = sum(len(s) for s in schedules.values())
    log.info(f"{total_games} games upserted across {len(schedules)} dates, {score_updates} scores finalized")

    # Refresh probable starters
    starters = fetch_probable_starters(days_ahead=2)
    if starters.empty:
        log.info("No probable starters announced")
        return

    n = upsert_probable_starters(starters)
    log.info(f"{n} probable starters refreshed" if n else "No starters matched to games in DB")


def upsert_probable_starters(starters: pd.DataFrame) -> int:
    """Upsert probable starters (FK-filtered to games in the DB), return rows written.

    Shared by the morning/nightly schedule refresh and the intraday lineup
    refresh, so a starter announced after the morning run still lands in the DB.
    """
    if starters.empty:
        return 0
    with engine.connect() as conn:
        existing_pks = set(pd.read_sql(text("SELECT game_pk FROM games"), conn)["game_pk"].tolist())
    starters = starters[starters["game_pk"].isin(existing_pks)]
    if starters.empty:
        return 0
    starters = starters.drop_duplicates(subset=["game_pk", "team"], keep="last")
    with engine.begin() as conn:
        for _, s in starters.iterrows():
            conn.execute(
                text("""
                    INSERT INTO probable_starters (game_pk, team, pitcher_name, pitcher_id, handedness, is_home)
                    VALUES (:game_pk, :team, :pitcher_name, :pitcher_id, :handedness, :is_home)
                    ON CONFLICT (game_pk, team) DO UPDATE SET
                        pitcher_name = EXCLUDED.pitcher_name,
                        pitcher_id = EXCLUDED.pitcher_id,
                        handedness = EXCLUDED.handedness,
                        is_home = EXCLUDED.is_home
                """),
                {
                    "game_pk": int(s["game_pk"]),
                    "team": s["team"],
                    "pitcher_name": s["pitcher_name"],
                    "pitcher_id": int(s["pitcher_id"]) if pd.notna(s.get("pitcher_id")) else None,
                    "handedness": s.get("handedness"),
                    "is_home": bool(s["is_home"]),
                },
            )
    return len(starters)


def fetch_statcast_stats():
    """Pitcher / bullpen / batting stats. All share the same cached pitch data."""
    pitchers = fetch_pitcher_stats()
    if not pitchers.empty:
        _truncate_and_load("pitcher_stats", pitchers)
        log.info(f"{len(pitchers)} pitcher stats")
    else:
        log.info("No pitcher stats (Statcast may not have data yet)")

    bullpen = fetch_bullpen_stats()
    if not bullpen.empty:
        _truncate_and_load("bullpen_stats", bullpen)
        log.info(f"{len(bullpen)} bullpen team stats")
    else:
        log.info("No bullpen stats")

    batting = fetch_team_batting()
    if not batting.empty:
        _truncate_and_load("team_batting", batting)
        log.info(f"{len(batting)} team batting rows")
    else:
        log.info("No batting stats")


def fetch_and_load_park_factors():
    with engine.connect() as conn:
        count = conn.execute(text("SELECT count(*) FROM park_factors")).scalar()

    if count >= 28:
        log.info(f"Already loaded ({count} rows)")
        return

    df = fetch_park_factors()
    if df.empty:
        log.info("No park factors available")
        return

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM park_factors"))
        df.to_sql("park_factors", conn, if_exists="append", index=False)

    log.info(f"{len(df)} park factors loaded")


def _upcoming_games(target_date: date) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(
            text("""
                SELECT game_pk, game_date, home_team, away_team, start_time
                FROM games
                WHERE game_date = :target_date
                  AND home_score IS NULL
                  AND start_time IS NOT NULL
                  AND start_time > NOW()
                  AND LOWER(COALESCE(status, '')) NOT IN ('final', 'cancelled', 'postponed')
                ORDER BY game_date, start_time, game_pk
            """),
            conn,
            params={"target_date": target_date.isoformat()},
        )


def _has_recent_stored_odds(game_pks: list[int], cutoff: datetime) -> bool:
    if not game_pks:
        return False
    with engine.connect() as conn:
        recent_game_count = conn.execute(
            text("""
                SELECT COUNT(DISTINCT complete.game_pk)
                FROM (
                    SELECT o.game_pk, o.book
                    FROM odds o
                    JOIN games g USING (game_pk)
                    WHERE o.game_pk = ANY(:game_pks)
                      AND o.scraped_at >= :cutoff
                      AND o.team IN (g.home_team, g.away_team)
                    GROUP BY o.game_pk, o.book
                    HAVING COUNT(DISTINCT o.team) = 2
                       AND BOOL_AND(o.moneyline IS NOT NULL)
                       AND BOOL_AND(o.spread IS NOT NULL)
                       AND BOOL_AND(o.spread_odds IS NOT NULL)
                       AND BOOL_AND(o.total IS NOT NULL)
                       AND BOOL_AND(o.total_over_odds IS NOT NULL)
                       AND BOOL_AND(o.total_under_odds IS NOT NULL)
                ) complete
            """),
            {"game_pks": game_pks, "cutoff": cutoff},
        ).scalar()
        return int(recent_game_count or 0) == len(game_pks)


def _replace_odds(insert_df: pd.DataFrame) -> None:
    game_pks = insert_df["game_pk"].unique().tolist()
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM odds WHERE game_pk = ANY(:pks)"), {"pks": game_pks})
        insert_df.to_sql("odds", conn, if_exists="append", index=False)


def _nonnegative_env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _reserve_blocks_optional_refresh() -> bool:
    quota = latest_quota_state()
    if not quota:
        log.info("No persisted Odds API quota state; allowing fallback refresh")
        return False

    remaining = quota.get("x-requests-remaining")
    last_cost = quota.get("x-requests-last")
    if not isinstance(remaining, int):
        log.info("Persisted Odds API remaining quota is unknown; allowing fallback refresh")
        return False
    if not isinstance(last_cost, int) or last_cost <= 0:
        last_cost = 3

    # This repository-local snapshot cannot coordinate simultaneous CFB or NFL
    # consumers. Account-wide safety requires a shared durable ledger and lock.
    reserve = _nonnegative_env_int("ODDS_API_RESERVE_CREDITS", 50)
    projected = remaining - last_cost
    if projected < reserve:
        log.info(
            "Skipping optional Odds API refresh: "
            f"remaining={remaining}, expected_cost={last_cost}, reserve={reserve}"
        )
        return True
    return False


def fetch_and_load_odds(
    target_date: date | str | None = None,
    *,
    optional: bool = False,
) -> int:
    """Load odds for one schedule-backed date, skipping safe duplicates."""
    if target_date is None:
        target_date = date.today()
    elif isinstance(target_date, str):
        target_date = date.fromisoformat(target_date)

    games = _upcoming_games(target_date)
    now = datetime.now(UTC)
    if games.empty:
        log.info(f"No upcoming unstarted games on {target_date}; no Odds API request")
        return 0

    game_pks = sorted({int(pk) for pk in games["game_pk"]})
    upcoming_game_pks = tuple(game_pks)

    if optional:
        cutoff = now - timedelta(hours=3)
        if _has_recent_stored_odds(game_pks, cutoff):
            log.info("Skipping optional Odds API refresh: fresh odds already persisted for this window")
            return 0
        if _reserve_blocks_optional_refresh():
            return 0

    odds = fetch_odds(upcoming_game_pks)
    if odds.empty:
        log.info("No odds data from API")
        return 0

    team_games = defaultdict(list)
    for game in games.itertuples(index=False):
        entry = (int(game.game_pk), game.start_time)
        team_games[game.home_team].append(entry)
        team_games[game.away_team].append(entry)

    def _match_game_pk(team, commence_time):
        candidates = team_games.get(team, [])
        if not candidates:
            return None
        if pd.notna(commence_time):
            commence = pd.to_datetime(commence_time, utc=True)
            timed = [
                (abs((pd.to_datetime(start, utc=True) - commence).total_seconds()), game_pk)
                for game_pk, start in candidates
                if pd.notna(start)
            ]
            if timed:
                difference, game_pk = min(timed)
                return game_pk if difference <= 90 * 60 else None
        return candidates[-1][0]

    odds["game_pk"] = odds.apply(
        lambda row: _match_game_pk(row["team"], row.get("commence_time")), axis=1
    )
    matched = odds.dropna(subset=["game_pk"])
    if matched.empty:
        log.warning("Could not match any odds to games")
        return 0

    matched = matched.copy()
    matched["game_pk"] = matched["game_pk"].astype(int)

    db_cols = ["game_pk", "team", "book", "moneyline", "spread", "spread_odds",
               "total", "total_over_odds", "total_under_odds", "scraped_at"]
    insert_df = matched[[c for c in db_cols if c in matched.columns]].copy()
    if "scraped_at" not in insert_df.columns:
        insert_df["scraped_at"] = pd.Timestamp.utcnow()

    # Drop duplicate (game_pk, team, book) rows - keep last (most recent)
    key_cols = [c for c in ["game_pk", "team", "book"] if c in insert_df.columns]
    insert_df = insert_df.drop_duplicates(subset=key_cols, keep="last")

    _replace_odds(insert_df)
    log.info(f"{len(insert_df)} odds rows for {insert_df['game_pk'].nunique()} games")
    return len(insert_df)


def run_evaluation():
    from backend.evaluate_model import main as eval_main
    eval_main()


def update_weather():
    """Keep the weather table fresh for the dates the pipeline touches."""
    today = date.today()
    for d in [today - timedelta(days=x) for x in range(3, -1, -1)] + [today + timedelta(days=1)]:
        update_weather_for_date(d)


STEPS = [
    ("Schedule & scores", update_scores_and_schedule),
    ("Statcast stats", fetch_statcast_stats),
    ("Bullpen daily", update_bullpen_daily),
    ("Park factors", fetch_and_load_park_factors),
    ("Odds", fetch_and_load_odds),
    ("Weather", update_weather),
    ("Evaluation", run_evaluation),
]

# Nightly: refresh scores first so late west-coast games are Final before eval.
# Skip Statcast fetch and park factor reload; morning run handles those.
NIGHTLY_STEPS = [
    ("Schedule & scores", update_scores_and_schedule),
    ("Bullpen daily", update_bullpen_daily),
    ("Weather", update_weather),
    ("Evaluation", run_evaluation),
]


def _run_steps(steps):
    t0 = time.time()
    failed = []
    for name, fn in steps:
        log.info(f"step: {name}")
        try:
            step_t0 = time.time()
            fn()
            log.info(f"{name} done in {time.time() - step_t0:.1f}s")
        except Exception:
            log.exception(f"{name} failed")
            failed.append(name)
    elapsed = time.time() - t0
    if failed:
        log.warning(f"Pipeline finished in {elapsed:.0f}s with {len(failed)} error(s): {', '.join(failed)}")
    else:
        log.info(f"Pipeline finished in {elapsed:.0f}s - all steps OK")
    return failed


def main():
    log.info(f"MLB pipeline - {date.today()}")
    return _run_steps(STEPS)


def nightly():
    log.info(f"MLB nightly - {date.today()}")
    return _run_steps(NIGHTLY_STEPS)


if __name__ == "__main__":
    setup_logging()
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "nightly":
        failed = nightly()
    else:
        failed = main()
    if failed:
        sys.exit(1)
