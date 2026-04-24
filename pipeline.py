"""
MLB model daily pipeline.

Fetches data, updates scores, populates stats, runs model, evaluates picks.

Usage: python pipeline.py
"""

import time
import traceback
from collections import defaultdict
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()

from backend.db import engine
from backend.data.mlb_api import fetch_schedule, fetch_probable_starters
from backend.data.fangraphs import fetch_pitcher_stats, fetch_bullpen_stats, fetch_team_batting
from backend.data.savant import fetch_park_factors
from backend.data.odds_api import fetch_odds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timed(label, fn):
    """Run fn, print elapsed time, return result."""
    t0 = time.time()
    result = fn()
    elapsed = time.time() - t0
    print(f"  [{elapsed:.1f}s] {label}")
    return result


def _batch_upsert_games(conn, games_df):
    """Upsert a DataFrame of games into the games table."""
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
    """Replace a stats table's contents atomically."""
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {table_name} RESTART IDENTITY"))
        df.to_sql(table_name, conn, if_exists="append", index=False)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def update_scores_and_schedule():
    """Fetch schedules for recent + upcoming days.

    - Updates final scores for the last 3 days
    - Upserts today's and tomorrow's schedule
    - Refreshes probable starters
    """
    today = date.today()

    # Fetch all dates we care about in one pass: 3 days back + today + tomorrow
    dates = [today - timedelta(days=d) for d in range(3, -1, -1)] + [today + timedelta(days=1)]
    schedules = {}
    for d in dates:
        sched = fetch_schedule(d)
        if not sched.empty:
            schedules[d] = sched

    if not schedules:
        print("  No schedule data returned for any date")
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
    print(f"  {total_games} games upserted across {len(schedules)} dates, {score_updates} scores finalized")

    # Refresh probable starters
    starters = fetch_probable_starters(days_ahead=2)
    if starters.empty:
        print("  No probable starters announced")
        return

    # Filter to only game_pks that exist in the games table (FK constraint)
    with engine.connect() as conn:
        existing = pd.read_sql(text("SELECT game_pk FROM games"), conn)
    existing_pks = set(existing["game_pk"].tolist())
    starters = starters[starters["game_pk"].isin(existing_pks)]
    if starters.empty:
        print("  No starters matched to games in DB")
        return

    # Deduplicate — keep last entry per (game_pk, team) to avoid unique constraint violations
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

    print(f"  {len(starters)} probable starters refreshed")


def fetch_statcast_stats():
    """Fetch pitcher, bullpen, and batting stats from Statcast.

    All three use the same cached Statcast pitch data, so running them
    together avoids redundant API calls.
    """
    pitchers = fetch_pitcher_stats()
    if not pitchers.empty:
        _truncate_and_load("pitcher_stats", pitchers)
        print(f"  {len(pitchers)} pitcher stats")
    else:
        print("  No pitcher stats (Statcast may not have data yet)")

    bullpen = fetch_bullpen_stats()
    if not bullpen.empty:
        _truncate_and_load("bullpen_stats", bullpen)
        print(f"  {len(bullpen)} bullpen team stats")
    else:
        print("  No bullpen stats")

    batting = fetch_team_batting()
    if not batting.empty:
        _truncate_and_load("team_batting", batting)
        print(f"  {len(batting)} team batting rows")
    else:
        print("  No batting stats")


def fetch_and_load_park_factors():
    """Load park factors if not already populated."""
    with engine.connect() as conn:
        count = conn.execute(text("SELECT count(*) FROM park_factors")).scalar()

    if count >= 28:
        print(f"  Already loaded ({count} rows)")
        return

    df = fetch_park_factors()
    if df.empty:
        print("  No park factors available")
        return

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM park_factors"))
        df.to_sql("park_factors", conn, if_exists="append", index=False)

    print(f"  {len(df)} park factors loaded")


def fetch_and_load_odds():
    """Fetch odds and match to game_pk via team name.

    The Odds API doesn't provide game_pk, so we match by team name against
    today's/tomorrow's games. Doubleheaders are an edge case — team name alone
    can't distinguish Game 1 vs Game 2, but game_pk keeps them separate in the DB
    once matched. For doubleheaders, the last game_pk wins (usually Game 2).
    """
    odds = fetch_odds()
    if odds.empty:
        print("  No odds data from API")
        return

    today = date.today()
    tomorrow = today + timedelta(days=1)

    with engine.connect() as conn:
        games = pd.read_sql(
            text("SELECT game_pk, game_date, home_team, away_team, start_time FROM games WHERE game_date IN (:d1, :d2)"),
            conn,
            params={"d1": str(today), "d2": str(tomorrow)},
        )

    if games.empty:
        print("  No games in DB to match odds against")
        return

    # Build team -> [(game_pk, start_time), ...] for time-aware matching
    team_games = defaultdict(list)
    for _, g in games.iterrows():
        entry = (int(g["game_pk"]), g.get("start_time"))
        team_games[g["home_team"]].append(entry)
        team_games[g["away_team"]].append(entry)

    def _match_game_pk(team, commence_time):
        """Match odds to game_pk by team, using nearest start_time for doubleheaders."""
        candidates = team_games.get(team, [])
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0][0]
        # Multiple games (doubleheader) — pick closest by start time
        if pd.notna(commence_time):
            ct = pd.to_datetime(commence_time)
            best_pk, best_diff = None, None
            for pk, st in candidates:
                if pd.notna(st):
                    diff = abs((pd.to_datetime(st) - ct).total_seconds())
                    if best_diff is None or diff < best_diff:
                        best_pk, best_diff = pk, diff
            if best_pk is not None:
                return best_pk
        # Fallback: return last game_pk
        return candidates[-1][0]

    odds["game_pk"] = odds.apply(
        lambda row: _match_game_pk(row["team"], row.get("commence_time")), axis=1
    )

    matched = odds.dropna(subset=["game_pk"])
    if matched.empty:
        print("  Could not match any odds to games")
        return

    matched = matched.copy()
    matched["game_pk"] = matched["game_pk"].astype(int)

    db_cols = ["game_pk", "team", "book", "moneyline", "spread", "spread_odds",
               "total", "total_over_odds", "total_under_odds", "scraped_at"]
    insert_df = matched[[c for c in db_cols if c in matched.columns]].copy()
    if "scraped_at" not in insert_df.columns:
        insert_df["scraped_at"] = pd.Timestamp.utcnow()

    # Drop duplicate (game_pk, team, book) rows — keep last (most recent)
    key_cols = [c for c in ["game_pk", "team", "book"] if c in insert_df.columns]
    insert_df = insert_df.drop_duplicates(subset=key_cols, keep="last")

    game_pks = insert_df["game_pk"].unique().tolist()
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM odds WHERE game_pk = ANY(:pks)"), {"pks": game_pks})
        insert_df.to_sql("odds", conn, if_exists="append", index=False)

    print(f"  {len(insert_df)} odds rows for {len(game_pks)} games")


_model_artifacts = {}

def run_model():
    """Train/predict and write to model_outputs tables."""
    from backend.model import main as model_main
    result = model_main()
    if result:
        _model_artifacts.update(result)


def run_evaluation():
    """Evaluate predictions against actual results."""
    from backend.evaluate_model import main as eval_main
    eval_main(
        model=_model_artifacts.get("model"),
        cv_metrics=_model_artifacts.get("cv_metrics"),
        best_params=_model_artifacts.get("best_params"),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STEPS = [
    ("Schedule & scores", update_scores_and_schedule),
    ("Statcast stats", fetch_statcast_stats),
    ("Park factors", fetch_and_load_park_factors),
    ("Odds", fetch_and_load_odds),
    ("Model", run_model),
    ("Evaluation", run_evaluation),
]

# Nightly steps: refresh scores first so late west-coast games are Final
# before evaluation, then eval yesterday, fetch odds, and rerun predictions.
# No Statcast fetch or park factor reload — morning handles those.
NIGHTLY_STEPS = [
    ("Schedule & scores", update_scores_and_schedule),
    ("Odds", fetch_and_load_odds),
    ("Model", run_model),
    ("Evaluation", run_evaluation),
]


def _run_steps(steps):
    t0 = time.time()
    failed = []
    for name, fn in steps:
        print(f"\n>> {name}")
        try:
            step_t0 = time.time()
            fn()
            print(f"   done ({time.time() - step_t0:.1f}s)")
        except Exception as e:
            print(f"   FAILED: {e}")
            traceback.print_exc()
            failed.append(name)
    elapsed = time.time() - t0
    print(f"\n{'=' * 50}")
    if failed:
        print(f"Pipeline finished in {elapsed:.0f}s with {len(failed)} error(s): {', '.join(failed)}")
    else:
        print(f"Pipeline finished in {elapsed:.0f}s — all steps OK")
    return failed


def main():
    print(f"MLB Pipeline — {date.today()}")
    print("=" * 50)
    return _run_steps(STEPS)


def nightly():
    print(f"MLB Nightly — {date.today()}")
    print("=" * 50)
    return _run_steps(NIGHTLY_STEPS)


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "nightly":
        failed = nightly()
    else:
        failed = main()
    if failed:
        sys.exit(1)
