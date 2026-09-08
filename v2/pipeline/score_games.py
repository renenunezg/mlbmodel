"""Score games for a date and write to model_outputs.

Usage:
    python -m v2.pipeline.score_games --date 2026-09-09 --n-sims 10000

Steps per game:
  1. Read probable_starters and odds from Supabase.
  2. Fetch posted lineups via MLB Stats API; fall back to top-9 by season PA per team.
  3. Build pregame bullpen queues from active rosters and prior role/rest evidence.
  4. Run simulate_game for n_sims.
  5. Compute market probs + EV + Kelly + percentiles, write rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from backend.data.mlb_api import fetch_lineup
from backend.data.odds_api import DEFAULT_BOOKS
from backend.db import engine
from backend.log import setup_logging
from backend.strategy import WEATHER_ENABLED
from backend.team_mappings import normalize_team
from v2.bayesian._common import POSTERIORS_DIR
from v2.markets.probs import paired_market_quotes
from v2.markets.writer import (
    append_season,
    build_game_rows,
    posterior_age_days,
    write_daily,
)
from v2.simulator import (
    BullpenQueue,
    GameInputs,
    load_advancement_table,
    load_out_subtype_table,
    load_posterior_draws,
    simulate_game,
)
from v2.simulator.bullpen import LiveQueueContext, PitchingPlan, apply_pitching_plan, build_queues_live
from v2.simulator.posteriors import posterior_provenance
from v2.simulator.uncertainty import win_probability_uncertainty

log = logging.getLogger(__name__)


# Posterior realizations; parameter bands are published only above MC resolution.
N_DRAWS = 30

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"

# Park posteriors are trained on Statcast abbreviations, while live games use
# MLB API abbreviations. Unknown venues otherwise silently get a zero offset.
STATCAST_VENUE_CODES = {
    "ARI": "AZ", "CHW": "CWS", "KCR": "KC", "SDP": "SD",
    "SFG": "SF", "TBR": "TB", "WSN": "WSH",
}


@dataclass
class GameContext:
    game_pk: int
    game_date: pd.Timestamp
    start_time: pd.Timestamp | None
    home_team: str
    away_team: str
    home_starter_id: int | None
    away_starter_id: int | None
    home_starter_name: str | None
    away_starter_name: str | None
    home_starter_throws: str
    away_starter_throws: str
    home_odds: dict | None
    away_odds: dict | None
    wind_speed_mph: float | None = None
    wind_out_component: float | None = None
    temp_f: float | None = None
    is_dome: bool = False


def lineup_hash(lineup: dict[str, list[int]]) -> str:
    """Hash each side and its batting order, including incomplete posted lineups."""
    ordered = (tuple(lineup.get("home", [])), tuple(lineup.get("away", [])))
    return hashlib.sha256(repr(ordered).encode()).hexdigest()[:16]


def is_started(start_time, now: pd.Timestamp) -> bool:
    """True once first pitch has passed. A null/TBD start_time is treated as
    not-started: we can't prove it began, so don't freeze it."""
    if start_time is None or pd.isna(start_time):
        return False
    ts = pd.Timestamp(start_time)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts <= now


def fetch_games_for_date(date: str) -> pd.DataFrame:
    q = text("SELECT game_pk, game_date, start_time, home_team, away_team FROM games WHERE game_date = :d")
    with engine.begin() as conn:
        return pd.read_sql(q, conn, params={"d": date})


def fetch_starters(game_pks: list[int]) -> pd.DataFrame:
    if not game_pks:
        return pd.DataFrame()
    q = text(
        "SELECT game_pk, team, pitcher_name, pitcher_id, handedness, is_home "
        "FROM probable_starters WHERE game_pk = ANY(:ids)"
    )
    with engine.begin() as conn:
        return pd.read_sql(q, conn, params={"ids": game_pks})


def fetch_odds(
    game_pks: list[int],
    books: tuple[str, ...] = DEFAULT_BOOKS,
) -> pd.DataFrame:
    if not game_pks:
        return pd.DataFrame()
    q = text(
        "SELECT game_pk, team, book, moneyline, spread, spread_odds, total, "
        "total_over_odds, total_under_odds, scraped_at "
        "FROM odds WHERE game_pk = ANY(:ids) AND book = ANY(:books) ORDER BY scraped_at DESC"
    )
    with engine.begin() as conn:
        df = pd.read_sql(q, conn, params={"ids": game_pks, "books": list(books)})
    if df.empty:
        return df
    return df.drop_duplicates(subset=["game_pk", "team", "book"], keep="first")


def _odds_package(rows: pd.DataFrame) -> dict | None:
    if rows.empty:
        return None
    rank = {book: i for i, book in enumerate(DEFAULT_BOOKS)}
    records = rows.to_dict("records")
    records.sort(key=lambda row: rank.get(row["book"], len(rank)))
    return {**records[0], "offers": records}


def fetch_weather(game_pks: list[int]) -> pd.DataFrame:
    if not game_pks:
        return pd.DataFrame()
    q = text(
        "SELECT game_pk, wind_speed_mph, wind_out_component, temp_f, is_dome "
        "FROM weather WHERE game_pk = ANY(:ids)"
    )
    with engine.begin() as conn:
        return pd.read_sql(q, conn, params={"ids": game_pks})


def build_contexts(date: str) -> list[GameContext]:
    games = fetch_games_for_date(date)
    if games.empty:
        return []
    starters = fetch_starters(games["game_pk"].tolist())
    odds = fetch_odds(games["game_pk"].tolist())
    weather = fetch_weather(games["game_pk"].tolist())

    contexts = []
    for _, g in games.iterrows():
        gp = int(g.game_pk)
        s_home = starters[(starters.game_pk == gp) & (starters.is_home == True)]  # noqa: E712
        s_away = starters[(starters.game_pk == gp) & (starters.is_home == False)]  # noqa: E712
        o_home = odds[(odds.game_pk == gp) & (odds.team == g.home_team)]
        o_away = odds[(odds.game_pk == gp) & (odds.team == g.away_team)]
        wx = weather[weather.game_pk == gp] if not weather.empty else weather
        wx_row = wx.iloc[0] if len(wx) else None
        contexts.append(
            GameContext(
                game_pk=gp,
                game_date=pd.Timestamp(g.game_date),
                start_time=pd.Timestamp(g.start_time) if pd.notna(g.start_time) else None,
                home_team=g.home_team,
                away_team=g.away_team,
                home_starter_id=int(s_home.iloc[0].pitcher_id) if len(s_home) and pd.notna(s_home.iloc[0].pitcher_id) else None,
                away_starter_id=int(s_away.iloc[0].pitcher_id) if len(s_away) and pd.notna(s_away.iloc[0].pitcher_id) else None,
                home_starter_name=s_home.iloc[0].pitcher_name if len(s_home) else None,
                away_starter_name=s_away.iloc[0].pitcher_name if len(s_away) else None,
                home_starter_throws=(s_home.iloc[0].handedness if len(s_home) and pd.notna(s_home.iloc[0].handedness) else "R"),
                away_starter_throws=(s_away.iloc[0].handedness if len(s_away) and pd.notna(s_away.iloc[0].handedness) else "R"),
                home_odds=_odds_package(o_home),
                away_odds=_odds_package(o_away),
                wind_speed_mph=float(wx_row.wind_speed_mph) if wx_row is not None and pd.notna(wx_row.wind_speed_mph) else None,
                wind_out_component=float(wx_row.wind_out_component) if wx_row is not None and pd.notna(wx_row.wind_out_component) else None,
                temp_f=float(wx_row.temp_f) if wx_row is not None and pd.notna(wx_row.temp_f) else None,
                is_dome=bool(wx_row.is_dome) if wx_row is not None and pd.notna(wx_row.is_dome) else False,
            )
        )
    return contexts


def load_cache_for_year(year: int) -> pd.DataFrame:
    """Load cached batting appearances and pitcher handedness."""
    path = CACHE_DIR / f"statcast_{year}.parquet"
    return pd.read_parquet(path, columns=[
        "game_pk", "batter", "pitcher", "inning", "inning_topbot",
        "at_bat_number", "pitch_number", "events", "home_team", "away_team",
        "p_throws", "game_date",
    ])


def top9_batters_by_team(cache: pd.DataFrame) -> dict[str, list[int]]:
    """Per team, top 9 batters by total PAs in the cache (PA-source = terminating pitches)."""
    pa = cache[cache["events"].notna()].copy()
    pa["bat_team"] = pd.Series(np.where(pa["inning_topbot"] == "Top", pa["away_team"], pa["home_team"]), index=pa.index).map(normalize_team)
    counts = pa.groupby(["bat_team", "batter"]).size().reset_index(name="n")
    out: dict[str, list[int]] = {}
    for team, grp in counts.groupby("bat_team"):
        top = grp.nlargest(9, "n")
        out[team] = top["batter"].astype(np.int64).tolist()
    return out


def fetch_lineups_for_games(game_pks: list[int]) -> dict[int, dict[str, list[int]]]:
    """Per game_pk, fetch posted home/away batting orders from MLB Stats API.

    Empty lists for sides where the lineup hasn't posted yet. Errors are
    swallowed per-game and yield empty lists so the caller can fall back.
    """
    out: dict[str, dict[str, list[int]]] = {}
    for gp in game_pks:
        try:
            out[gp] = fetch_lineup(gp)
        except Exception as e:
            log.warning(f"{gp} failed, falling back: {e}")
            out[gp] = {"home": [], "away": []}
    return out


def p_throws_for_pitchers(cache: pd.DataFrame, pitcher_ids: list[int]) -> dict[int, str]:
    """Read p_throws from the cache for a set of pitcher_ids."""
    if not pitcher_ids:
        return {}
    sub = cache[cache["pitcher"].isin(pitcher_ids)][["pitcher", "p_throws"]].drop_duplicates("pitcher")
    return {int(r.pitcher): str(r.p_throws) for _, r in sub.iterrows()}


def _resolve_lineup(
    live: list[int],
    fallback: list[int],
) -> tuple[list[int], str]:
    """Use live posted lineup if it's a complete 9 of non-zero ids; else fallback."""
    if len(live) == 9 and len(set(live)) == 9 and all(int(b) > 0 for b in live):
        return [int(b) for b in live], "live"
    padded = (list(fallback) + [0] * 9)[:9]
    return padded, "top9"


def _resolve_queue(game_pk: int, side: str, live: dict, starter: int) -> tuple[BullpenQueue, str]:
    if (game_pk, side) in live:
        return live[(game_pk, side)], "live"
    return BullpenQueue(starter=starter, relievers=[0], starter_role=1), "neutral"


def weather_scalars(ctx: GameContext) -> tuple[float, float]:
    """(wind_signal, temp_c) for the sim. Zero when disabled, dome, or missing.

    wind_signal = wind_speed_mph * signed out-component; temp_c = temp_f - 70.
    """
    if not WEATHER_ENABLED or ctx.is_dome:
        return 0.0, 0.0
    wind = 0.0
    if ctx.wind_speed_mph is not None and ctx.wind_out_component is not None:
        wind = float(ctx.wind_speed_mph) * float(ctx.wind_out_component)
    temp_c = float(ctx.temp_f) - 70.0 if ctx.temp_f is not None else 0.0
    return wind, temp_c


def build_inputs(
    ctx: GameContext,
    live_home: list[int],
    live_away: list[int],
    fallback_lineups_by_team: dict[str, list[int]],
    live_queues: dict[tuple[int, str], BullpenQueue],
    throws_lookup: dict[int, str],
) -> tuple[GameInputs, str, str]:
    """Build GameInputs + lineup_tag + queue_source.

    lineup_tag ∈ {live, top9, mixed}. queue_source aggregates the two sides:
    if both match, that value; otherwise 'mixed'.
    """
    home_lineup, home_tag = _resolve_lineup(live_home, fallback_lineups_by_team.get(normalize_team(ctx.home_team), []))
    away_lineup, away_tag = _resolve_lineup(live_away, fallback_lineups_by_team.get(normalize_team(ctx.away_team), []))
    lineup_tag = home_tag if home_tag == away_tag else "mixed"

    home_queue, home_qsrc = _resolve_queue(
        ctx.game_pk, "home", live_queues, ctx.home_starter_id or 0,
    )
    away_queue, away_qsrc = _resolve_queue(
        ctx.game_pk, "away", live_queues, ctx.away_starter_id or 0,
    )
    queue_source = home_qsrc if home_qsrc == away_qsrc else "mixed"

    # Throws: starter handedness from probable_starters, relievers from cache.
    home_throws = dict(throws_lookup)
    away_throws = dict(throws_lookup)
    if ctx.home_starter_id:
        home_throws[ctx.home_starter_id] = ctx.home_starter_throws or "R"
    if ctx.away_starter_id:
        away_throws[ctx.away_starter_id] = ctx.away_starter_throws or "R"

    wind_signal, temp_c = weather_scalars(ctx)
    inputs = GameInputs(
        home_lineup=np.array(home_lineup, dtype=np.int64),
        away_lineup=np.array(away_lineup, dtype=np.int64),
        home_queue=home_queue,
        away_queue=away_queue,
        venue=STATCAST_VENUE_CODES.get(ctx.home_team, ctx.home_team),
        home_p_throws_lookup=home_throws,
        away_p_throws_lookup=away_throws,
        wind_signal=wind_signal,
        temp_c=temp_c,
    )
    return inputs, lineup_tag, queue_source


def score(
    date: str,
    n_sims: int = 10000,
    write: bool = True,
    seed: int = 0,
    game_pks: list[int] | None = None,
    update_season: bool = True,
    freeze_started: bool = True,
    posteriors_dir: Path = POSTERIORS_DIR,
    pitching_plans: list[PitchingPlan] | None = None,
) -> pd.DataFrame:
    """Score games for a date.

    Args:
        date: YYYY-MM-DD slate to score.
        n_sims: total sims per game (split across N_DRAWS posterior draws).
        write: write rows to model_outputs (and model_outputs_season if update_season).
        seed: rng seed.
        game_pks: if set, only score these game_pks (others on the date are
            untouched in model_outputs). Used by the hourly lineup refresh
            so a single posted lineup doesn't rewrite the whole slate.
        update_season: mirror pregame forecasts to model_outputs_season.
            Production refreshes keep this enabled so the displayed and graded
            forecasts agree at first pitch.
        freeze_started: if True (the production default), games whose start_time
            has passed are dropped before scoring, so neither model_outputs nor
            model_outputs_season can be rewritten once a game is underway. The
            pick is frozen at the last pre-first-pitch score. False permits
            read-only replay with pre-cutoff artifacts; historical writes are
            rejected and hindsight replays cannot pass the acceptance gate.
    """
    log.info(f"loading {N_DRAWS} posterior draws + tables...")
    rng = np.random.default_rng(seed)
    draws = load_posterior_draws(rng, K=N_DRAWS, posteriors_dir=posteriors_dir)
    provenance = posterior_provenance(posteriors_dir)
    if provenance["training_max_date"] >= date:
        raise ValueError("Training data reaches the prediction date; use a pregame posterior snapshot")
    adv = load_advancement_table()
    if adv.training_max_date is None or adv.training_max_date >= date:
        raise ValueError("Advancement tables need a known training cutoff before the prediction date")
    sub_table = load_out_subtype_table()
    if sub_table.training_max_date != adv.training_max_date:
        raise ValueError("Simulator tables have inconsistent training cutoffs; rebuild them together")
    age = posterior_age_days(posteriors_dir=posteriors_dir)

    contexts = build_contexts(date)
    if not contexts:
        log.info(f"no games on {date}")
        return pd.DataFrame()
    if game_pks is not None:
        wanted = set(int(p) for p in game_pks)
        contexts = [c for c in contexts if int(c.game_pk) in wanted]
        if not contexts:
            log.info(f"none of the requested game_pks scheduled on {date}")
            return pd.DataFrame()
    if freeze_started:
        now = pd.Timestamp.now(tz="UTC")
        started = {c.game_pk for c in contexts if is_started(c.start_time, now)}
        if started:
            log.info(f"freezing {len(started)} already-started games (not re-scored): {sorted(started)}")
            contexts = [c for c in contexts if c.game_pk not in started]
        if not contexts:
            log.info(f"all games on {date} already started; nothing to score")
            return pd.DataFrame()
    log.info(f"{len(contexts)} games on {date}")

    year = pd.Timestamp(date).year
    cache = load_cache_for_year(year)
    cache = cache[pd.to_datetime(cache["game_date"]) < pd.Timestamp(date)]
    fallback_lineups = top9_batters_by_team(cache)

    # Live queues use pregame role and rest evidence plus the active roster.
    live_q_contexts: list[LiveQueueContext] = []
    for c in contexts:
        if c.home_starter_id:
            live_q_contexts.append(LiveQueueContext(c.game_pk, "home", c.home_team, c.home_starter_id))
        if c.away_starter_id:
            live_q_contexts.append(LiveQueueContext(c.game_pk, "away", c.away_team, c.away_starter_id))
    try:
        live_queues = build_queues_live(pd.Timestamp(date).date(), live_q_contexts)
    except Exception as e:
        log.warning(f"build_queues_live failed ({e}); using neutral pitching fallback")
        live_queues = {}

    live_lineups = fetch_lineups_for_games([c.game_pk for c in contexts])

    all_pitchers: set[int] = set()
    for q in list(live_queues.values()):
        all_pitchers.add(q.starter)
        all_pitchers.update(q.relievers)
    throws_lookup = p_throws_for_pitchers(cache, list(all_pitchers))

    n_per_draw = n_sims // N_DRAWS
    if n_per_draw < 2:
        raise ValueError(f"n_sims must be at least {2 * N_DRAWS}")
    actual_n_sims = n_per_draw * N_DRAWS
    if actual_n_sims != n_sims:
        log.info(f"rounding n_sims {n_sims} -> {actual_n_sims} ({N_DRAWS} draws × {n_per_draw} sims)")

    all_rows: list[dict] = []
    flagged = 0
    for ctx in contexts:
        live = live_lineups.get(ctx.game_pk, {"home": [], "away": []})
        inputs, lineup_tag, queue_source = build_inputs(
            ctx, live["home"], live["away"], fallback_lineups,
            live_queues, throws_lookup,
        )
        accepted_plans = []
        for side in ("home", "away"):
            queue = getattr(inputs, f"{side}_queue")
            matches = [p for p in (pitching_plans or []) if p.game_pk == ctx.game_pk and p.side == side]
            context = LiveQueueContext(ctx.game_pk, side, getattr(ctx, f"{side}_team"), queue.starter)
            if len(matches) == 1 and matches[0].valid_for(context, ctx.start_time):
                queue = apply_pitching_plan(queue, matches[0])
                setattr(inputs, f"{side}_queue", queue)
                accepted_plans.append(asdict(matches[0]))
            elif matches:
                # Contradictory or stale reports do not extend workloads.
                queue.usage_known = False
                queue.starter_role = 1
                queue.workloads[queue.starter] = (3,)
        lineup_source = f"lineup_{lineup_tag}+queue_{queue_source}"
        lhash = lineup_hash(live)
        h_chunks: list[np.ndarray] = []
        a_chunks: list[np.ndarray] = []
        per_draw_home_wp: list[float] = []
        for pm_k in draws:
            h_k, a_k = simulate_game(rng, pm_k, adv, sub_table, inputs, n_sims=n_per_draw)
            h_chunks.append(h_k)
            a_chunks.append(a_k)
            margin = h_k - a_k
            wp_k = float((margin > 0).mean()) + 0.5 * float((margin == 0).mean())
            per_draw_home_wp.append(wp_k)
        h = np.concatenate(h_chunks)
        a = np.concatenate(a_chunks)
        uncertainty = win_probability_uncertainty(per_draw_home_wp, n_per_draw)
        home_wp_p10 = uncertainty["p10"]
        home_wp_p90 = uncertainty["p90"]
        snapshot = {
            **provenance, "tables_training_max_date": adv.training_max_date,
            "inputs_as_of": pd.Timestamp.now(tz="UTC").isoformat(),
            "uncertainty": uncertainty, "pitching_plans": accepted_plans,
            "home_lineup": inputs.home_lineup.tolist(), "away_lineup": inputs.away_lineup.tolist(),
            "home_queue": asdict(inputs.home_queue), "away_queue": asdict(inputs.away_queue),
            "opener": inputs.home_queue.starter_role == 1 or inputs.away_queue.starter_role == 1,
            "home_bp_outs_2d": inputs.home_queue.team_outs_2d,
            "away_bp_outs_2d": inputs.away_queue.team_outs_2d,
        }
        snapshot["market_pairs"] = paired_market_quotes(ctx.home_odds, ctx.away_odds)

        rows = build_game_rows(
            game_pk=ctx.game_pk,
            game_date=ctx.game_date,
            start_time=ctx.start_time,
            home_team=ctx.home_team,
            away_team=ctx.away_team,
            home_starter=ctx.home_starter_name,
            away_starter=ctx.away_starter_name,
            home_runs=h,
            away_runs=a,
            home_odds=ctx.home_odds,
            away_odds=ctx.away_odds,
            lineup_source=lineup_source,
            lineups_locked=False,
            posterior_age_days=age,
            home_wp_p10=home_wp_p10,
            home_wp_p90=home_wp_p90,
            lineup_hash=lhash,
            starters_known=ctx.home_starter_id is not None and ctx.away_starter_id is not None,
            lineups_live=lineup_tag == "live",
            pitching_usage_known=inputs.home_queue.usage_known and inputs.away_queue.usage_known,
            prediction_context=snapshot,
        )
        all_rows.extend(rows)
        for r in rows:
            if r["ev_flag"] != "No Play" or r["run_line_ev_flag"] != "No Play" or r["total_play"] != "No Play":
                flagged += 1
        log.info(
            f"game {ctx.game_pk} {ctx.away_team}@{ctx.home_team}: "
            f"xR {rows[1]['expected_runs']:.2f} / {rows[0]['expected_runs']:.2f}, "
            f"home_wp {rows[0]['win_prob']:.3f}; parameter band {uncertainty['status']}"
        )

    if write and not freeze_started:
        raise ValueError("Historical replay is read-only; started-game forecasts cannot be replaced")
    if write and all_rows:
        now = pd.Timestamp.now(tz="UTC")
        all_rows = [r for r in all_rows if not is_started(r["start_time"], now)]
        write_daily(pd.Timestamp(date), all_rows)
        if update_season:
            append_season(all_rows)
            log.info(f"wrote {len(all_rows)} rows to model_outputs + season")
        else:
            log.info(f"wrote {len(all_rows)} rows to model_outputs (season skipped)")

    log.info(f"{flagged} +EV flags across {len(all_rows)} rows; posterior_age_days={age}")
    return pd.DataFrame(all_rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--n-sims", type=int, default=10000)
    p.add_argument("--no-write", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--posteriors-dir", type=Path, default=POSTERIORS_DIR)
    p.add_argument("--pitching-plans", type=Path, help="JSON list of explicitly confirmed pregame reports")
    p.add_argument("--export", type=Path, help="Save a local frozen forecast for later paired acceptance")
    args = p.parse_args()
    plans = [PitchingPlan(**p) for p in json.loads(args.pitching_plans.read_text())] if args.pitching_plans else []
    rows = score(args.date, n_sims=args.n_sims, write=not args.no_write, seed=args.seed,
                 posteriors_dir=args.posteriors_dir, pitching_plans=plans)
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        rows.to_json(args.export, orient="records", date_format="iso", indent=2)


if __name__ == "__main__":
    setup_logging()
    main()
