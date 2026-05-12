"""Phase 5 entrypoint: score games for a date and write to model_outputs_v2.

Usage:
    python -m v2.pipeline.score_games --date 2025-09-15 --n-sims 10000

Steps per game:
  1. Read probable_starters and odds from Supabase.
  2. Fetch posted lineups via MLB Stats API; fall back to top-9 by season PA per team.
  3. Build bullpen queues (from statcast cache if game already played; else fallback).
  4. Run simulate_game for n_sims.
  5. Compute market probs + EV + Kelly + percentiles, write rows.
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from backend.data.mlb_api import fetch_lineup
from backend.db import engine
from v2.markets.writer import (
    append_season,
    build_game_rows,
    posterior_age_days,
    write_daily,
)
from v2.simulator import (
    AdvancementTable,
    BullpenQueue,
    GameInputs,
    PosteriorMeans,
    build_queues_from_cache,
    load_advancement_table,
    load_out_subtype_table,
    load_posterior_draws,
    simulate_game,
)
from v2.simulator.bullpen import LiveQueueContext, build_queues_live


# K posterior draws per game. ~30 gives stable p10/p90 win-prob bands without
# per-game cost ballooning (~30 × 50ms PosteriorMeans assembly = 1.5s overhead).
N_DRAWS = 30

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"


@dataclass
class GameContext:
    game_pk: int
    game_date: pd.Timestamp
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


def fetch_games_for_date(date: str) -> pd.DataFrame:
    q = text("SELECT game_pk, game_date, home_team, away_team FROM games WHERE game_date = :d")
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


def fetch_odds(game_pks: list[int], book: str = "draftkings") -> pd.DataFrame:
    if not game_pks:
        return pd.DataFrame()
    q = text(
        "SELECT game_pk, team, book, moneyline, spread, spread_odds, total, "
        "total_over_odds, total_under_odds "
        "FROM odds WHERE game_pk = ANY(:ids) AND book = :b"
    )
    with engine.begin() as conn:
        df = pd.read_sql(q, conn, params={"ids": game_pks, "b": book})
    if df.empty:
        return df
    return df.drop_duplicates(subset=["game_pk", "team"], keep="first")


def build_contexts(date: str) -> list[GameContext]:
    games = fetch_games_for_date(date)
    if games.empty:
        return []
    starters = fetch_starters(games["game_pk"].tolist())
    odds = fetch_odds(games["game_pk"].tolist())

    contexts = []
    for _, g in games.iterrows():
        gp = int(g.game_pk)
        s_home = starters[(starters.game_pk == gp) & (starters.is_home == True)]  # noqa: E712
        s_away = starters[(starters.game_pk == gp) & (starters.is_home == False)]  # noqa: E712
        o_home = odds[(odds.game_pk == gp) & (odds.team == g.home_team)]
        o_away = odds[(odds.game_pk == gp) & (odds.team == g.away_team)]
        contexts.append(
            GameContext(
                game_pk=gp,
                game_date=pd.Timestamp(g.game_date),
                home_team=g.home_team,
                away_team=g.away_team,
                home_starter_id=int(s_home.iloc[0].pitcher_id) if len(s_home) and pd.notna(s_home.iloc[0].pitcher_id) else None,
                away_starter_id=int(s_away.iloc[0].pitcher_id) if len(s_away) and pd.notna(s_away.iloc[0].pitcher_id) else None,
                home_starter_name=s_home.iloc[0].pitcher_name if len(s_home) else None,
                away_starter_name=s_away.iloc[0].pitcher_name if len(s_away) else None,
                home_starter_throws=(s_home.iloc[0].handedness if len(s_home) and pd.notna(s_home.iloc[0].handedness) else "R"),
                away_starter_throws=(s_away.iloc[0].handedness if len(s_away) and pd.notna(s_away.iloc[0].handedness) else "R"),
                home_odds=o_home.iloc[0].to_dict() if len(o_home) else None,
                away_odds=o_away.iloc[0].to_dict() if len(o_away) else None,
            )
        )
    return contexts


def load_cache_for_year(year: int) -> pd.DataFrame:
    """Load minimal columns from the statcast cache for lineup + queue derivation."""
    path = CACHE_DIR / f"statcast_{year}.parquet"
    return pd.read_parquet(path, columns=[
        "game_pk", "batter", "pitcher", "inning", "inning_topbot",
        "at_bat_number", "pitch_number", "events", "home_team", "away_team",
        "p_throws",
    ])


def top9_batters_by_team(cache: pd.DataFrame) -> dict[str, list[int]]:
    """Per team, top 9 batters by total PAs in the cache (PA-source = terminating pitches)."""
    pa = cache[cache["events"].notna()].copy()
    pa["bat_team"] = np.where(pa["inning_topbot"] == "Top", pa["away_team"], pa["home_team"])
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
            print(f"  [fetch_lineup] {gp} failed, falling back: {e}")
            out[gp] = {"home": [], "away": []}
    return out


def reliever_queue_for_team(cache: pd.DataFrame, team: str, max_n: int = 6) -> list[int]:
    """Pitchers who appeared in relief (not the inning-1 starter) for `team`, ranked by appearances."""
    pa = cache[cache["events"].notna()].copy()
    pa["pitch_team"] = np.where(pa["inning_topbot"] == "Top", pa["home_team"], pa["away_team"])
    pa = pa[pa["pitch_team"] == team]
    if pa.empty:
        return []
    inn1 = pa[pa["inning"] == 1]
    starter_ids = set(inn1["pitcher"].astype(np.int64).unique().tolist())
    relievers = pa[~pa["pitcher"].isin(starter_ids)]
    if relievers.empty:
        return []
    counts = relievers.groupby("pitcher").size().sort_values(ascending=False)
    return counts.head(max_n).index.astype(np.int64).tolist()


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
    if len(live) == 9 and all(int(b) > 0 for b in live):
        return [int(b) for b in live], "live"
    padded = (list(fallback) + [0] * 9)[:9]
    return padded, "top9"


def _resolve_queue(
    game_pk: int,
    side: str,
    live: dict[tuple[int, str], BullpenQueue],
    cache: dict[tuple[int, str], BullpenQueue],
    stub_starter: int,
    stub_relievers: list[int],
) -> tuple[BullpenQueue, str]:
    """Pick queue for one side. Returns (queue, source) where source ∈ live|cache|stub."""
    key = (game_pk, side)
    if key in live:
        return live[key], "live"
    if key in cache:
        return cache[key], "cache"
    return BullpenQueue(starter=stub_starter, relievers=stub_relievers[:5]), "stub"


def build_inputs(
    ctx: GameContext,
    live_home: list[int],
    live_away: list[int],
    fallback_lineups_by_team: dict[str, list[int]],
    live_queues: dict[tuple[int, str], BullpenQueue],
    cache_queues: dict[tuple[int, str], BullpenQueue],
    relievers_by_team: dict[str, list[int]],
    throws_lookup: dict[int, str],
) -> tuple[GameInputs, str, str]:
    """Build GameInputs + lineup_tag + queue_source.

    lineup_tag ∈ {live, top9, mixed}. queue_source aggregates the two sides:
    if both match, that value; otherwise 'mixed'.
    """
    home_lineup, home_tag = _resolve_lineup(live_home, fallback_lineups_by_team.get(ctx.home_team, []))
    away_lineup, away_tag = _resolve_lineup(live_away, fallback_lineups_by_team.get(ctx.away_team, []))
    lineup_tag = home_tag if home_tag == away_tag else "mixed"

    home_queue, home_qsrc = _resolve_queue(
        ctx.game_pk, "home", live_queues, cache_queues,
        ctx.home_starter_id or 0, relievers_by_team.get(ctx.home_team, []),
    )
    away_queue, away_qsrc = _resolve_queue(
        ctx.game_pk, "away", live_queues, cache_queues,
        ctx.away_starter_id or 0, relievers_by_team.get(ctx.away_team, []),
    )
    queue_source = home_qsrc if home_qsrc == away_qsrc else "mixed"

    # Throws: starter handedness from probable_starters, relievers from cache.
    home_throws = dict(throws_lookup)
    away_throws = dict(throws_lookup)
    if ctx.home_starter_id:
        home_throws[ctx.home_starter_id] = ctx.home_starter_throws or "R"
    if ctx.away_starter_id:
        away_throws[ctx.away_starter_id] = ctx.away_starter_throws or "R"

    inputs = GameInputs(
        home_lineup=np.array(home_lineup, dtype=np.int64),
        away_lineup=np.array(away_lineup, dtype=np.int64),
        home_queue=home_queue,
        away_queue=away_queue,
        venue=ctx.home_team,
        home_p_throws_lookup=home_throws,
        away_p_throws_lookup=away_throws,
    )
    return inputs, lineup_tag, queue_source


def score(date: str, n_sims: int = 10000, write: bool = True, seed: int = 0) -> pd.DataFrame:
    """Score all games for a date. Returns the DataFrame of rows written."""
    print(f"[score_games] loading {N_DRAWS} posterior draws + tables...")
    rng = np.random.default_rng(seed)
    draws = load_posterior_draws(rng, K=N_DRAWS)
    adv = load_advancement_table()
    sub_table = load_out_subtype_table()
    age = posterior_age_days()

    contexts = build_contexts(date)
    if not contexts:
        print(f"[score_games] no games on {date}")
        return pd.DataFrame()
    print(f"[score_games] {len(contexts)} games on {date}")

    year = pd.Timestamp(date).year
    cache = load_cache_for_year(year)
    fallback_lineups = top9_batters_by_team(cache)
    relievers = {team: reliever_queue_for_team(cache, team) for team in {c.home_team for c in contexts} | {c.away_team for c in contexts}}
    cache_queues = build_queues_from_cache(year)

    # Live queues: rest-aware, built from pitcher_workload + active roster.
    live_q_contexts: list[LiveQueueContext] = []
    for c in contexts:
        if c.home_starter_id:
            live_q_contexts.append(LiveQueueContext(c.game_pk, "home", c.home_team, c.home_starter_id))
        if c.away_starter_id:
            live_q_contexts.append(LiveQueueContext(c.game_pk, "away", c.away_team, c.away_starter_id))
    try:
        live_queues = build_queues_live(pd.Timestamp(date).date(), live_q_contexts)
    except Exception as e:
        print(f"[score_games] build_queues_live failed ({e}); falling back to cache")
        live_queues = {}

    live_lineups = fetch_lineups_for_games([c.game_pk for c in contexts])

    all_pitchers: set[int] = set()
    for q in list(cache_queues.values()) + list(live_queues.values()):
        all_pitchers.add(q.starter)
        all_pitchers.update(q.relievers)
    for relievers_list in relievers.values():
        all_pitchers.update(relievers_list)
    throws_lookup = p_throws_for_pitchers(cache, list(all_pitchers))

    n_per_draw = max(1, n_sims // N_DRAWS)
    actual_n_sims = n_per_draw * N_DRAWS
    if actual_n_sims != n_sims:
        print(f"[score_games] rounding n_sims {n_sims} -> {actual_n_sims} ({N_DRAWS} draws × {n_per_draw} sims)")

    all_rows: list[dict] = []
    flagged = 0
    for ctx in contexts:
        live = live_lineups.get(ctx.game_pk, {"home": [], "away": []})
        inputs, lineup_tag, queue_source = build_inputs(
            ctx, live["home"], live["away"], fallback_lineups,
            live_queues, cache_queues, relievers, throws_lookup,
        )
        lineup_source = f"lineup_{lineup_tag}+queue_{queue_source}"
        _hash_input = sorted(live.get("home", [])) + sorted(live.get("away", []))
        lhash = hashlib.sha1(str(_hash_input).encode()).hexdigest()[:16]
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
        home_wp_p10 = round(float(np.quantile(per_draw_home_wp, 0.10)), 4)
        home_wp_p90 = round(float(np.quantile(per_draw_home_wp, 0.90)), 4)
        rows = build_game_rows(
            game_pk=ctx.game_pk,
            game_date=ctx.game_date,
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
        )
        all_rows.extend(rows)
        for r in rows:
            if r["ev_flag"] != "No Play" or r["run_line_ev_flag"] != "No Play" or r["total_play"] != "No Play":
                flagged += 1
        print(
            f"  game {ctx.game_pk} {ctx.away_team}@{ctx.home_team}: "
            f"xR {rows[1]['expected_runs']:.2f} / {rows[0]['expected_runs']:.2f}, "
            f"home_wp {rows[0]['win_prob']:.3f} [{home_wp_p10:.3f}-{home_wp_p90:.3f}]"
        )

    if write and all_rows:
        write_daily(pd.Timestamp(date), all_rows)
        append_season(all_rows)
        print(f"[score_games] wrote {len(all_rows)} rows to model_outputs_v2 + season_v2")

    print(f"[score_games] {flagged} +EV flags across {len(all_rows)} rows; posterior_age_days={age}")
    return pd.DataFrame(all_rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--n-sims", type=int, default=10000)
    p.add_argument("--no-write", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    score(args.date, n_sims=args.n_sims, write=not args.no_write, seed=args.seed)


if __name__ == "__main__":
    main()
