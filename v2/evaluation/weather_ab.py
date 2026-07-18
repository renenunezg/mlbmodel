"""Non-destructive weather A/B backtest.

Scores each completed game in a window twice on identical inputs (weather off vs
on) and reports whether the weather shift improves the totals point estimate and
edge-bucket ROI. Writes NOTHING to the DB. The only variable between the two
passes is the per-game weather logit shift: lineups, queues, posteriors, and the
RNG stream are shared (each posterior draw is re-seeded identically for off/on).

Inputs use actual game lineups and bullpen queues reconstructed from the
Statcast cache. Games with incomplete cache coverage fall back to top9-by-PA
lineups and team-level reliever stubs.

    env/bin/python -m v2.evaluation.weather_ab --start 2026-03-26 --end 2026-06-21
"""
from __future__ import annotations

import argparse
import dataclasses
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text

from backend.db import engine
from backend.strategy import EV_THRESHOLDS
from v2.evaluation.metrics import _american_to_decimal, _american_to_implied
from v2.pipeline.score_games import (
    N_DRAWS,
    build_contexts,
    build_inputs,
    load_cache_for_year,
    p_throws_for_pitchers,
    reliever_queue_for_team,
    top9_batters_by_team,
)
from v2.markets.probs import market_probs
from v2.simulator import (
    build_queues_from_cache,
    load_advancement_table,
    load_out_subtype_table,
    load_posterior_draws,
    simulate_game,
)

TOT_THRESH = EV_THRESHOLDS["totals"]
BUCKETS = [-1, 0.0, 0.02, 0.045, 0.065, 0.10, 0.15, 1.0]


def _daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _on_scalars(ctx) -> tuple[float, float]:
    """Weather scalars regardless of the global flag; dome/missing -> (0, 0)."""
    if ctx.is_dome:
        return 0.0, 0.0
    wind = 0.0
    if ctx.wind_speed_mph is not None and ctx.wind_out_component is not None:
        wind = float(ctx.wind_speed_mph) * float(ctx.wind_out_component)
    temp_c = float(ctx.temp_f) - 70.0 if ctx.temp_f is not None else 0.0
    return wind, temp_c


def _lineups_from_cache(cache: pd.DataFrame) -> dict[int, dict[str, list[int]]]:
    pa = cache[cache["events"].notna()].sort_values(
        ["game_pk", "at_bat_number", "pitch_number"]
    )
    lineups: dict[int, dict[str, list[int]]] = {}
    for game_pk, game_pa in pa.groupby("game_pk", sort=False):
        sides: dict[str, list[int]] = {}
        for side, half in (("away", "Top"), ("home", "Bot")):
            order: list[int] = []
            for batter in game_pa.loc[game_pa["inning_topbot"] == half, "batter"]:
                batter_id = int(batter)
                if batter_id not in order:
                    order.append(batter_id)
                if len(order) == 9:
                    break
            sides[side] = order
        lineups[int(game_pk)] = sides
    return lineups


def _sim_total(draws, adv, sub, inputs, n_per_draw, seed) -> tuple[np.ndarray, np.ndarray]:
    h_chunks, a_chunks = [], []
    for k, pm_k in enumerate(draws):
        rng = np.random.default_rng(seed + k)  # identical stream per draw across off/on
        h_k, a_k = simulate_game(rng, pm_k, adv, sub, inputs, n_sims=n_per_draw)
        h_chunks.append(h_k)
        a_chunks.append(a_k)
    return np.concatenate(h_chunks), np.concatenate(a_chunks)


def _final_games(start: date, end: date) -> pd.DataFrame:
    q = text(
        "SELECT game_pk, game_date, home_score, away_score FROM games "
        "WHERE status='Final' AND game_date BETWEEN :s AND :e "
        "AND home_score IS NOT NULL AND away_score IS NOT NULL"
    )
    with engine.begin() as conn:
        g = pd.read_sql(q, conn, params={"s": start, "e": end})
    g["actual_total"] = g.home_score + g.away_score
    g["actual_margin"] = g.home_score - g.away_score
    g["home_win"] = (g.actual_margin > 0).astype(int)
    return g


def collect(start: date, end: date, n_sims: int, seed: int) -> pd.DataFrame:
    rng0 = np.random.default_rng(seed)
    draws = load_posterior_draws(rng0, K=N_DRAWS)
    adv = load_advancement_table()
    sub = load_out_subtype_table()
    n_per_draw = max(1, n_sims // N_DRAWS)

    finals = _final_games(start, end)
    final_pks = set(finals.game_pk.astype(int))

    year = start.year
    cache = load_cache_for_year(year)
    fallback_lineups = top9_batters_by_team(cache)
    game_lineups = _lineups_from_cache(cache)
    cache_queues = build_queues_from_cache(year)
    all_pitchers = set()
    for qd in cache_queues.values():
        all_pitchers.add(qd.starter)
        all_pitchers.update(qd.relievers)
    throws_lookup = p_throws_for_pitchers(cache, list(all_pitchers))

    recs = []
    for d in _daterange(start, end):
        contexts = [c for c in build_contexts(str(d)) if int(c.game_pk) in final_pks]
        if not contexts:
            continue
        teams = {c.home_team for c in contexts} | {c.away_team for c in contexts}
        relievers = {t: reliever_queue_for_team(cache, t) for t in teams}
        t0 = time.time()
        for ctx in contexts:
            actual_lineup = game_lineups.get(
                int(ctx.game_pk), {"home": [], "away": []}
            )
            inputs, lineup_source, queue_source = build_inputs(
                ctx,
                actual_lineup["home"],
                actual_lineup["away"],
                fallback_lineups,
                {},
                cache_queues,
                relievers,
                throws_lookup,
            )
            on_w, on_t = _on_scalars(ctx)
            inputs_off = dataclasses.replace(inputs, wind_signal=0.0, temp_c=0.0)
            inputs_on = dataclasses.replace(inputs, wind_signal=on_w, temp_c=on_t)

            h_off, a_off = _sim_total(draws, adv, sub, inputs_off, n_per_draw, seed)
            h_on, a_on = _sim_total(draws, adv, sub, inputs_on, n_per_draw, seed)

            line = _get(ctx.home_odds, "total")
            over_odds = _get(ctx.home_odds, "total_over_odds")
            under_odds = _get(ctx.home_odds, "total_under_odds")
            spread = _get(ctx.home_odds, "spread")
            pr_off = market_probs(h_off, a_off, line, spread)
            pr_on = market_probs(h_on, a_on, line, spread)
            recs.append({
                "game_pk": int(ctx.game_pk),
                "date": pd.Timestamp(ctx.game_date),
                "our_total_off": float((h_off + a_off).mean()),
                "our_total_on": float((h_on + a_on).mean()),
                "p_over_off": pr_off["p_over"],
                "p_over_on": pr_on["p_over"],
                "home_wp_off": pr_off["p_home_win"],
                "home_wp_on": pr_on["p_home_win"],
                "p_home_cover_off": pr_off["p_home_cover"],
                "p_home_cover_on": pr_on["p_home_cover"],
                "line": line,
                "over_odds": over_odds,
                "under_odds": under_odds,
                "spread": spread,
                "wind_signal": on_w,
                "temp_c": on_t,
                "lineup_source": lineup_source,
                "queue_source": queue_source,
            })
        print(f"[weather_ab] {d}: {len(contexts)} games in {time.time()-t0:.1f}s")

    keep = ["game_pk", "actual_total", "actual_margin", "home_win"]
    df = pd.DataFrame(recs).merge(finals[keep], on="game_pk", how="inner")
    return df


def _get(d, key):
    if d is None:
        return np.nan
    v = d.get(key)
    return v if v is not None else np.nan


def _rmse(a, b):
    return float(np.sqrt(np.nanmean((np.asarray(a) - np.asarray(b)) ** 2)))


def _totals_ledger(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Flat-stake totals ledger (no edge floor) for the off/on pass given by suffix."""
    d = df[df.actual_total.notna() & df.line.notna()
           & df.over_odds.notna() & df.under_odds.notna()
           & (df.over_odds != 0) & (df.under_odds != 0)].copy()
    po = d[f"p_over_{suffix}"]
    pu = 1.0 - po
    imp_over = _american_to_implied(d.over_odds.to_numpy())
    imp_under = _american_to_implied(d.under_odds.to_numpy())
    edge_over = po - imp_over
    edge_under = pu - imp_under
    pick_over = edge_over >= edge_under
    edge = np.where(pick_over, edge_over, edge_under)
    odds = np.where(pick_over, d.over_odds, d.under_odds)
    went_over = d.actual_total > d.line
    won = np.where(pick_over, went_over, ~went_over)
    dec = _american_to_decimal(np.asarray(odds, dtype=float))
    pnl = np.where(won, dec - 1.0, -1.0)
    return pd.DataFrame({"edge": edge, "won": won.astype(bool), "pnl": pnl})


def _report(df: pd.DataFrame) -> None:
    dl = df[df.line.notna()].copy()
    print(f"\n==== WEATHER A/B  n={len(df)} games, {df.line.notna().sum()} with a line ====")
    print(
        f"  actual lineups: {(df.lineup_source == 'live').mean():.1%}  "
        f"actual queues: {(df.queue_source == 'cache').mean():.1%}"
    )

    for label, sub in [("ALL", dl), ("WINDY |signal|>=8", dl[dl.wind_signal.abs() >= 8])]:
        if len(sub) < 20:
            continue
        print(f"\n--- {label}  (n={len(sub)}) ---")
        for suffix in ("off", "on"):
            ot = sub[f"our_total_{suffix}"]
            rmse = _rmse(ot, sub.actual_total)
            sig = ot - sub.line
            res = sub.actual_total - sub.line
            corr = pd.Series(sig.values).corr(pd.Series(res.values))
            print(f"  weather {suffix:>3}: RMSE(our_total,actual)={rmse:.3f}  "
                  f"corr(our-line, actual-line)={corr:+.3f}  meanbias={float((ot-sub.actual_total).mean()):+.2f}")

    # ML/RL neutrality: weather hits both teams, so the run difference (and thus
    # sides markets) should barely move. Confirm the disturbance is tiny and Brier
    # doesn't worsen.
    dwp = (df.home_wp_on - df.home_wp_off).abs()
    brier_off = float(((df.home_wp_off - df.home_win) ** 2).mean())
    brier_on = float(((df.home_wp_on - df.home_win) ** 2).mean())
    print("\n--- ML/RL neutrality ---")
    print(f"  |Δ home_wp| (on-off):  mean={dwp.mean():.4f}  median={dwp.median():.4f}  p95={dwp.quantile(0.95):.4f}  max={dwp.max():.4f}")
    print(f"  ML Brier (home side):  off={brier_off:.4f}  on={brier_on:.4f}  (lower=better; want ~equal)")
    rl = df[df.p_home_cover_off.notna()]
    if len(rl):
        dcov = (rl.p_home_cover_on - rl.p_home_cover_off).abs()
        cover = (rl.actual_margin > -rl.spread).astype(int)
        bc_off = float(((rl.p_home_cover_off - cover) ** 2).mean())
        bc_on = float(((rl.p_home_cover_on - cover) ** 2).mean())
        print(f"  |Δ p_home_cover|:      mean={dcov.mean():.4f}  median={dcov.median():.4f}  p95={dcov.quantile(0.95):.4f}")
        print(f"  RL Brier (cover):      off={bc_off:.4f}  on={bc_on:.4f}")

    print("\n--- totals edge-bucket ROI (flat 1u, no edge floor) ---")
    for suffix in ("off", "on"):
        led = _totals_ledger(df, suffix)
        led = led.assign(b=pd.cut(led.edge, BUCKETS))
        print(f"\n  weather {suffix.upper()}  (n={len(led)}, threshold={TOT_THRESH})")
        print(f"    {'edge bucket':>14} | {'n':>5} | {'ROI':>8} | {'win%':>6}")
        for bk, g in led.groupby("b", observed=True):
            if len(g) == 0:
                continue
            print(f"    {str(bk):>14} | {len(g):>5} | {g.pnl.sum()/len(g):>+7.1%} | {g.won.mean():>5.0%}")
        played = led[led.edge >= TOT_THRESH]
        if len(played):
            print(f"    >= threshold: n={len(played)}  ROI={played.pnl.sum()/len(played):+.1%}  win%={played.won.mean():.0%}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--n-sims", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="optional parquet dump of per-game records")
    a = p.parse_args()
    s = pd.Timestamp(a.start).date()
    e = pd.Timestamp(a.end).date()
    df = collect(s, e, a.n_sims, a.seed)
    if a.out:
        df.to_parquet(a.out)
        print(f"[weather_ab] wrote {a.out}")
    _report(df)


if __name__ == "__main__":
    main()
