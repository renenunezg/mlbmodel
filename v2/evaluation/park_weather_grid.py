"""Weather-on park-weight grid using a saved weather A/B control.

The reference parquet must come from weather_ab.py with the same window, seed,
and n_sims. This script reuses its park=1.00 weather-on result and simulates
only the requested reduced park weights.

    env/bin/python -m v2.evaluation.park_weather_grid \
        --reference analysis/outputs/weather_ab_actual_prelim.parquet \
        --start 2026-03-26 --end 2026-06-21 --n-sims 300 \
        --park-scales 0.75 0.50 0.25
"""
from __future__ import annotations

import argparse
import dataclasses
import time
from datetime import date

import numpy as np
import pandas as pd

from backend.strategy import EV_THRESHOLDS
from v2.evaluation.weather_ab import (
    BUCKETS,
    _daterange,
    _final_games,
    _get,
    _lineups_from_cache,
    _on_scalars,
    _rmse,
    _sim_total,
    _totals_ledger,
)
from v2.markets.probs import market_probs
from v2.pipeline.score_games import (
    N_DRAWS,
    build_contexts,
    build_inputs,
    load_cache_for_year,
    p_throws_for_pitchers,
    reliever_queue_for_team,
    top9_batters_by_team,
)
from v2.simulator import (
    PosteriorMeans,
    build_queues_from_cache,
    load_advancement_table,
    load_out_subtype_table,
    load_posterior_draws,
)


def _suffix(scale: float) -> str:
    return f"park{int(round(scale * 100)):03d}"


def _scaled_draws(
    draws: list[PosteriorMeans], scale: float
) -> list[PosteriorMeans]:
    return [
        dataclasses.replace(draw, park_log=draw.park_log * scale)
        for draw in draws
    ]


def collect(
    reference: pd.DataFrame,
    start: date,
    end: date,
    n_sims: int,
    seed: int,
    park_scales: list[float],
) -> pd.DataFrame:
    draws = load_posterior_draws(np.random.default_rng(seed), K=N_DRAWS)
    draws_by_scale = {
        scale: _scaled_draws(draws, scale) for scale in park_scales
    }
    adv = load_advancement_table()
    sub = load_out_subtype_table()
    n_per_draw = max(1, n_sims // N_DRAWS)

    finals = _final_games(start, end)
    final_pks = set(finals.game_pk.astype(int))
    cache = load_cache_for_year(start.year)
    fallback_lineups = top9_batters_by_team(cache)
    game_lineups = _lineups_from_cache(cache)
    cache_queues = build_queues_from_cache(start.year)

    all_pitchers = set()
    for queue in cache_queues.values():
        all_pitchers.add(queue.starter)
        all_pitchers.update(queue.relievers)
    throws_lookup = p_throws_for_pitchers(cache, list(all_pitchers))

    records = []
    for run_date in _daterange(start, end):
        contexts = [
            ctx
            for ctx in build_contexts(str(run_date))
            if int(ctx.game_pk) in final_pks
        ]
        if not contexts:
            continue
        teams = {ctx.home_team for ctx in contexts} | {
            ctx.away_team for ctx in contexts
        }
        relievers = {
            team: reliever_queue_for_team(cache, team) for team in teams
        }
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
            wind_signal, temp_c = _on_scalars(ctx)
            inputs = dataclasses.replace(
                inputs, wind_signal=wind_signal, temp_c=temp_c
            )
            line = _get(ctx.home_odds, "total")
            spread = _get(ctx.home_odds, "spread")
            record = {
                "game_pk": int(ctx.game_pk),
                "lineup_source_grid": lineup_source,
                "queue_source_grid": queue_source,
            }
            for scale, scaled in draws_by_scale.items():
                suffix = _suffix(scale)
                home, away = _sim_total(
                    scaled, adv, sub, inputs, n_per_draw, seed
                )
                probs = market_probs(home, away, line, spread)
                record.update({
                    f"our_total_{suffix}": float((home + away).mean()),
                    f"p_over_{suffix}": probs["p_over"],
                    f"home_wp_{suffix}": probs["p_home_win"],
                    f"p_home_cover_{suffix}": probs["p_home_cover"],
                })
            records.append(record)
        print(
            f"[park_weather_grid] {run_date}: {len(contexts)} games "
            f"in {time.time() - t0:.1f}s"
        )

    candidate = pd.DataFrame(records)
    missing = final_pks - set(candidate.game_pk.astype(int))
    if missing:
        print(f"[park_weather_grid] warning: {len(missing)} final games missing")
    return reference.merge(candidate, on="game_pk", how="inner")


def _bucket_report(df: pd.DataFrame, suffix: str) -> None:
    ledger = _totals_ledger(df, suffix)
    ledger = ledger.assign(bucket=pd.cut(ledger.edge, BUCKETS))
    print(f"\n  {_label(suffix)}")
    for bucket, group in ledger.groupby("bucket", observed=True):
        print(
            f"    {str(bucket):>14} | n={len(group):>4} | "
            f"ROI={group.pnl.mean():>+7.1%} | win={group.won.mean():>5.0%}"
        )
    played = ledger[ledger.edge >= EV_THRESHOLDS["totals"]]
    print(
        f"    >= threshold: n={len(played)} "
        f"ROI={played.pnl.mean():+.1%} win={played.won.mean():.0%}"
    )


def _label(suffix: str) -> str:
    if suffix == "on":
        return "weather on, park 1.00"
    return f"weather on, park {int(suffix[-3:]) / 100:.2f}"


def report(df: pd.DataFrame, park_scales: list[float]) -> None:
    suffixes = ["on", *[_suffix(scale) for scale in park_scales]]
    complete = df.dropna(subset=["line", "actual_total"])
    print(
        f"\n==== WEATHER + PARK GRID n={len(df)}, "
        f"{len(complete)} complete with line ===="
    )
    print(
        f"  actual lineups: "
        f"{(df.lineup_source_grid == 'live').mean():.1%}  "
        f"actual queues: {(df.queue_source_grid == 'cache').mean():.1%}"
    )
    print("\n--- point estimate and sides accuracy ---")
    print(
        f"  {'configuration':<28} {'RMSE':>7} {'corr':>7} "
        f"{'ML Brier':>10} {'RL Brier':>10}"
    )
    cover_rows = df[df.spread.notna()].copy()
    cover = (cover_rows.actual_margin > -cover_rows.spread).astype(int)
    for suffix in suffixes:
        total = complete[f"our_total_{suffix}"]
        corr = (total - complete.line).corr(
            complete.actual_total - complete.line
        )
        ml_brier = (
            (df[f"home_wp_{suffix}"] - df.home_win) ** 2
        ).mean()
        rl_brier = (
            (cover_rows[f"p_home_cover_{suffix}"] - cover) ** 2
        ).mean()
        print(
            f"  {_label(suffix):<28} "
            f"{_rmse(total, complete.actual_total):>7.3f} "
            f"{corr:>+7.3f} {ml_brier:>10.4f} {rl_brier:>10.4f}"
        )

    print("\n--- totals edge buckets ---")
    for suffix in suffixes:
        _bucket_report(df, suffix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--n-sims", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--park-scales",
        type=float,
        nargs="+",
        default=[0.75, 0.50, 0.25],
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if any(scale < 0 or scale > 1 for scale in args.park_scales):
        parser.error("park scales must be between 0 and 1")

    reference = pd.read_parquet(args.reference)
    result = collect(
        reference,
        pd.Timestamp(args.start).date(),
        pd.Timestamp(args.end).date(),
        args.n_sims,
        args.seed,
        args.park_scales,
    )
    if args.out:
        result.to_parquet(args.out)
        print(f"[park_weather_grid] wrote {args.out}")
    report(result, args.park_scales)


if __name__ == "__main__":
    main()
