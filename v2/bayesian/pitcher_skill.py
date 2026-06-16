"""Hierarchical D-M pitcher outcome model with role-conditional shrinkage.

Per pitcher Multinomial likelihood over OUTCOMES. Hierarchy widths split by
role (SP vs RP) so starters and relievers shrink toward separate population
spreads. Position-player-pitching contamination is removed by dropping any
pitcher who also appears as a real batter (>=50 PAs in the window).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

from v2.data.pa_dataset import OUTCOMES, load_pa_dataset
from v2.bayesian._common import (
    ActorIndex,
    POSTERIORS_DIR,
    encode_outcomes,
    evaluate_gate,
    league_log_p,
    write_diagnostics,
)

REF_IDX = OUTCOMES.index("OUT")
NON_REF_IDX = [i for i in range(len(OUTCOMES)) if i != REF_IDX]
NON_REF_LABELS = [OUTCOMES[i] for i in NON_REF_IDX]
K_FREE = len(NON_REF_IDX)
ROLES = ("SP", "RP")
POSITION_PLAYER_PA_THRESHOLD = 50


def classify_roles(pa_df: pd.DataFrame) -> pd.Series:
    """Per-pitcher SP/RP from majority of game appearances as inning-1 starter."""
    inning1 = pa_df[(pa_df["inning"] == 1)].copy()
    starter_side = np.where(inning1["inning_topbot"] == "Top", "away", "home")
    inning1 = inning1.assign(side=starter_side)
    starters_per_game = (
        inning1.groupby(["game_pk", "side"])["pitcher"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
        .dropna()
        .astype("int64")
    )
    starts_count = starters_per_game.value_counts()
    games_count = pa_df.groupby("pitcher")["game_pk"].nunique()
    start_share = (starts_count / games_count).fillna(0.0)
    role = np.where(start_share >= 0.5, "SP", "RP")
    return pd.Series(role, index=games_count.index, name="role")


def filter_position_player_pitching(pa_df: pd.DataFrame) -> pd.DataFrame:
    batter_pa = pa_df.groupby("batter").size()
    real_batters = set(batter_pa[batter_pa >= POSITION_PLAYER_PA_THRESHOLD].index)
    pitchers_to_drop = set(pa_df["pitcher"].unique()) & real_batters
    if pitchers_to_drop:
        pa_df = pa_df[~pa_df["pitcher"].isin(pitchers_to_drop)].reset_index(drop=True)
    return pa_df, pitchers_to_drop


def build_model(pa_df: pd.DataFrame, frozen_intercept: np.ndarray | None = None) -> tuple[pm.Model, dict]:
    import pytensor.tensor as pt

    outcome_codes = encode_outcomes(pa_df["outcome"])
    log_p_lg = league_log_p(outcome_codes)
    intercept_prior = log_p_lg[NON_REF_IDX] - log_p_lg[REF_IDX]

    pitcher_idx = ActorIndex.from_series(pa_df["pitcher"])
    p_codes = pitcher_idx.encode(pa_df["pitcher"].to_numpy())

    counts = np.zeros((pitcher_idx.n, len(OUTCOMES)), dtype=np.int64)
    np.add.at(counts, (p_codes, outcome_codes), 1)
    n_per_pitcher = counts.sum(axis=1)

    role_series = classify_roles(pa_df).reindex(pitcher_idx.ids).fillna("RP")
    role_codes = (role_series.to_numpy() == "SP").astype(np.int64)  # 1 = SP, 0 = RP
    role_idx_for_sigma = 1 - role_codes  # 0 = SP slot, 1 = RP slot in ROLES tuple

    coords = {
        "outcome": list(OUTCOMES),
        "outcome_free": NON_REF_LABELS,
        "pitcher": pitcher_idx.ids,
        "role": list(ROLES),
    }

    with pm.Model(coords=coords) as model:
        if frozen_intercept is None:
            intercept = pm.Normal("intercept", mu=intercept_prior, sigma=0.5, dims="outcome_free")
        else:
            # Anchored to the batter fit's baseline so both halves share one
            # league intercept; no free param, nothing to average downstream.
            intercept = np.asarray(frozen_intercept, dtype=float)
        sigma_pitcher = pm.HalfNormal("sigma_pitcher", sigma=0.6, dims=("role", "outcome_free"))
        z_pitcher = pm.Normal("z_pitcher", 0.0, 1.0, dims=("pitcher", "outcome_free"))
        sigma_per_pitcher = sigma_pitcher[role_idx_for_sigma]  # (pitcher, outcome_free)
        beta_pitcher = sigma_per_pitcher * z_pitcher

        logit_free = intercept[None, :] + beta_pitcher

        zeros_ref = pt.zeros((pitcher_idx.n, 1))
        cols = []
        free_iter = iter(range(K_FREE))
        for i in range(len(OUTCOMES)):
            if i == REF_IDX:
                cols.append(zeros_ref)
            else:
                fi = next(free_iter)
                cols.append(logit_free[:, fi : fi + 1])
        logit_full = pt.concatenate(cols, axis=1)

        p = pm.math.softmax(logit_full, axis=1)
        pm.Multinomial(
            "outcome_counts",
            n=n_per_pitcher,
            p=p,
            observed=counts,
            dims=("pitcher", "outcome"),
        )

    n_sp = int((role_series == "SP").sum())
    n_rp = int((role_series == "RP").sum())
    meta = {
        "pitcher_idx": pitcher_idx,
        "log_p_league": log_p_lg.tolist(),
        "intercept_prior": intercept_prior.tolist(),
        "frozen_intercept": (np.asarray(frozen_intercept, dtype=float).tolist()
                             if frozen_intercept is not None else None),
        "n_pa": int(len(pa_df)),
        "n_pitchers": int(pitcher_idx.n),
        "n_sp": n_sp,
        "n_rp": n_rp,
        "min_pa_per_pitcher": int(n_per_pitcher.min()),
        "median_pa_per_pitcher": int(np.median(n_per_pitcher)),
        "max_pa_per_pitcher": int(n_per_pitcher.max()),
    }
    return model, meta


def fit(
    pa_df: pd.DataFrame,
    *,
    frozen_intercept: np.ndarray | None = None,
    draws: int = 1500,
    tune: int = 1500,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 20260504,
) -> tuple[az.InferenceData, dict, float]:
    model, meta = build_model(pa_df, frozen_intercept=frozen_intercept)
    t0 = time.time()
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            nuts_sampler="numpyro",
            nuts_sampler_kwargs={"chain_method": "vectorized"},
            random_seed=random_seed,
            progressbar=True,
        )
    elapsed = time.time() - t0
    return idata, meta, elapsed


def summarize(idata: az.InferenceData) -> dict:
    gate_vars = [v for v in ("intercept", "sigma_pitcher") if v in idata.posterior]
    summary = az.summary(idata, var_names=gate_vars)
    return {
        "max_rhat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "min_ess_tail": float(summary["ess_tail"].min()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2024)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--draws", type=int, default=1500)
    parser.add_argument("--tune", type=int, default=1500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--output-dir", type=Path, default=POSTERIORS_DIR)
    parser.add_argument("--save-trace", action="store_true")
    parser.add_argument("--subsample", type=int, default=None)
    args = parser.parse_args()

    print(f"[pitcher_skill] loading PAs {args.start_year}-{args.end_year}...")
    pa_df = load_pa_dataset(args.start_year, args.end_year)
    pa_df, dropped = filter_position_player_pitching(pa_df)
    print(f"  filtered position-player-pitching: dropped {len(dropped)} pitchers")

    if args.subsample:
        pa_df = pa_df.sample(args.subsample, random_state=args.seed).reset_index(drop=True)
        print(f"  subsampled to {len(pa_df):,} PAs")

    print(f"[pitcher_skill] fitting (chains={args.chains}, draws={args.draws}, tune={args.tune})...")
    idata, meta, elapsed = fit(
        pa_df,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        random_seed=args.seed,
    )
    print(f"  fit complete in {elapsed:.1f}s ({elapsed/60:.2f} min)")

    diag = summarize(idata)
    n_div = int(idata.sample_stats["diverging"].sum().item()) if "diverging" in idata.sample_stats else 0
    print(f"  max_rhat={diag['max_rhat']:.4f}  min_ess_bulk={diag['min_ess_bulk']:.0f}  n_divergent={n_div}")

    gate_passed = evaluate_gate(diag["max_rhat"], diag["min_ess_bulk"])
    print(f"  gate_passed={gate_passed}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "n_pa": meta["n_pa"],
        "n_pitchers": meta["n_pitchers"],
        "n_sp": meta["n_sp"],
        "n_rp": meta["n_rp"],
        "n_dropped_position_player_pitchers": len(dropped),
        "fit_seconds": elapsed,
        "fit_minutes": elapsed / 60,
        "n_divergent": n_div,
        "sampler": {"chains": args.chains, "draws": args.draws, "tune": args.tune},
        **diag,
        "gate_passed": gate_passed,
    }
    write_diagnostics(args.output_dir / "pitcher_skill.json", report)
    print(f"  wrote {args.output_dir / 'pitcher_skill.json'}")

    if args.save_trace:
        trace_path = args.output_dir / "pitcher_skill.nc"
        idata.to_netcdf(trace_path)
        print(f"  wrote {trace_path}")

    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
