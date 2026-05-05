"""Hierarchical Dirichlet-Multinomial batter outcome model.

Per-batter Multinomial likelihood over the 8 OUTCOMES, with non-centered
additive logit offsets vs OUT (reference). Platoon split is added in a
follow-up commit.
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


def build_model(pa_df: pd.DataFrame) -> tuple[pm.Model, dict]:
    import pytensor.tensor as pt

    outcome_codes = encode_outcomes(pa_df["outcome"])
    log_p_lg = league_log_p(outcome_codes)
    intercept_prior = log_p_lg[NON_REF_IDX] - log_p_lg[REF_IDX]

    batter_idx = ActorIndex.from_series(pa_df["batter"])
    b = batter_idx.encode(pa_df["batter"].to_numpy())

    counts = np.zeros((batter_idx.n, len(OUTCOMES)), dtype=np.int64)
    np.add.at(counts, (b, outcome_codes), 1)
    n_per_batter = counts.sum(axis=1)

    coords = {
        "outcome": list(OUTCOMES),
        "outcome_free": NON_REF_LABELS,
        "batter": batter_idx.ids,
    }

    with pm.Model(coords=coords) as model:
        intercept = pm.Normal("intercept", mu=intercept_prior, sigma=0.5, dims="outcome_free")
        sigma_batter = pm.HalfNormal("sigma_batter", sigma=0.6, dims="outcome_free")
        z_batter = pm.Normal("z_batter", 0.0, 1.0, dims=("batter", "outcome_free"))
        beta_batter = sigma_batter[None, :] * z_batter

        logit_free = intercept[None, :] + beta_batter

        # Insert a zero column at REF_IDX so logits are in canonical OUTCOMES order.
        zeros_ref = pt.zeros((batter_idx.n, 1))
        logit_cols = []
        free_iter = iter(range(K_FREE))
        for i in range(len(OUTCOMES)):
            if i == REF_IDX:
                logit_cols.append(zeros_ref)
            else:
                fi = next(free_iter)
                logit_cols.append(logit_free[:, fi : fi + 1])
        logit_full = pt.concatenate(logit_cols, axis=1)

        p = pm.math.softmax(logit_full, axis=1)
        pm.Multinomial(
            "outcome_counts",
            n=n_per_batter,
            p=p,
            observed=counts,
            dims=("batter", "outcome"),
        )

    meta = {
        "batter_idx": batter_idx,
        "log_p_league": log_p_lg.tolist(),
        "intercept_prior": intercept_prior.tolist(),
        "n_pa": int(len(pa_df)),
        "n_batters": int(batter_idx.n),
        "min_pa_per_batter": int(n_per_batter.min()),
        "median_pa_per_batter": int(np.median(n_per_batter)),
        "max_pa_per_batter": int(n_per_batter.max()),
    }
    return model, meta


def fit(
    pa_df: pd.DataFrame,
    *,
    draws: int = 1000,
    tune: int = 1500,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 20260504,
) -> tuple[az.InferenceData, dict, float]:
    model, meta = build_model(pa_df)
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
    summary = az.summary(idata, var_names=["intercept", "sigma_batter"])
    return {
        "max_rhat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "min_ess_tail": float(summary["ess_tail"].min()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2024)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--output-dir", type=Path, default=POSTERIORS_DIR)
    parser.add_argument("--save-trace", action="store_true")
    parser.add_argument("--subsample", type=int, default=None,
                        help="If set, randomly subsample PAs before fit (debug).")
    args = parser.parse_args()

    print(f"[batter_skill] loading PAs {args.start_year}-{args.end_year}...")
    t_load = time.time()
    pa_df = load_pa_dataset(args.start_year, args.end_year)
    print(f"  loaded {len(pa_df):,} PAs in {time.time()-t_load:.1f}s")

    if args.subsample:
        pa_df = pa_df.sample(args.subsample, random_state=args.seed).reset_index(drop=True)
        print(f"  subsampled to {len(pa_df):,} PAs")

    print(f"[batter_skill] fitting (chains={args.chains}, draws={args.draws}, tune={args.tune})...")
    idata, meta, elapsed = fit(
        pa_df,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        random_seed=args.seed,
    )
    print(f"  fit complete in {elapsed:.1f}s ({elapsed/60:.2f} min)")

    diag = summarize(idata)
    print(f"  max_rhat={diag['max_rhat']:.4f}  min_ess_bulk={diag['min_ess_bulk']:.0f}  min_ess_tail={diag['min_ess_tail']:.0f}")

    n_div = int(idata.sample_stats["diverging"].sum().item()) if "diverging" in idata.sample_stats else 0
    print(f"  n_divergent={n_div}")

    gate_passed = evaluate_gate(diag["max_rhat"], diag["min_ess_bulk"])
    print(f"  gate_passed={gate_passed}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    spike_report = {
        "n_pa": meta["n_pa"],
        "n_batters": meta["n_batters"],
        "fit_seconds": elapsed,
        "fit_minutes": elapsed / 60,
        "n_divergent": n_div,
        "sampler": {"chains": args.chains, "draws": args.draws, "tune": args.tune},
        **diag,
        "gate_passed": gate_passed,
    }
    write_diagnostics(args.output_dir / "spike_batter.json", spike_report)
    print(f"  wrote {args.output_dir / 'spike_batter.json'}")

    if args.save_trace:
        trace_path = args.output_dir / "batter_skill.nc"
        idata.to_netcdf(trace_path)
        print(f"  wrote {trace_path}")

    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
