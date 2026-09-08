"""Fit a park logit coefficient through the simulator's expected-wOBA response.

The observation likelihood remains in wOBA units. The coefficient is in logit
units; these are linked by the actual softmax response, never equated.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from backend.log import setup_logging
from v2.bayesian._common import (
    POSTERIORS_DIR,
    WOBA_WEIGHTS,
    encode_outcomes,
    evaluate_gate,
    write_diagnostics,
)
from v2.data.pa_dataset import OUTCOMES, load_pa_dataset

log = logging.getLogger(__name__)

REF_IDX = OUTCOMES.index("OUT")
NON_REF_IDX = [i for i in range(len(OUTCOMES)) if i != REF_IDX]
WOBA_VEC = np.array([WOBA_WEIGHTS[o] for o in OUTCOMES], dtype=np.float64)

def predict_probs_per_pa(
    pa_df: pd.DataFrame,
    batter_idata: az.InferenceData,
    pitcher_idata: az.InferenceData,
) -> np.ndarray:
    """Predicted wOBA per PA combining batter (with platoon) and pitcher logits."""
    import scipy.special as sp

    bat_post = batter_idata.posterior
    pit_post = pitcher_idata.posterior

    intercept = pit_post["intercept"].mean(("chain", "draw")).values  # (K_FREE,)
    sigma_b = bat_post["sigma_batter"].mean(("chain", "draw")).values
    z_b = bat_post["z_batter"].mean(("chain", "draw")).values  # (n_batter, K_FREE)
    sigma_pl = bat_post["sigma_platoon"].mean(("chain", "draw")).values
    z_pl = bat_post["z_platoon"].mean(("chain", "draw")).values
    beta_main = sigma_b * z_b
    beta_platoon = sigma_pl * z_pl

    sigma_p = pit_post["sigma_pitcher"].mean(("chain", "draw")).values  # (role, K_FREE)
    z_p = pit_post["z_pitcher"].mean(("chain", "draw")).values  # (n_pitcher, K_FREE)
    from v2.bayesian.pitcher_skill import classify_roles
    roles = classify_roles(pa_df).reindex(pit_post["pitcher"].values).fillna("RP")
    role_codes = (roles.to_numpy() == "RP").astype(int)
    beta_pitcher = sigma_p[role_codes] * z_p

    batter_ids = bat_post["batter"].values
    pitcher_ids = pit_post["pitcher"].values
    bmap = {int(b): i for i, b in enumerate(batter_ids)}
    pmap = {int(p): i for i, p in enumerate(pitcher_ids)}

    bs = pa_df["batter"].astype("int64").to_numpy()
    ps = pa_df["pitcher"].astype("int64").to_numpy()
    keep = np.array([(b in bmap) and (p in pmap) for b, p in zip(bs, ps)])
    if not keep.all():
        pa_df = pa_df.loc[keep].reset_index(drop=True)
        bs = pa_df["batter"].astype("int64").to_numpy()
        ps = pa_df["pitcher"].astype("int64").to_numpy()

    b_codes = np.array([bmap[int(b)] for b in bs])
    p_codes = np.array([pmap[int(p)] for p in ps])
    vs_lhp = (pa_df["p_throws"].to_numpy() == "L").astype(np.float64)[:, None]

    # Single shared baseline (batter-anchored); pitcher offsets are deviations
    # from it, so nothing to double-count or average.
    logit_free = (
        intercept[None, :]
        + beta_main[b_codes]
        + vs_lhp * beta_platoon[b_codes]
        + beta_pitcher[p_codes]
    )

    n_pa = logit_free.shape[0]
    logit_full = np.zeros((n_pa, len(OUTCOMES)), dtype=np.float64)
    free_iter = iter(range(len(NON_REF_IDX)))
    for i in range(len(OUTCOMES)):
        if i == REF_IDX:
            continue
        fi = next(free_iter)
        logit_full[:, i] = logit_free[:, fi]
    probs = sp.softmax(logit_full, axis=1)
    return probs, pa_df


PARK_GRID = np.linspace(-2.0, 2.0, 401)
PARK_MODEL_VERSION = "woba-logit-response-v2"


def venue_residuals(pa_df: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    """Expected wOBA response under the exact softmax transformation used live.

    Tabulate the per-PA response before averaging, avoiding a ratio-of-means
    approximation. Linear interpolation in the fit uses a 0.01-logit grid.
    """
    if probabilities.shape != (len(pa_df), len(OUTCOMES)):
        raise ValueError("Park fitting requires per-PA outcome probabilities")
    observed = WOBA_VEC[encode_outcomes(pa_df["outcome"])]
    tilt = np.exp(WOBA_VEC[:, None] * PARK_GRID[None, :])
    rows = []
    for venue in sorted(pa_df["home_team"].unique()):
        mask = pa_df["home_team"].to_numpy() == venue
        probs = probabilities[mask]
        curve = np.zeros(len(PARK_GRID))
        for start in range(0, len(probs), 2000):
            block = probs[start:start + 2000]
            curve += ((block * WOBA_VEC) @ tilt / (block @ tilt)).sum(axis=0)
        curve /= len(probs)
        residual = observed[mask] - probs @ WOBA_VEC
        rows.append({"home_team": venue, "n": len(probs),
                     "observed_woba": float(observed[mask].mean()),
                     "resid_var": float(np.var(residual, ddof=1)),
                     "response_curve": curve})
    return pd.DataFrame(rows)


def build_model(venue_df: pd.DataFrame) -> tuple[pm.Model, dict]:
    import pytensor.tensor as pt

    required = {"response_curve", "observed_woba", "resid_var", "n", "home_team"}
    if not required.issubset(venue_df.columns):
        raise ValueError("Park fit requires the nonlinear wOBA response, not a wOBA residual as a logit")
    n = venue_df["n"].to_numpy()
    curves = np.stack(venue_df["response_curve"])
    coords = {"venue": venue_df["home_team"].tolist()}
    with pm.Model(coords=coords) as model:
        # Neutral prior in logit units. A published run park factor has different
        # units and cannot be substituted for this coefficient.
        park_log = pm.TruncatedNormal("park_log", mu=0.0, sigma=0.25,
                                      lower=PARK_GRID[0], upper=PARK_GRID[-1], dims="venue")
        response = pt.stack([pt.interp(park_log[i], PARK_GRID, curve) for i, curve in enumerate(curves)])
        pm.Normal("obs_mean", mu=response,
                  sigma=np.sqrt(np.maximum(venue_df["resid_var"].to_numpy(), 1e-6) / n),
                  observed=venue_df["observed_woba"].to_numpy(), dims="venue")
    return model, {"venues": coords["venue"], "n_per_venue": n.tolist(),
                   "park_model_version": PARK_MODEL_VERSION}


def fit(
    venue_df: pd.DataFrame,
    *,
    draws: int = 2000,
    tune: int = 1500,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 20260504,
) -> tuple[az.InferenceData, dict, float]:
    model, meta = build_model(venue_df)
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
    idata.posterior.attrs["park_model_version"] = PARK_MODEL_VERSION
    return idata, meta, elapsed


GATE_VARS = ["park_log"]  # sigma_resid is a nuisance variance param the simulator never reads


def summarize(idata: az.InferenceData) -> dict:
    summary = az.summary(idata, var_names=GATE_VARS)
    return {
        "max_rhat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "min_ess_tail": float(summary["ess_tail"].min()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2024)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--batter-trace", type=Path, required=True)
    parser.add_argument("--pitcher-trace", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=1500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--output-dir", type=Path, default=POSTERIORS_DIR)
    parser.add_argument("--save-trace", action="store_true")
    args = parser.parse_args()

    log.info(f"loading PAs {args.start_year}-{args.end_year}...")
    pa_df = load_pa_dataset(args.start_year, args.end_year)

    log.info("loading batter/pitcher traces...")
    bat_idata = az.from_netcdf(args.batter_trace)
    pit_idata = az.from_netcdf(args.pitcher_trace)

    log.info("computing per-PA wOBA predictions...")
    woba_pred, pa_df = predict_probs_per_pa(pa_df, bat_idata, pit_idata)
    log.info(f"predicted on {len(pa_df):,} PAs")

    venue_df = venue_residuals(pa_df, woba_pred)
    log.info(f"{len(venue_df)} venues, total PAs {int(venue_df['n'].sum()):,}")

    idata, meta, elapsed = fit(
        venue_df,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        random_seed=args.seed,
    )
    log.info(f"fit complete in {elapsed:.1f}s")

    diag = summarize(idata)
    n_div = int(idata.sample_stats["diverging"].sum().item()) if "diverging" in idata.sample_stats else 0
    log.info(f"max_rhat={diag['max_rhat']:.4f}  min_ess_bulk={diag['min_ess_bulk']:.0f}  n_divergent={n_div}")

    gate_passed = evaluate_gate(diag["max_rhat"], diag["min_ess_bulk"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "venues": meta["venues"],
        "n_per_venue": meta["n_per_venue"],
        "park_model_version": PARK_MODEL_VERSION,
        "fit_seconds": elapsed,
        "fit_minutes": elapsed / 60,
        "n_divergent": n_div,
        "sampler": {"chains": args.chains, "draws": args.draws, "tune": args.tune},
        **diag,
        "gate_passed": gate_passed,
    }
    write_diagnostics(args.output_dir / "park_effects.json", report)
    log.info(f"wrote {args.output_dir / 'park_effects.json'}")

    if args.save_trace:
        trace_path = args.output_dir / "park_effects.nc"
        idata.to_netcdf(trace_path)
        log.info(f"wrote {trace_path}")

    return 0 if gate_passed else 1


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
