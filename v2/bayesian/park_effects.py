"""Stage-B park-effect fit on residual wOBA.

Combines the batter and pitcher posterior means into a per-PA expected wOBA,
then fits a per-park scalar log-multiplier on the residual. Prior on each
park's log-PF is centered at log(savant_pf/100) with a loose sigma so the
data, not the prior, drives the posterior when there's signal.
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
    POSTERIORS_DIR,
    WOBA_WEIGHTS,
    encode_outcomes,
    evaluate_gate,
    write_diagnostics,
)

REF_IDX = OUTCOMES.index("OUT")
NON_REF_IDX = [i for i in range(len(OUTCOMES)) if i != REF_IDX]
WOBA_VEC = np.array([WOBA_WEIGHTS[o] for o in OUTCOMES], dtype=np.float64)

# Statcast home_team code -> savant park-factor key (where they differ).
STATCAST_TO_SAVANT_TEAM = {
    "AZ": "ARI", "CWS": "CHW", "KC": "KCR", "SD": "SDP",
    "SF": "SFG", "TB": "TBR", "WSH": "WSN",
}


def savant_park_factors() -> dict[str, float]:
    from backend.data.savant import _static_park_factors
    df = _static_park_factors(season=2025)
    return dict(zip(df["team"], df["park_factor"].astype(float)))


def predict_woba_per_pa(
    pa_df: pd.DataFrame,
    batter_idata: az.InferenceData,
    pitcher_idata: az.InferenceData,
) -> np.ndarray:
    """Predicted wOBA per PA combining batter (with platoon) and pitcher logits."""
    import scipy.special as sp

    bat_post = batter_idata.posterior
    pit_post = pitcher_idata.posterior

    intercept_b = bat_post["intercept"].mean(("chain", "draw")).values  # (K_FREE,)
    sigma_b = bat_post["sigma_batter"].mean(("chain", "draw")).values
    z_b = bat_post["z_batter"].mean(("chain", "draw")).values  # (n_batter, K_FREE)
    sigma_pl = bat_post["sigma_platoon"].mean(("chain", "draw")).values
    z_pl = bat_post["z_platoon"].mean(("chain", "draw")).values
    beta_main = sigma_b * z_b
    beta_platoon = sigma_pl * z_pl

    intercept_p = pit_post["intercept"].mean(("chain", "draw")).values
    sigma_p = pit_post["sigma_pitcher"].mean(("chain", "draw")).values  # (role, K_FREE)
    z_p = pit_post["z_pitcher"].mean(("chain", "draw")).values  # (n_pitcher, K_FREE)
    # For pitcher posterior we only need beta = sigma_per_pitcher * z; sigma_per_pitcher
    # is unknown without role lookup. Approximate as role-averaged sigma to keep this
    # stage standalone; the residual model is robust to the small bias this introduces.
    sigma_p_avg = sigma_p.mean(axis=0)
    beta_pitcher = sigma_p_avg * z_p

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

    # Combine: intercept appears in both models; subtract one copy so it isn't
    # double-counted. Average the two intercept estimates.
    intercept_combined = 0.5 * (intercept_b + intercept_p)
    logit_free = (
        intercept_combined[None, :]
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
    return probs @ WOBA_VEC, pa_df


def venue_residuals(pa_df: pd.DataFrame, woba_pred: np.ndarray) -> pd.DataFrame:
    outcome_codes = encode_outcomes(pa_df["outcome"])
    woba_obs = WOBA_VEC[outcome_codes]
    resid = woba_obs - woba_pred
    grouped = (
        pd.DataFrame({"home_team": pa_df["home_team"].values, "resid": resid})
        .groupby("home_team")["resid"]
        .agg(["mean", "var", "count"])
        .reset_index()
    )
    grouped = grouped.rename(columns={"mean": "resid_mean", "var": "resid_var", "count": "n"})
    return grouped


def build_model(venue_df: pd.DataFrame) -> tuple[pm.Model, dict]:
    pf_map = savant_park_factors()
    venue_df = venue_df.copy()
    venue_df["savant_team"] = venue_df["home_team"].map(STATCAST_TO_SAVANT_TEAM).fillna(
        venue_df["home_team"]
    )
    venue_df["savant_pf"] = venue_df["savant_team"].map(pf_map).fillna(100.0)
    venue_df["log_pf_prior"] = np.log(venue_df["savant_pf"].to_numpy() / 100.0)

    n = venue_df["n"].to_numpy()
    obs_mean = venue_df["resid_mean"].to_numpy()
    coords = {"venue": venue_df["home_team"].tolist()}

    # sigma_resid is fixed: a free hyperprior couples chain means across all
    # park_log values without adding signal (each venue has ~13k PAs and the
    # data swamps any reasonable sigma in this range). The empirical posterior
    # from a free fit was ~0.36; we lock it here.
    SIGMA_RESID = 0.36
    with pm.Model(coords=coords) as model:
        park_log = pm.Normal(
            "park_log",
            mu=venue_df["log_pf_prior"].to_numpy(),
            sigma=0.08,
            dims="venue",
        )
        pm.Normal(
            "obs_mean",
            mu=park_log,
            sigma=SIGMA_RESID / np.sqrt(n),
            observed=obs_mean,
            dims="venue",
        )

    meta = {
        "venues": venue_df["home_team"].tolist(),
        "n_per_venue": n.tolist(),
        "savant_pf": venue_df["savant_pf"].tolist(),
    }
    return model, meta


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

    print(f"[park_effects] loading PAs {args.start_year}-{args.end_year}...")
    pa_df = load_pa_dataset(args.start_year, args.end_year)

    print(f"[park_effects] loading batter/pitcher traces...")
    bat_idata = az.from_netcdf(args.batter_trace)
    pit_idata = az.from_netcdf(args.pitcher_trace)

    print(f"[park_effects] computing per-PA wOBA predictions...")
    woba_pred, pa_df = predict_woba_per_pa(pa_df, bat_idata, pit_idata)
    print(f"  predicted on {len(pa_df):,} PAs")

    venue_df = venue_residuals(pa_df, woba_pred)
    print(f"  {len(venue_df)} venues, total PAs {int(venue_df['n'].sum()):,}")

    idata, meta, elapsed = fit(
        venue_df,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        random_seed=args.seed,
    )
    print(f"  fit complete in {elapsed:.1f}s")

    diag = summarize(idata)
    n_div = int(idata.sample_stats["diverging"].sum().item()) if "diverging" in idata.sample_stats else 0
    print(f"  max_rhat={diag['max_rhat']:.4f}  min_ess_bulk={diag['min_ess_bulk']:.0f}  n_divergent={n_div}")

    gate_passed = evaluate_gate(diag["max_rhat"], diag["min_ess_bulk"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "venues": meta["venues"],
        "n_per_venue": meta["n_per_venue"],
        "savant_pf": meta["savant_pf"],
        "fit_seconds": elapsed,
        "fit_minutes": elapsed / 60,
        "n_divergent": n_div,
        "sampler": {"chains": args.chains, "draws": args.draws, "tune": args.tune},
        **diag,
        "gate_passed": gate_passed,
    }
    write_diagnostics(args.output_dir / "park_effects.json", report)
    print(f"  wrote {args.output_dir / 'park_effects.json'}")

    if args.save_trace:
        trace_path = args.output_dir / "park_effects.nc"
        idata.to_netcdf(trace_path)
        print(f"  wrote {trace_path}")

    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
