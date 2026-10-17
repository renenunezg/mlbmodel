"""Posterior sigma summaries for the v2 Diagnostics tab.

Writes posterior_sigmas: hierarchical sigma magnitudes from the trained traces,
used by the Variance Decomposition chart. They are posterior-derived because
they describe the model itself, not players. Player-level numbers moved to the
player valuation (v2/players/) and its own tables.

Usage:
    python -m v2.pipeline.write_posterior_summaries [--refit-date YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import logging
from datetime import date

import arviz as az
import numpy as np
import pandas as pd
from sqlalchemy import text

from backend.db import engine
from backend.log import setup_logging
from v2.bayesian._common import POSTERIORS_DIR

log = logging.getLogger(__name__)


def _sigma_summary(idata: az.InferenceData, var: str) -> dict[str, float]:
    arr = idata.posterior[var].values
    flat = arr.mean(axis=tuple(range(2, arr.ndim)))
    return {
        "mean": round(float(flat.mean()), 5),
        "p10": round(float(np.quantile(flat, 0.10)), 5),
        "p90": round(float(np.quantile(flat, 0.90)), 5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit-date", default=str(date.today()))
    args = ap.parse_args()
    refit_date = pd.Timestamp(args.refit_date).date()

    bat = az.from_netcdf(POSTERIORS_DIR / "batter_skill.nc")
    pit = az.from_netcdf(POSTERIORS_DIR / "pitcher_skill.nc")
    park = az.from_netcdf(POSTERIORS_DIR / "park_effects.nc")

    sigma_rows = [
        {"refit_date": refit_date, "sigma_name": "sigma_batter", **_sigma_summary(bat, "sigma_batter")},
        {"refit_date": refit_date, "sigma_name": "sigma_platoon", **_sigma_summary(bat, "sigma_platoon")},
        {"refit_date": refit_date, "sigma_name": "sigma_pitcher", **_sigma_summary(pit, "sigma_pitcher")},
    ]
    park_log_std = park.posterior["park_log"].std(("venue",)).mean(("chain", "draw")).values
    sigma_rows.append({
        "refit_date": refit_date,
        "sigma_name": "sigma_park",
        "mean": round(float(park_log_std), 5),
        "p10": None,
        "p90": None,
    })

    log.info(f"writing {len(sigma_rows)} sigma rows for {refit_date}")
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM posterior_sigmas WHERE refit_date = :d"), {"d": refit_date})
        pd.DataFrame(sigma_rows).to_sql("posterior_sigmas", con=conn, if_exists="append", index=False)
    log.info("done")


if __name__ == "__main__":
    setup_logging()
    main()
