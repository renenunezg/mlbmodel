"""Sanity check: numpyro fits a 100-batter toy D-M in <60s."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pymc as pm

from v2.bayesian.batter_skill import build_model
from v2.data.pa_dataset import OUTCOMES


def make_synthetic(n_batters: int = 100, pa_per: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    league_p = np.array([0.221, 0.085, 0.012, 0.142, 0.044, 0.005, 0.030, 0.461])
    league_p = league_p / league_p.sum()

    rows = []
    for b in range(n_batters):
        logp = np.log(league_p) + rng.normal(0, 0.4, size=8)
        p = np.exp(logp - logp.max())
        p = p / p.sum()
        draws = rng.choice(8, size=pa_per, p=p)
        for o in draws:
            rows.append({"batter": b, "outcome": OUTCOMES[o]})
    return pd.DataFrame(rows)


def main() -> int:
    pa_df = make_synthetic()
    print(f"[toy_check] {len(pa_df):,} PAs, {pa_df['batter'].nunique()} batters")

    model, _ = build_model(pa_df)
    t0 = time.time()
    with model:
        pm.sample(
            draws=200, tune=300, chains=2,
            target_accept=0.9,
            nuts_sampler="numpyro",
            nuts_sampler_kwargs={"chain_method": "vectorized"},
            random_seed=0,
            progressbar=False,
        )
    elapsed = time.time() - t0
    print(f"[toy_check] fit in {elapsed:.1f}s")
    assert elapsed < 60, f"toy fit too slow: {elapsed:.1f}s"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
