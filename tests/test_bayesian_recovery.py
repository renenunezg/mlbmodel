"""Synthetic recovery gates for the three Bayesian skill models."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tests._synthetic_pa import synth_batter_pa, synth_pitcher_pa
from v2.bayesian import batter_skill, park_effects, pitcher_skill


def test_batter_model_recovers_platoon_direction():
    pa, truth = synth_batter_pa(n_batters=40, pa_per_cell_mean=150, seed=42)
    idata, _, _ = batter_skill.fit(
        pa, draws=300, tune=300, chains=2, target_accept=0.9, random_seed=0
    )
    posterior = idata.posterior
    beta_platoon = (
        posterior["sigma_platoon"].mean(("chain", "draw")).values
        * posterior["z_platoon"].mean(("chain", "draw")).values
    )

    assert np.corrcoef(truth["platoon"][:, 0], beta_platoon[:, 0])[0, 1] > 0.3


def test_pitcher_model_recovers_role_widths():
    pa, _ = synth_pitcher_pa(
        n_sp=20,
        n_rp=20,
        pa_per_sp=300,
        pa_per_rp=80,
        seed=7,
    )
    idata, _, _ = pitcher_skill.fit(
        pa, draws=300, tune=300, chains=2, target_accept=0.9, random_seed=0
    )
    sigma_pitcher = idata.posterior["sigma_pitcher"].mean(("chain", "draw")).values

    assert sigma_pitcher[0].mean() > sigma_pitcher[1].mean()


def test_park_model_recovers_synthetic_signal():
    rng = np.random.default_rng(0)
    true_log_pf = np.array([0.10, -0.05, 0.0, -0.07, 0.04, -0.06])
    venue_df = pd.DataFrame({
        "home_team": ["COL", "LAD", "NYY", "SDP", "BOS", "MIA"],
        "resid_mean": true_log_pf + rng.normal(0, 0.005, len(true_log_pf)),
        "resid_var": np.full(len(true_log_pf), 0.04),
        "n": np.full(len(true_log_pf), 5000),
    })
    idata, _, _ = park_effects.fit(
        venue_df, draws=400, tune=400, chains=2, target_accept=0.9, random_seed=0
    )
    estimated = idata.posterior["park_log"].mean(("chain", "draw")).values

    assert (np.abs(estimated - true_log_pf) < 0.06).all()
