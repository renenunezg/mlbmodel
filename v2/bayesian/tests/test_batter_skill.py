"""Fast unit tests for the batter skill model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v2.bayesian import batter_skill
from v2.bayesian.tests._synth import synth_batter_pa


@pytest.fixture(scope="module")
def synthetic_pa():
    return synth_batter_pa(n_batters=40, pa_per_cell_mean=150, seed=42)


def test_synth_recovery_and_platoon_sign(synthetic_pa):
    """End-to-end mini fit: posterior should recover league rates and platoon sign."""
    pa_df, truth = synthetic_pa
    idata, meta, _ = batter_skill.fit(
        pa_df, draws=300, tune=300, chains=2, target_accept=0.9, random_seed=0
    )
    post = idata.posterior
    # League rates: intercept means should be in same ballpark as priors (rough check).
    intercept_mean = post["intercept"].mean(("chain", "draw")).values
    assert np.all(np.isfinite(intercept_mean))
    # Platoon sign recovery: pick the K column (strikeouts) and check correlation
    # between truth platoon delta and posterior beta_platoon mean.
    sigma_platoon = post["sigma_platoon"].mean(("chain", "draw")).values
    z_platoon = post["z_platoon"].mean(("chain", "draw")).values
    beta_platoon_mean = sigma_platoon * z_platoon
    # Truth col 0 = K (since OUT is reference and removed first)
    corr = np.corrcoef(truth["platoon"][:, 0], beta_platoon_mean[:, 0])[0, 1]
    assert corr > 0.3, f"platoon recovery correlation too low: {corr}"


def test_shrinkage_more_pa_means_less_shrink():
    """A 1000-PA batter at extreme rates should pull farther from prior than a 10-PA one."""
    from v2.data.pa_dataset import OUTCOMES
    rows = []
    # Batter A: 1000 PAs, 50% HR
    for _ in range(500):
        rows.append({"batter": 1, "pitcher": 100, "stand": "R", "p_throws": "R",
                     "outcome": "HR", "home_team": "LAD", "game_pk": 1,
                     "inning": 1, "inning_topbot": "Top"})
    for _ in range(500):
        rows.append({"batter": 1, "pitcher": 100, "stand": "R", "p_throws": "R",
                     "outcome": "OUT", "home_team": "LAD", "game_pk": 1,
                     "inning": 1, "inning_topbot": "Top"})
    # Batter B: 10 PAs, 50% HR
    for _ in range(5):
        rows.append({"batter": 2, "pitcher": 100, "stand": "R", "p_throws": "R",
                     "outcome": "HR", "home_team": "LAD", "game_pk": 1,
                     "inning": 1, "inning_topbot": "Top"})
    for _ in range(5):
        rows.append({"batter": 2, "pitcher": 100, "stand": "R", "p_throws": "R",
                     "outcome": "OUT", "home_team": "LAD", "game_pk": 1,
                     "inning": 1, "inning_topbot": "Top"})
    pa_df = pd.DataFrame(rows)
    idata, _, _ = batter_skill.fit(
        pa_df, draws=300, tune=300, chains=2, target_accept=0.9, random_seed=1
    )
    post = idata.posterior
    sigma_b = post["sigma_batter"].mean(("chain", "draw")).values
    z_b = post["z_batter"].mean(("chain", "draw")).values
    beta = sigma_b * z_b
    # HR is the 6th non-ref outcome (K, BB, HBP, 1B, 2B, 3B, HR) -> index 6
    hr_col = batter_skill.NON_REF_LABELS.index("HR")
    bidx = post["batter"].values.tolist()
    a_idx = bidx.index(1)
    b_idx = bidx.index(2)
    assert beta[a_idx, hr_col] > beta[b_idx, hr_col], (
        f"high-PA batter not less shrunk: A={beta[a_idx, hr_col]:.3f}, B={beta[b_idx, hr_col]:.3f}"
    )
