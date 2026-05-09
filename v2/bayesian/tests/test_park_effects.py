"""Fast unit tests for the park-effect residual model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from v2.bayesian import park_effects


def test_statcast_team_mapping_round_trips():
    pf = park_effects.savant_park_factors()
    for stat_team in ["AZ", "CWS", "KC", "SD", "SF", "TB", "WSH"]:
        mapped = park_effects.STATCAST_TO_SAVANT_TEAM[stat_team]
        assert mapped in pf, f"{stat_team} -> {mapped} missing from savant table"


def test_venue_residuals_aggregation():
    pa_df = pd.DataFrame({
        "home_team": ["LAD"] * 10 + ["COL"] * 10,
        "outcome": ["OUT"] * 5 + ["1B"] * 5 + ["HR"] * 5 + ["OUT"] * 5,
    })
    woba_pred = np.full(20, 0.30)
    out = park_effects.venue_residuals(pa_df, woba_pred)
    assert set(out["home_team"]) == {"LAD", "COL"}
    assert (out["n"] == 10).all()


def test_park_model_recovers_synthetic_signal():
    """Inject a known per-park residual mean; posterior mean should track it."""
    rng = np.random.default_rng(0)
    teams = ["COL", "LAD", "NYY", "SDP", "BOS", "MIA"]
    true_log_pf = np.array([0.10, -0.05, 0.0, -0.07, 0.04, -0.06])
    venue_df = pd.DataFrame({
        "home_team": teams,
        "resid_mean": true_log_pf + rng.normal(0, 0.005, len(teams)),
        "resid_var": np.full(len(teams), 0.04),
        "n": np.full(len(teams), 5000),
    })
    idata, meta, _ = park_effects.fit(
        venue_df, draws=400, tune=400, chains=2, target_accept=0.9, random_seed=0
    )
    park_log_mean = idata.posterior["park_log"].mean(("chain", "draw")).values
    # Recovery within 0.05 log-units (prior pulls toward savant defaults somewhat).
    err = np.abs(park_log_mean - true_log_pf)
    assert (err < 0.06).all(), f"park_log recovery err={err}"
