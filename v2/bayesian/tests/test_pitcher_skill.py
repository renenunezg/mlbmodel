"""Fast unit tests for the pitcher skill model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v2.bayesian import pitcher_skill
from v2.bayesian.tests._synth import synth_pitcher_pa


@pytest.fixture(scope="module")
def synthetic_pa():
    return synth_pitcher_pa(n_sp=20, n_rp=20, pa_per_sp=300, pa_per_rp=80, seed=7)


def test_classify_roles_majority(synthetic_pa):
    pa_df, _ = synthetic_pa
    role = pitcher_skill.classify_roles(pa_df)
    assert set(role.unique()).issubset({"SP", "RP"})
    sp_count = (role == "SP").sum()
    rp_count = (role == "RP").sum()
    assert sp_count > 0 and rp_count > 0


def test_filter_position_player_pitching():
    """A pitcher who also has >=50 batter PAs gets dropped from the pitcher set."""
    rows = []
    # legit pitcher: 50 PAs as pitcher, 0 as batter
    for i in range(50):
        rows.append({"batter": 100 + i, "pitcher": 999, "p_throws": "R",
                     "outcome": "K", "home_team": "LAD", "game_pk": 1,
                     "inning": 1, "inning_topbot": "Top"})
    # contaminator: pitcher_id == batter_id == 200, 60 PAs as batter
    for i in range(60):
        rows.append({"batter": 200, "pitcher": 1234, "p_throws": "R",
                     "outcome": "1B", "home_team": "LAD", "game_pk": 2,
                     "inning": 1, "inning_topbot": "Top"})
    # add 10 PAs of contaminator pitching
    for i in range(10):
        rows.append({"batter": 300 + i, "pitcher": 200, "p_throws": "R",
                     "outcome": "OUT", "home_team": "LAD", "game_pk": 3,
                     "inning": 9, "inning_topbot": "Top"})
    df = pd.DataFrame(rows)
    filtered, dropped = pitcher_skill.filter_position_player_pitching(df)
    assert 200 in dropped
    assert 200 not in filtered["pitcher"].unique()
    assert 999 in filtered["pitcher"].unique()


def test_role_split_recovers_separate_widths(synthetic_pa):
    pa_df, truth = synthetic_pa
    idata, _, _ = pitcher_skill.fit(
        pa_df, draws=300, tune=300, chains=2, target_accept=0.9, random_seed=0
    )
    post = idata.posterior
    sigma_pitcher = post["sigma_pitcher"].mean(("chain", "draw")).values  # (role, K_FREE)
    # SP synthetic sigma was 0.4, RP was 0.2; SP estimate should be higher on average.
    sp_avg = sigma_pitcher[0].mean()
    rp_avg = sigma_pitcher[1].mean()
    assert sp_avg > rp_avg, f"SP sigma not larger than RP: SP={sp_avg:.3f} RP={rp_avg:.3f}"


def test_summarize_returns_required_keys(synthetic_pa):
    pa_df, _ = synthetic_pa
    idata, _, _ = pitcher_skill.fit(
        pa_df, draws=200, tune=200, chains=2, target_accept=0.9, random_seed=0
    )
    out = pitcher_skill.summarize(idata)
    assert {"max_rhat", "min_ess_bulk", "min_ess_tail"}.issubset(out.keys())
    assert np.isfinite(out["max_rhat"])
