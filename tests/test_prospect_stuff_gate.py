"""Pitcher grading gate: only tracked pitchers are graded, and untracked pitchers keep the base prediction."""
import numpy as np
import pandas as pd
import xgboost as xgb

from v2.prospects import adjust
from v2.prospects.score import pitcher_gate


def _adjustment(passed: bool) -> dict:
    rng = np.random.default_rng(0)
    feats = pd.DataFrame(rng.normal(50, 10, (200, len(adjust.FEATURES))), columns=adjust.FEATURES)
    base = pd.DataFrame({"p_mlb": rng.uniform(0.05, 0.6, 200), "peak_war": rng.uniform(0.0, 2.0, 200)})
    reach, war = adjust._train(feats, base, pd.Series(rng.integers(0, 2, 200)), pd.Series(rng.uniform(0, 3, 200)))
    return {"reach": adjust._encode(reach), "war": adjust._encode(war), "adoption_passed": passed, "elite_passed": passed}


def test_untracked_pitchers_stay_ungraded_and_unadjusted():
    rows = pd.DataFrame({"gradeable": [True, True, False]}, index=[10, 11, 12])
    tracked = pd.Series([True, False, True], index=rows.index)
    adjustment = _adjustment(True)
    assert pitcher_gate(rows, tracked, adjustment).tolist() == [None, "not_tracked", "insufficient_sample"]
    assert pitcher_gate(rows, tracked, {**adjustment, "elite_passed": False}).tolist() == ["model_not_validated"] * 3

    base = pd.DataFrame({"p_mlb": [0.3, 0.3], "p_regular": [0.1, 0.1], "peak_war": [1.0, 1.0], "control_war": [3.0, 3.0],
                         "control_q25": [0.5, 0.5], "control_q75": [5.0, 5.0]}, index=[10, 11])
    feats = pd.DataFrame([[70, 65, 66, 55, 97.0, 4, 0.0]] + [[np.nan] * len(adjust.FEATURES)], columns=adjust.FEATURES, index=[10, 11])
    out = adjust.apply(base, feats, adjustment)
    pd.testing.assert_series_equal(out.loc[11], base.loc[11])
    assert out.loc[10, "p_mlb"] != base.loc[10, "p_mlb"]
