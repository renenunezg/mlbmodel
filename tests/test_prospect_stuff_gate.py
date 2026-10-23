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

    base = pd.DataFrame({"p_mlb": [0.6, 0.2, 0.3], "p_regular": [0.3, 0.05, 0.1], "peak_war": [2.0, 0.1, 1.0],
                         "control_war": [6.0, 0.2, 3.0], "control_q25": [1.0, 0.0, 0.5], "control_q75": [9.0, 1.0, 5.0]}, index=[10, 11, 12])
    feats = pd.DataFrame([[40, 40, 40, 40, 90.0, 2, 1.0, 0.0, 2, 90, 1], [75, 75, 75, 60, 99.0, 5, 0.0, 0.0, 0, 0, 0],
                          [np.nan] * len(adjust.STUFF_FEATURES) + [0, 0, 0]],
                         columns=adjust.FEATURES, index=[10, 11, 12])
    out = adjust.apply(base, feats, adjustment)
    pd.testing.assert_series_equal(out.loc[12], base.loc[12].astype("float64"))
    # Tracked rows are re-ordered, never re-leveled: the base model's values among them are kept.
    for col in base:
        assert sorted(out.loc[[10, 11], col]) == sorted(base.loc[[10, 11], col])
