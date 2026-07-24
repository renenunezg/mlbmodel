import math
import numpy as np
import pandas as pd

from backend.features import (
    LEAGUE_STARTER_SHARE,
    STARTER_SHARE_MIN,
    STARTER_SHARE_MAX,
    compute_starter_inning_share,
    blend_batting_split,
)


def test_starter_share_scalar_fallback_and_clamps():
    assert compute_starter_inning_share(np.nan) == LEAGUE_STARTER_SHARE
    assert compute_starter_inning_share(None) == LEAGUE_STARTER_SHARE
    assert math.isclose(compute_starter_inning_share(6.0), 6.0 / 9.0, rel_tol=1e-9)
    assert compute_starter_inning_share(1.0) == STARTER_SHARE_MIN
    assert compute_starter_inning_share(9.0) == STARTER_SHARE_MAX


def test_starter_share_series_with_mixed_nans():
    s = pd.Series([5.4, np.nan, 1.0, 9.9])
    result = compute_starter_inning_share(s)
    assert math.isclose(result.iloc[0], 5.4 / 9.0, rel_tol=1e-9)
    assert result.iloc[1] == LEAGUE_STARTER_SHARE  # fallback
    assert result.iloc[2] == STARTER_SHARE_MIN      # clamp
    assert result.iloc[3] == STARTER_SHARE_MAX      # clamp


def test_blend_matches_hand_calculation_for_both_hands():
    vs_rhp = blend_batting_split(
        vs_r=0.750, vs_l=0.700,
        opp_handedness="R",
        starter_share=0.55, bullpen_rhp_share=0.7,
    )
    vs_lhp = blend_batting_split(
        vs_r=0.8, vs_l=0.6,
        opp_handedness="L",
        starter_share=0.6, bullpen_rhp_share=0.6,
    )
    assert math.isclose(float(vs_rhp), 0.74325, rel_tol=1e-6)
    assert math.isclose(float(vs_lhp), 0.648, rel_tol=1e-6)


def test_blend_vectorized():
    # Two rows: one vs RHP, one vs LHP - confirms array inputs work.
    result = blend_batting_split(
        vs_r=np.array([0.75, 0.80]),
        vs_l=np.array([0.70, 0.60]),
        opp_handedness=np.array(["R", "L"]),
        starter_share=np.array([0.55, 0.60]),
        bullpen_rhp_share=np.array([0.70, 0.60]),
    )
    assert math.isclose(float(result[0]), 0.74325, rel_tol=1e-6)
    assert math.isclose(float(result[1]), 0.648, rel_tol=1e-6)
