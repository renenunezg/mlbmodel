"""Tests for dynamic starter/bullpen inning-share blend."""
import math
import numpy as np
import pandas as pd

from backend.features import (
    LEAGUE_STARTER_SHARE,
    LEAGUE_BULLPEN_RHP_SHARE,
    STARTER_SHARE_MIN,
    STARTER_SHARE_MAX,
    compute_starter_inning_share,
    blend_batting_split,
)


def test_starter_share_fallback_when_missing():
    assert compute_starter_inning_share(np.nan) == LEAGUE_STARTER_SHARE
    assert compute_starter_inning_share(None) == LEAGUE_STARTER_SHARE


def test_starter_share_scalar_computation():
    # 6.0 IP/start -> 6/9 = 0.667, within clamp range
    assert math.isclose(compute_starter_inning_share(6.0), 6.0 / 9.0, rel_tol=1e-9)
    # 4.5 IP/start -> 0.5
    assert math.isclose(compute_starter_inning_share(4.5), 0.5, rel_tol=1e-9)


def test_starter_share_clamps():
    # Opener (1 IP) should clamp to min
    assert compute_starter_inning_share(1.0) == STARTER_SHARE_MIN
    # Complete game (9 IP) should clamp to max
    assert compute_starter_inning_share(9.0) == STARTER_SHARE_MAX


def test_starter_share_series_with_mixed_nans():
    s = pd.Series([5.4, np.nan, 1.0, 9.9])
    result = compute_starter_inning_share(s)
    assert math.isclose(result.iloc[0], 5.4 / 9.0, rel_tol=1e-9)
    assert result.iloc[1] == LEAGUE_STARTER_SHARE  # fallback
    assert result.iloc[2] == STARTER_SHARE_MIN      # clamp
    assert result.iloc[3] == STARTER_SHARE_MAX      # clamp


def test_blend_matches_hand_calculation():
    # starter_share=0.55, bullpen_rhp_share=0.7, opp_handedness=R, vs_r=0.750, vs_l=0.700
    # starter portion (vs RHP): 0.750
    # bullpen portion: 0.7 * 0.750 + 0.3 * 0.700 = 0.525 + 0.21 = 0.735
    # blended: 0.55 * 0.750 + 0.45 * 0.735 = 0.4125 + 0.33075 = 0.74325
    blended = blend_batting_split(
        vs_r=0.750, vs_l=0.700,
        opp_handedness="R",
        starter_share=0.55, bullpen_rhp_share=0.7,
    )
    assert math.isclose(float(blended), 0.74325, rel_tol=1e-6)


def test_blend_uses_opp_handedness_for_starter_portion():
    # When facing LHP, starter portion should use vs_l
    # starter_share=0.6, bullpen_rhp_share=0.6, vs_r=0.8, vs_l=0.6
    # starter portion (vs LHP): 0.6
    # bullpen portion: 0.6 * 0.8 + 0.4 * 0.6 = 0.48 + 0.24 = 0.72
    # blended: 0.6 * 0.6 + 0.4 * 0.72 = 0.36 + 0.288 = 0.648
    blended = blend_batting_split(
        vs_r=0.8, vs_l=0.6,
        opp_handedness="L",
        starter_share=0.6, bullpen_rhp_share=0.6,
    )
    assert math.isclose(float(blended), 0.648, rel_tol=1e-6)


def test_blend_vectorized():
    # Two rows: one vs RHP, one vs LHP — confirms array inputs work.
    result = blend_batting_split(
        vs_r=np.array([0.75, 0.80]),
        vs_l=np.array([0.70, 0.60]),
        opp_handedness=np.array(["R", "L"]),
        starter_share=np.array([0.55, 0.60]),
        bullpen_rhp_share=np.array([0.70, 0.60]),
    )
    assert math.isclose(float(result[0]), 0.74325, rel_tol=1e-6)
    assert math.isclose(float(result[1]), 0.648, rel_tol=1e-6)


def test_blend_is_identity_when_splits_equal():
    # If vs_r == vs_l, blend should equal that value regardless of shares.
    blended = blend_batting_split(
        vs_r=0.725, vs_l=0.725,
        opp_handedness="R",
        starter_share=0.42, bullpen_rhp_share=0.33,
    )
    assert math.isclose(float(blended), 0.725, rel_tol=1e-9)


def test_blend_handles_nan_inputs_without_crashing():
    # NaN in splits propagates (expected), no crash.
    result = blend_batting_split(
        vs_r=np.nan, vs_l=0.700,
        opp_handedness="R",
        starter_share=0.55, bullpen_rhp_share=LEAGUE_BULLPEN_RHP_SHARE,
    )
    assert np.isnan(float(result))
