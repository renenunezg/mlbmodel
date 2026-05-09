import numpy as np
import pytest

from v2.markets.ev import (
    flag_ml,
    flag_runline,
    flag_total_play,
    high_variance_flag,
    kelly_pair,
    kelly_total,
    ml_confidence,
    our_odds_from_prob,
    rl_confidence,
)


def test_flag_ml_below_threshold():
    # p=0.55 vs -110 (implied 0.524) → edge 0.026 < 0.045
    assert flag_ml("LAD", 0.55, -110) == "No Play"


def test_flag_ml_above_threshold():
    # p=0.60 vs -110 → edge 0.076 > 0.045
    assert flag_ml("LAD", 0.60, -110) == "LAD"


def test_flag_ml_missing_book():
    assert flag_ml("LAD", 0.60, float("nan")) == "No Play"


def test_flag_runline_above_threshold():
    # p_cover=0.60 vs +120 (implied 0.4545) → edge 0.145 > 0.045
    assert flag_runline("NYY", 0.60, +120) == "NYY"


def test_flag_runline_missing():
    assert flag_runline("NYY", float("nan"), +120) == "No Play"


def test_flag_total_play_over():
    # p_over=0.60 vs -110 over (implied 0.524) → edge 0.076 > 0.065
    assert flag_total_play(0.60, 0.40, -110, -110) == "Over"


def test_flag_total_play_under():
    assert flag_total_play(0.40, 0.60, -110, -110) == "Under"


def test_flag_total_play_no_play():
    # both edges below 0.065 threshold
    assert flag_total_play(0.55, 0.45, -110, -110) == "No Play"


def test_flag_total_play_fallback_diff_over():
    # No book over/under odds; total_diff drives direction
    assert flag_total_play(float("nan"), float("nan"), float("nan"), float("nan"), total_diff=1.5) == "Over"


def test_kelly_pair_zero_edge():
    full, q = kelly_pair(0.50, -110)
    assert full == 0.0
    assert q == 0.0


def test_kelly_pair_positive_edge():
    full, q = kelly_pair(0.65, -110)
    assert full > 0
    assert q == pytest.approx(full * 0.25, abs=1e-5)


def test_kelly_pair_monotonic():
    f1, _ = kelly_pair(0.55, -110)
    f2, _ = kelly_pair(0.65, -110)
    assert f2 > f1


def test_kelly_total_no_play():
    full, q = kelly_total("No Play", 0.55, 0.45, -110, -110)
    assert full == 0.0 and q == 0.0


def test_kelly_total_over():
    full, q = kelly_total("Over", 0.65, 0.35, -110, -110)
    assert full > 0


def test_high_variance_yes():
    # synthetic high-variance: stdev ~ 5
    rng = np.random.default_rng(0)
    samples = rng.normal(4.5, 5.0, size=1000)
    assert high_variance_flag(samples) == "Yes"


def test_high_variance_no():
    rng = np.random.default_rng(0)
    samples = rng.normal(4.5, 2.5, size=1000)
    assert high_variance_flag(samples) == "No"


def test_our_odds_round_trip():
    # p=0.6 → ~-150
    o = our_odds_from_prob(0.6)
    assert -160 < o < -140


def test_ml_confidence():
    # win_prob 0.60 vs -110 (0.524) → 0.076
    c = ml_confidence(0.60, -110)
    assert c == pytest.approx(0.0762, abs=1e-3)


def test_rl_confidence_nan_propagates():
    assert np.isnan(rl_confidence(float("nan"), -110))
    assert np.isnan(rl_confidence(0.55, float("nan")))
