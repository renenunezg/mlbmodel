import numpy as np
import pytest

from v2.markets.ev import (
    flag_ml,
    flag_runline,
    flag_total_play,
    high_variance_flag,
    kelly_pair,
    kelly_total,
)


def test_flag_ml_threshold(monkeypatch):
    monkeypatch.setattr("v2.markets.ev.MONEYLINE_ENABLED", True)
    assert flag_ml("LAD", 0.55, -110) == "No Play"
    assert flag_ml("LAD", 0.60, -110) == "LAD"


def test_flag_ml_killswitch_off():
    assert flag_ml("LAD", 0.60, -110) == "No Play"


def test_flag_runline_threshold_and_killswitch(monkeypatch):
    assert flag_runline("NYY", 0.60, +120) == "No Play"
    monkeypatch.setattr("v2.markets.ev.RUNLINE_ENABLED", True)
    assert flag_runline("NYY", 0.60, +120) == "NYY"


@pytest.fixture
def totals_on(monkeypatch):
    """Threshold-logic tests must run with the kill-switch off (it's a
    separate concern; the math has to keep working for reactivation)."""
    monkeypatch.setattr("v2.markets.ev.TOTALS_ENABLED", True)


def test_flag_total_play_killswitch_off():
    """Default: totals disabled, always No Play regardless of edge."""
    assert flag_total_play(0.60, 0.40, -110, -110) == "No Play"


def test_flag_total_play_threshold(totals_on):
    assert flag_total_play(0.60, 0.40, -110, -110) == "Over"
    assert flag_total_play(0.40, 0.60, -110, -110) == "Under"
    assert flag_total_play(0.55, 0.45, -110, -110) == "No Play"


def test_flag_total_play_fallback_diff_over(totals_on):
    # No book over/under odds; total_diff drives direction
    assert flag_total_play(float("nan"), float("nan"), float("nan"), float("nan"), total_diff=1.5) == "Over"


def test_kelly_positive_edge():
    full, q = kelly_pair(0.65, -110)
    assert full > 0
    assert abs(q - full * 0.25) < 1e-5
    total_full, total_q = kelly_total("Over", 0.65, 0.35, -110, -110)
    assert total_full > 0
    assert abs(total_q - total_full * 0.25) < 1e-5


def test_high_variance_yes():
    # synthetic high-variance: stdev ~ 5
    rng = np.random.default_rng(0)
    samples = rng.normal(4.5, 5.0, size=1000)
    assert high_variance_flag(samples) == "Yes"
