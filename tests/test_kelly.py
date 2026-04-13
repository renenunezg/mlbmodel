"""Tests for Kelly criterion utilities."""
import math
import numpy as np
from backend.kelly import american_to_decimal, kelly_fraction, quarter_kelly, compute_kelly_row


def test_american_to_decimal_positive():
    assert american_to_decimal(100) == 2.0
    assert american_to_decimal(200) == 3.0


def test_american_to_decimal_negative():
    assert abs(american_to_decimal(-150) - 1.6667) < 0.001
    assert american_to_decimal(-100) == 2.0


def test_american_to_decimal_nan():
    assert np.isnan(american_to_decimal(np.nan))


def test_kelly_fraction_even_money():
    # p=0.55, +100 (decimal 2.0): f* = (0.55*2 - 1) / 1 = 0.10
    assert abs(kelly_fraction(0.55, 2.0) - 0.10) < 1e-9


def test_kelly_fraction_negative_edge():
    # p=0.40, decimal 2.0: f* = (0.40*2 - 1) / 1 = -0.20 → clipped to 0
    assert kelly_fraction(0.40, 2.0) == 0.0


def test_kelly_fraction_certain_win():
    # p=1.0, decimal 2.0: f* = (1.0*2 - 1) / 1 = 1.0
    assert kelly_fraction(1.0, 2.0) == 1.0


def test_quarter_kelly_even_money():
    # p=0.55, decimal 2.0: full = 0.10, quarter = 0.025
    assert abs(quarter_kelly(0.55, 2.0) - 0.025) < 1e-6


def test_compute_kelly_row_integration():
    full, qk = compute_kelly_row(0.55, 100)  # +100 American
    assert abs(full - 0.10) < 1e-9
    assert abs(qk - 0.025) < 1e-6


def test_compute_kelly_row_favorite():
    full, qk = compute_kelly_row(0.65, -150)  # -150 American
    # decimal = 1.6667, b = 0.6667
    # f = (0.65 * 1.6667 - 1) / 0.6667 = 0.1250 / 0.6667 ≈ 0.1250
    assert full > 0
    assert qk > 0
    assert abs(qk - full * 0.25) < 1e-6
