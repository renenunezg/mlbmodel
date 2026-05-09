import numpy as np
import pytest

from v2.markets.probs import market_probs, runs_percentiles


def test_all_home_wins():
    h = np.array([5, 6, 7, 8, 9])
    a = np.array([0, 1, 2, 3, 4])
    p = market_probs(h, a, total_line=None, spread_home=None)
    assert p["p_home_win"] == 1.0
    assert p["p_away_win"] == 0.0


def test_symmetric_distribution():
    rng = np.random.default_rng(42)
    n = 50_000
    h = rng.poisson(4.5, size=n)
    a = rng.poisson(4.5, size=n)
    p = market_probs(h, a, total_line=9.5, spread_home=-1.5)
    assert abs(p["p_home_win"] + p["p_away_win"] - 1.0) < 1e-9
    assert abs(p["p_home_win"] - 0.5) < 0.02
    assert abs(p["p_over"] + p["p_under"] - 1.0) < 1e-9


def test_run_line_deterministic_cover():
    h = np.array([5, 5, 5, 5])
    a = np.array([3, 3, 3, 3])
    # home favored at -1.5 → covers when (h-a) > 1.5 → 2 > 1.5 → 1.0
    p = market_probs(h, a, total_line=None, spread_home=-1.5)
    assert p["p_home_cover"] == 1.0
    assert p["p_away_cover"] == 0.0


def test_run_line_underdog_cover():
    h = np.array([2, 2, 2])
    a = np.array([3, 3, 3])
    # away favored, spread_home=+1.5 → home covers when (h-a) > -1.5 → -1 > -1.5 → 1.0
    p = market_probs(h, a, total_line=None, spread_home=1.5)
    assert p["p_home_cover"] == 1.0


def test_total_threshold():
    # totals = 8 in all sims, line = 8.5 → p_under = 1.0
    h = np.array([4, 4, 4])
    a = np.array([4, 4, 4])
    p = market_probs(h, a, total_line=8.5, spread_home=None)
    assert p["p_over"] == 0.0
    assert p["p_under"] == 1.0


def test_integer_total_push():
    # totals = 8 in all sims, line = 8 → push, split 50/50
    h = np.array([4, 4])
    a = np.array([4, 4])
    p = market_probs(h, a, total_line=8.0, spread_home=None)
    assert p["p_over"] == 0.5
    assert p["p_under"] == 0.5


def test_percentiles():
    arr = np.arange(101)  # 0..100 inclusive, n=101
    p10, p50, p90 = runs_percentiles(arr)
    assert p10 == pytest.approx(10.0, abs=0.01)
    assert p50 == pytest.approx(50.0, abs=0.01)
    assert p90 == pytest.approx(90.0, abs=0.01)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        market_probs(np.array([1, 2]), np.array([1]), None, None)


def test_zero_length_raises():
    with pytest.raises(ValueError):
        market_probs(np.array([]), np.array([]), None, None)


def test_no_lines_returns_none():
    h = np.array([5, 6])
    a = np.array([3, 4])
    p = market_probs(h, a, total_line=None, spread_home=None)
    assert p["p_over"] is None
    assert p["p_under"] is None
    assert p["p_home_cover"] is None
