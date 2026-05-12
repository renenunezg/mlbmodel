import numpy as np
from backend.simulation import win_prob, compute_game_probs


def test_win_prob_symmetric():
    """Equal xR should give ~50/50 win probability."""
    p = win_prob(4.5, 4.5)
    assert abs(p - 0.5) < 0.01


def test_win_prob_favorite():
    """Higher xR team should have > 50% win probability."""
    p = win_prob(6.0, 3.0)
    assert p > 0.6


def test_win_prob_sum_to_one():
    """p_home_win + p_away_win should sum to ~1."""
    probs = compute_game_probs(5.0, 3.5)
    total = probs["p_home_win"] + probs["p_away_win"]
    assert abs(total - 1.0) < 0.01


def test_cover_less_than_win():
    """P(cover -1.5) should always be less than P(win)."""
    probs = compute_game_probs(5.0, 3.5, spread_home=-1.5)
    assert probs["p_home_cover"] < probs["p_home_win"]


def test_cover_tightens_in_high_scoring():
    """In high-scoring games, cover and win prob should be closer together."""
    low = compute_game_probs(2.5, 2.0, spread_home=-1.5)
    high = compute_game_probs(7.0, 5.5, spread_home=-1.5)
    gap_low = low["p_home_win"] - low["p_home_cover"]
    gap_high = high["p_home_win"] - high["p_home_cover"]
    assert gap_high < gap_low


def test_over_under_sum():
    """p_over + p_under should sum to ~1 (accounting for pushes)."""
    probs = compute_game_probs(4.5, 4.0, total_line=8.5)
    total = probs["p_over"] + probs["p_under"]
    assert abs(total - 1.0) < 0.01


def test_over_prob_increases_with_xr():
    """Higher expected runs should increase over probability."""
    lo = compute_game_probs(3.0, 3.0, total_line=8.5)
    hi = compute_game_probs(5.0, 5.0, total_line=8.5)
    assert hi["p_over"] > lo["p_over"]


def test_run_line_cover_sums():
    """Home cover + away cover should sum to ~1."""
    probs = compute_game_probs(5.0, 3.5, spread_home=-1.5)
    total = probs["p_home_cover"] + probs["p_away_cover"]
    assert abs(total - 1.0) < 0.01


def test_none_when_no_line():
    """Should return None for cover/over/under when no line provided."""
    probs = compute_game_probs(4.5, 4.0)
    assert probs["p_home_cover"] is None
    assert probs["p_over"] is None
