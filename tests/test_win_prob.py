import numpy as np
from backend.simulation import win_prob, compute_game_probs


def test_moneyline_probability_contract():
    p = win_prob(4.5, 4.5)
    assert abs(p - 0.5) < 0.01
    assert win_prob(6.0, 3.0) > 0.6
    probs = compute_game_probs(5.0, 3.5)
    assert abs(probs["p_home_win"] + probs["p_away_win"] - 1.0) < 0.01


def test_run_line_probability_contract():
    probs = compute_game_probs(5.0, 3.5, spread_home=-1.5)
    assert probs["p_home_cover"] < probs["p_home_win"]
    assert abs(probs["p_home_cover"] + probs["p_away_cover"] - 1.0) < 0.01


def test_total_probability_contract():
    probs = compute_game_probs(4.5, 4.0, total_line=8.5)
    assert abs(probs["p_over"] + probs["p_under"] - 1.0) < 0.01
    lo = compute_game_probs(3.0, 3.0, total_line=8.5)
    hi = compute_game_probs(5.0, 5.0, total_line=8.5)
    assert hi["p_over"] > lo["p_over"]


def test_none_when_no_line():
    """Should return None for cover/over/under when no line provided."""
    probs = compute_game_probs(4.5, 4.0)
    assert probs["p_home_cover"] is None
    assert probs["p_over"] is None
