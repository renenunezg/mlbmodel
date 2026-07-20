import numpy as np

from v2.markets.writer import _best_moneyline, _best_runline, _best_total


def _package(*offers):
    return {**offers[0], "offers": list(offers)}


def test_moneyline_uses_best_available_price():
    odds = _package(
        {"book": "draftkings", "moneyline": -125},
        {"book": "fanduel", "moneyline": -115},
        {"book": "betmgm", "moneyline": -120},
    )

    selected = _best_moneyline(odds, win_prob=0.6)

    assert selected["book"] == "fanduel"
    assert selected["moneyline"] == -115


def test_runline_uses_best_price_at_one_and_a_half():
    odds = _package(
        {"book": "draftkings", "spread": -1.5, "spread_odds": -110},
        {"book": "fanduel", "spread": -1.5, "spread_odds": 100},
        {"book": "betmgm", "spread": -2.5, "spread_odds": 180},
    )
    team_runs = np.array([2, 3, 4, 5])
    opponent_runs = np.array([1, 4, 3, 2])

    selected, p_cover = _best_runline(odds, team_runs, opponent_runs)

    assert selected["book"] == "fanduel"
    assert selected["spread"] == -1.5
    assert selected["spread_odds"] == 100
    assert p_cover == 0.25


def test_total_compares_each_books_line_and_price():
    odds = _package(
        {
            "book": "draftkings",
            "total": 8.5,
            "total_over_odds": -200,
            "total_under_odds": 160,
        },
        {
            "book": "fanduel",
            "total": 9.5,
            "total_over_odds": 150,
            "total_under_odds": -180,
        },
        {
            "book": "betmgm",
            "total": 8.5,
            "total_over_odds": -150,
            "total_under_odds": 130,
        },
    )
    home_runs = np.array([4, 4, 5, 6])
    away_runs = np.array([4, 5, 5, 5])

    selected, p_over, p_under = _best_total(
        odds,
        None,
        home_runs,
        away_runs,
    )

    assert selected["book"] == "betmgm"
    assert selected["total"] == 8.5
    assert selected["total_over_odds"] == -150
    assert p_over == 0.75
    assert p_under == 0.25
