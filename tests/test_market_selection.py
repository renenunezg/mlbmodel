import numpy as np

from v2.markets.writer import _best_moneyline, _best_runline, build_game_rows


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


def test_fallback_lineup_suppresses_market_flags(monkeypatch):
    odds = {
        "moneyline": -150,
        "spread": -1.5,
        "spread_odds": 120,
        "total": 8.0,
        "total_over_odds": -110,
        "total_under_odds": -110,
    }
    home_runs = np.concatenate([np.full(900, 6), np.full(100, 2)])
    away_runs = np.concatenate([np.full(900, 3), np.full(100, 7)])
    kwargs = {
        "game_pk": 1,
        "game_date": np.datetime64("2026-06-13"),
        "start_time": None,
        "home_team": "LAD",
        "away_team": "CHW",
        "home_starter": "x",
        "away_starter": "y",
        "home_runs": home_runs,
        "away_runs": away_runs,
        "home_odds": odds,
        "away_odds": {**odds, "moneyline": 130, "spread": 1.5},
        "lineup_source": "lineup_top9+queue_cache",
        "lineups_locked": False,
        "posterior_age_days": 0,
    }
    monkeypatch.setattr("v2.markets.ev.MONEYLINE_ENABLED", True)
    live_home, _ = build_game_rows(**kwargs, lineups_live=True)
    fallback_home, fallback_away = build_game_rows(**kwargs, lineups_live=False)

    assert live_home["ev_flag"] == "LAD"
    for row in (fallback_home, fallback_away):
        assert row["ev_flag"] == "No Play"
        assert row["run_line_ev_flag"] == "No Play"
        assert row["expected_runs"] > 0
