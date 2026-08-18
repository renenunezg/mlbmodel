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
    monkeypatch.setattr("v2.markets.ev.RUNLINE_ENABLED", True)
    live_home, _ = build_game_rows(**kwargs, lineups_live=True)
    fallback_home, fallback_away = build_game_rows(**kwargs, lineups_live=False)

    assert live_home["ev_flag"] == "LAD"
    assert live_home["run_line_ev_flag"] == "LAD"
    for row in (fallback_home, fallback_away):
        assert row["ev_flag"] == "No Play"
        assert row["run_line_ev_flag"] == "No Play"
        assert row["expected_runs"] > 0


def test_market_anchor_stops_flagging_big_dogs(monkeypatch):
    """The load-bearing consequence of market anchoring: a big market underdog
    the sim over-rates (the systematic failure measured 2026-08-16) must not
    surface as a +EV moneyline play, and the published prob must sit near the
    de-vigged market, not the raw sim."""
    monkeypatch.setattr("v2.markets.ev.MONEYLINE_ENABLED", True)
    rng = np.random.default_rng(0)
    # sim says home wins 60% -- compressed vs a market pricing home ~72%
    home_wins = rng.random(4000) < 0.60
    home_runs = np.where(home_wins, 5, 2)
    away_runs = np.where(home_wins, 2, 5)
    kwargs = {
        "game_pk": 2,
        "game_date": np.datetime64("2026-08-16"),
        "start_time": None,
        "home_team": "LAD",
        "away_team": "COL",
        "home_starter": "x",
        "away_starter": "y",
        "home_runs": home_runs,
        "away_runs": away_runs,
        "lineup_source": "lineup_live+queue_live",
        "lineups_locked": False,
        "posterior_age_days": 0,
        "home_wp_p10": 0.55,
        "home_wp_p90": 0.65,
    }
    home, away = build_game_rows(
        **kwargs,
        home_odds={"book": "draftkings", "moneyline": -300},
        away_odds={"book": "draftkings", "moneyline": 250},
    )

    # de-vig: 0.75 / (0.75 + 0.2857) = 0.724; blend pulls published prob toward it
    assert 0.65 < home["win_prob"] < 0.724
    assert abs(home["win_prob"] + away["win_prob"] - 1.0) < 1e-6
    # raw sim would flag the dog (0.40 - 0.2857 = +0.11 edge); anchored must not
    assert away["ev_flag"] == "No Play"
    # bands transform through the same map: still ordered, still anti-correlated
    assert home["win_prob_p10"] <= home["win_prob"] <= home["win_prob_p90"]
    assert abs(away["win_prob_p10"] + home["win_prob_p90"] - 1.0) < 1e-3

    # no odds -> HFA-shifted sim prob passes through (no market to anchor to)
    solo_home, _ = build_game_rows(**kwargs, home_odds=None, away_odds=None)
    assert solo_home["win_prob"] > 0.60  # +0.09 logit HFA on a 0.60 sim prob
    assert solo_home["ev_flag"] == "No Play"
