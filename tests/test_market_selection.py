import numpy as np
import pandas as pd
import pytest

from v2.market_model.features import build_feature_frame
from v2.market_model.residual import prepare_games
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

    selected, p_cover = _best_runline(odds, team_runs, opponent_runs, home=True)

    assert selected["book"] == "fanduel"
    assert selected["spread"] == -1.5
    assert selected["spread_odds"] == 100
    # raw sim cover = 0.25; home gets the +0.09 logit home-field shift
    assert p_cover == pytest.approx(0.2673, abs=1e-4)
    _, p_cover_away = _best_runline(odds, team_runs, opponent_runs, home=False)
    assert p_cover_away == pytest.approx(0.2336, abs=1e-4)

    opposite = _package(
        {"book": "draftkings", "spread": 1.5, "spread_odds": -110},
        {"book": "fanduel", "spread": 1.5, "spread_odds": -120},
    )
    selected, p_cover = _best_runline(odds, team_runs, opponent_runs, home=True, opponent_odds=opposite)
    assert selected["book"] == "fanduel"
    assert p_cover == pytest.approx((0.5 + 0.5 / (0.5 + 120 / 220)) / 2)
    # Changing the simulator cannot reintroduce an unsupported residual edge.
    _, alternative = _best_runline(odds, team_runs + 10, opponent_runs, home=True, opponent_odds=opposite)
    assert alternative == p_cover


def test_fallback_lineup_suppresses_market_flags(monkeypatch):
    odds = {
        "book": "draftkings",
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
    assert all(live_home[key] == value for key, value in odds.items() if key != "book")
    away_odds = {**odds, "moneyline": 130, "spread": 1.5}
    assert all(fallback_away[key] == value for key, value in away_odds.items() if key != "book")
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

    # A one-sided or unpaired price must not turn the unanchored fallback into
    # a +EV recommendation or a positive stake for this same +250 underdog.
    for home_odds, away_odds in (
        (None, {"book": "draftkings", "moneyline": 250}),
        ({"book": "fanduel", "moneyline": -300}, {"book": "draftkings", "moneyline": 250}),
        ({"moneyline": -300}, {"moneyline": 250}),
        ({"book": "draftkings", "moneyline": float("inf")}, {"book": "draftkings", "moneyline": 250}),
    ):
        unpaired = build_game_rows(**kwargs, home_odds=home_odds, away_odds=away_odds)
        assert unpaired[1]["win_prob"] > 0.35
        for row in unpaired:
            assert row["ev_flag"] == "No Play"
            assert pd.isna(row["ml_confidence"])
            assert row["kelly_full_ml"] == row["kelly_quarter_ml"] == 0
            assert row["expected_runs"] > 0

    # Colorado's repeated +1.5 regression: an inflated sim probability must
    # not manufacture a run-line edge against a valid paired market.
    rl_kwargs = {**kwargs, "home_runs": np.r_[np.full(320, 5), np.full(680, 2)],
                 "away_runs": np.r_[np.full(320, 2), np.full(680, 5)]}
    home_quote = {"book": "draftkings", "spread": -1.5, "spread_odds": 120}
    away_quote = {"book": "draftkings", "spread": 1.5, "spread_odds": -142}
    paired = build_game_rows(**rl_kwargs, home_odds=home_quote, away_odds=away_quote)
    expected_cover = (142 / 242) / (142 / 242 + 100 / 220)
    assert paired[1]["p_cover"] == pytest.approx(expected_cover)
    assert paired[0]["p_cover"] + paired[1]["p_cover"] == pytest.approx(1)
    assert paired[1]["run_line_ev_flag"] == "No Play"
    assert paired[1]["kelly_full_rl"] == paired[1]["kelly_quarter_rl"] == 0
    # No moneylines are needed, but missing, invalid, mismatched, and
    # asynchronous opposite run-line quotes must fail closed.
    for opposite in (None, {**home_quote, "book": "fanduel"},
                     {**home_quote, "spread": 1.5},
                     {**home_quote, "spread_odds": float("inf")},
                     {**home_quote, "scraped_at": "2026-09-13T12:00:00Z"}):
        _, unpaired = build_game_rows(**rl_kwargs, home_odds=opposite, away_odds=away_quote)
        assert unpaired["p_cover"] > 0.65
        assert unpaired["run_line_ev_flag"] == "No Play"
        assert pd.isna(unpaired["run_line_confidence"])
        assert unpaired["kelly_full_rl"] == unpaired["kelly_quarter_rl"] == 0


def test_market_research_refuses_independently_shopped_baseline():
    games = pd.DataFrame([{
        "game_pk": 1,
        "game_type": "R",
        "game_date": pd.Timestamp("2026-08-01"),
        "start_time": pd.Timestamp("2026-08-01T19:10:00Z"),
        "home_model_prob": 0.55,
        "home_moneyline": -105,
        "away_moneyline": 120,
        "home_win": 1,
    }])

    with pytest.raises(ValueError, match="paired same-book pregame odds"):
        prepare_games(games)
    with pytest.raises(ValueError, match="paired same-book pregame odds"):
        build_feature_frame(games)

    paired = prepare_games(games.assign(home_market_prob=0.61, probability_source="raw_simulator", model_version="test"))
    assert paired.loc[0, "home_market_prob"] == 0.61
    with pytest.raises(ValueError, match="raw simulator"):
        prepare_games(games.assign(home_market_prob=.61))
    for game_type in ("F", "D", "L", "W", None):
        with pytest.raises(ValueError, match="regular-season"):
            prepare_games(paired.assign(game_type=game_type))

    # The last regular-season result must not seed next year's team-form prior.
    seasons = pd.concat([paired.assign(
        home_team="BOS", away_team="NYY", home_score=8, away_score=1,
        home_expected_runs=4., away_expected_runs=4.,
        home_bp_outs_2d=3, away_bp_outs_2d=3,
        home_win_prob_p10=.4, home_win_prob_p90=.6,
        lineup_source="lineup_live+queue_live", posterior_age_days=1,
    )] * 3, ignore_index=True)
    seasons["game_pk"] = [1, 2, 3]
    seasons["game_date"] = pd.to_datetime(["2026-09-26", "2026-09-27", "2027-04-01"])
    seasons["start_time"] = seasons.game_date + pd.Timedelta(hours=19)
    features = build_feature_frame(seasons)
    assert features.loc[1, "win_form_diff"] > 0
    assert features.loc[2, ["win_form_diff", "run_margin_form_diff", "offense_residual_diff", "defense_residual_diff"]].eq(0).all()
