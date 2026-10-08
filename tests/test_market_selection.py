import numpy as np
import pandas as pd
import pytest

from v2.market_model.features import build_feature_frame
from v2.market_model.residual import prepare_games
from v2.markets.writer import build_game_rows


def _package(*offers):
    return {**offers[0], "offers": list(offers)}


def _fresh(offer):
    return {**offer, "scraped_at": (pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=10)).isoformat()} if offer else None


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
    home_runs = np.r_[np.full(500, 6), np.full(400, 4), np.full(100, 2)]
    away_runs = np.r_[np.full(900, 3), np.full(100, 5)]
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
        "home_odds": _fresh(odds),
        "away_odds": _fresh({**odds, "moneyline": 130, "spread": 1.5}),
        "lineup_source": "lineup_top9+queue_cache",
        "lineups_locked": False,
        "posterior_age_days": 0,
    }
    monkeypatch.setattr("v2.markets.ev.MONEYLINE_ENABLED", True)
    monkeypatch.setattr("v2.markets.ev.RUNLINE_ENABLED", True)
    shopping = _package(_fresh(odds),
                        _fresh({**odds, "book": "fanduel", "moneyline": -160, "spread_odds": 110}),
                        _fresh({**odds, "book": "betmgm", "spread": -2.5, "spread_odds": 180, "moneyline": -170}))
    live_home, _ = build_game_rows(**{**kwargs, "home_odds": shopping}, lineups_live=True)
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

    # All three markets constrain actual score pairs, including integer-total
    # pushes. Verify the published rows rather than the optimizer internals.
    rng = np.random.default_rng(91)
    h, a = rng.poisson(4.5, 20000), rng.poisson(4.1, 20000)
    tied = h == a
    h[tied] += 1
    home_offer = _fresh({**odds, "moneyline": -140, "spread_odds": 150})
    away_offer = _fresh({**odds, "moneyline": 125, "spread": 1.5, "spread_odds": -170})
    joint_kwargs = {**kwargs, "home_runs": h, "away_runs": a,
                    "home_odds": home_offer, "away_odds": away_offer}
    rows = build_game_rows(**joint_kwargs)
    context = rows[0]["prediction_context"]
    joint = context["joint_adjustment"]
    assert context == rows[1]["prediction_context"]
    assert joint["status"] == "adjusted"
    assert {t["market"] for t in joint["targets"]} == {"moneyline", "runline", "total"}
    raw = np.array(joint["raw_joint_counts"])
    assert raw[:, 2].sum() == len(h)
    assert raw[:, 0] @ raw[:, 2] / len(h) == h.mean()
    scores = np.array(joint["joint_distribution"])
    weights = scores[:, 2]
    margin, total = scores[:, 0] - scores[:, 1], scores[:, :2].sum(axis=1)
    assert weights.sum() == pytest.approx(1)
    for target in joint["targets"]:
        if target["market"] == "moneyline":
            actual = weights @ (margin > 0)
        elif target["market"] == "runline":
            actual = weights @ (margin > -target["point"])
        else:
            actual = weights @ (total > target["point"]) / (weights @ (total != target["point"]))
        assert actual == pytest.approx(target["target_probability"], abs=1e-6)
    for side, row in enumerate(rows):
        assert row["expected_runs"] == pytest.approx(weights @ scores[:, side], abs=5e-5)
        assert row["win_prob"] == pytest.approx(weights @ (margin * (1 if side == 0 else -1) > 0), abs=5e-5)
        assert row["p_cover"] == pytest.approx(weights @ (margin * (1 if side == 0 else -1) > -row["spread"]), abs=1e-6)
        assert row["p_over"] == pytest.approx(weights @ ((total > row["total"]) + .5 * (total == row["total"])))
        assert row["p_under"] + row["p_over"] == pytest.approx(1)
        expected_hist = np.bincount(np.minimum(scores[:, side], 20).astype(int), weights=weights, minlength=21)
        assert row["runs_hist"] == pytest.approx(expected_hist, abs=5.1e-6)
        assert row["total_play"] == "No Play"
        assert row["kelly_full_total"] == row["kelly_quarter_total"] == 0
    assert rows[0]["expected_runs"] != round(h.mean(), 4)

    # Contradictory targets (cover -1.5 more likely than a win) cannot leave
    # a partially adjusted forecast or recommendation behind.
    rejected = build_game_rows(**{**joint_kwargs,
        "home_odds": {**home_offer, "spread_odds": -900},
        "away_odds": {**away_offer, "spread_odds": 700}})
    assert rejected[0]["prediction_context"]["joint_adjustment"]["reason"] == "infeasible_or_unconverged_targets"
    for row in rejected:
        assert row["ev_flag"] == row["run_line_ev_flag"] == "No Play"
        assert row["kelly_full_ml"] == row["kelly_full_rl"] == 0

    # Stale, future, missing, and malformed timestamps never become anchors.
    now = pd.Timestamp.now(tz="UTC")
    for stamp in (now - pd.Timedelta(hours=25), now + pd.Timedelta(hours=1), None, "invalid"):
        rejected = build_game_rows(**{**joint_kwargs,
            "home_odds": {**home_offer, "scraped_at": stamp},
            "away_odds": {**away_offer, "scraped_at": stamp}})
        assert rejected[0]["prediction_context"]["market_home_win_prob"] is None
        assert len(rejected[0]["prediction_context"]["joint_adjustment"]["targets"]) == 1
        assert all(row["ev_flag"] == row["run_line_ev_flag"] == "No Play" for row in rejected)
    # The day's single odds pull must survive a re-score hours later (2026-10-08 regression).
    aged = now - pd.Timedelta(hours=8)
    kept = build_game_rows(**{**joint_kwargs,
        "home_odds": {**home_offer, "scraped_at": aged},
        "away_odds": {**away_offer, "scraped_at": aged}})
    assert kept[0]["prediction_context"]["market_home_win_prob"] is not None
    assert all(row["moneyline"] is not None for row in kept)

    # Run-line and total pairs remain usable without any moneyline prices.
    independent = build_game_rows(**{**joint_kwargs,
        "home_odds": {**home_offer, "moneyline": None},
        "away_odds": {**away_offer, "moneyline": None}})
    joint = independent[0]["prediction_context"]["joint_adjustment"]
    assert joint["status"] == "adjusted"
    assert len(joint["targets"]) == 3
    assert joint["targets"][0]["market_probability"] is None

    # Different bookmakers can quote different totals. Each conditional target
    # must hold in the same distribution, including a totals-only provider.
    extra = _fresh({"book": "fanduel", "total": 8.5, "total_over_odds": -110, "total_under_odds": -110})
    multiple = build_game_rows(**{**joint_kwargs, "away_odds": _package(away_offer, extra)})
    joint = multiple[0]["prediction_context"]["joint_adjustment"]
    assert joint["status"] == "adjusted"
    assert [t["point"] for t in joint["targets"] if t["market"] == "total"] == [8, 8.5]
    assert joint["target_error"] <= 1e-6

    # Mathematically feasible targets can still put unreasonable weight on
    # rare simulations. That forecast is rejected rather than extrapolated.
    rare = build_game_rows(**{**joint_kwargs,
        "home_runs": np.r_[np.full(20, 5), np.full(19980, 2)],
        "away_runs": np.r_[np.full(20, 2), np.full(19980, 5)],
        "home_odds": _fresh({"book": "draftkings", "moneyline": -110}),
        "away_odds": _fresh({"book": "draftkings", "moneyline": -110})})
    assert rare[0]["prediction_context"]["joint_adjustment"]["reason"] == "excessive_reweighting"
    assert all(row["ev_flag"] == row["run_line_ev_flag"] == "No Play" for row in rare)


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
        home_odds=_fresh({"book": "draftkings", "moneyline": -300}),
        away_odds=_fresh({"book": "draftkings", "moneyline": 250}),
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
        unpaired = build_game_rows(**kwargs, home_odds=_fresh(home_odds), away_odds=_fresh(away_odds))
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
    home_quote = _fresh({"book": "draftkings", "spread": -1.5, "spread_odds": 120})
    away_quote = _fresh({"book": "draftkings", "spread": 1.5, "spread_odds": -142})
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


def test_market_research_refuses_independently_shopped_baseline(monkeypatch):
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

    # Market-residual fitting must not relabel joint-adjusted predictions as
    # independent simulator evidence, even though those are the published rows.
    from v2.market_model import residual
    from v2.markets.probs import paired_market_quotes

    home = _fresh({"book": "draftkings", "moneyline": -130, "spread": -1.5, "spread_odds": 150})
    away = _fresh({"book": "draftkings", "moneyline": 120, "spread": 1.5, "spread_odds": -170})
    frozen = paired.assign(home_spread=-1.5, away_spread=1.5, home_spread_odds=150, away_spread_odds=-170,
                           home_cover_prob=.9, home_expected_runs=99., away_expected_runs=99.,
                           home_score=5, away_score=3)
    frozen["prediction_context"] = [{
        "forecast_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "market_pairs": paired_market_quotes(home, away), "joint_adjustment": {"status": "adjusted"},
        "raw_home_expected_runs": 4.1, "raw_away_expected_runs": 3.9,
        "margin_distribution": {"-3": .45, "1": .3, "3": .25},
    }]
    monkeypatch.setattr(residual, "load_frozen_games", lambda *_: frozen)
    research = residual.load_runline_games("2026-01-01", "2026-12-31")
    assert research.loc[0, "home_model_prob"] == .25
    assert research.loc[0, "home_expected_runs"] == 4.1
    assert research.loc[0, "away_expected_runs"] == 3.9
