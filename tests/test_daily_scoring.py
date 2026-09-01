"""Smoke test: end-to-end scoring on a real date.

Uses the live Supabase (read-only against games + probable_starters + odds) and
the 2026 statcast cache. Skips if posteriors aren't built. Always runs with
write=False so it doesn't touch model_outputs.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from v2.bayesian._common import POSTERIORS_DIR

POSTERIORS_PRESENT = (POSTERIORS_DIR / "batter_skill.nc").exists() and (
    POSTERIORS_DIR / "pitcher_skill.nc"
).exists() and (POSTERIORS_DIR / "park_effects.nc").exists()

CACHE_2026 = Path(__file__).resolve().parents[1] / "cache" / "statcast_2026.parquet"


SMOKE_DATE = "2026-04-15"
N_SIMS = 1000


@pytest.mark.skipif(not POSTERIORS_PRESENT, reason="posteriors not built")
@pytest.mark.skipif(not CACHE_2026.exists(), reason="2026 statcast cache missing")
def test_score_games_end_to_end():
    from v2.pipeline.score_games import score

    # SMOKE_DATE is in the past, so freeze_started would drop every game; opt
    # out to exercise the full scoring path (this is the backtest-replay case).
    df = score(SMOKE_DATE, n_sims=N_SIMS, write=False, seed=0, freeze_started=False)
    if df.empty:
        pytest.skip("no games for SMOKE_DATE; pick a different date")

    # Two rows per game.
    games = df["game_pk"].unique()
    assert len(df) == 2 * len(games), f"expected 2 rows per game, got {len(df)} rows for {len(games)} games"

    # Per-game invariants.
    for gp in games:
        g = df[df.game_pk == gp]
        assert len(g) == 2
        # win_prob sums to 1
        assert abs(float(g["win_prob"].sum()) - 1.0) < 1e-6, f"win_prob sum != 1 for game {gp}"
        # our_total = sum of expected_runs
        et_sum = float(g["expected_runs"].sum())
        ot = float(g["our_total"].iloc[0])
        assert abs(et_sum - ot) < 0.01, f"our_total {ot} != sum(expected_runs) {et_sum}"
        # percentile ordering
        for col_root in ("expected_runs", "total"):
            for _, row in g.iterrows():
                p10 = row[f"{col_root}_p10"]
                p50 = row[f"{col_root}_p50"]
                p90 = row[f"{col_root}_p90"]
                assert p10 <= p50 <= p90, f"{col_root} percentile ordering violated"

    # Kelly bounds + numeric finiteness on key cols.
    for col in ("kelly_full_ml", "kelly_quarter_ml", "kelly_full_rl", "kelly_quarter_rl",
                "kelly_full_total", "kelly_quarter_total"):
        vals = df[col].dropna()
        assert (vals >= 0).all() and (vals <= 1).all(), f"{col} out of [0,1]"

    # win_prob in [0,1]
    assert (df["win_prob"] >= 0).all() and (df["win_prob"] <= 1).all()

    # win_prob_p10/p90 from per-posterior-draw sampling. Should be populated,
    # finite, in [0,1], ordered, and anti-correlated across the home/away pair.
    assert df["win_prob_p10"].notna().all() and df["win_prob_p90"].notna().all()
    assert (df["win_prob_p10"] >= 0).all() and (df["win_prob_p90"] <= 1).all()
    assert (df["win_prob_p10"] <= df["win_prob_p90"]).all(), "win_prob_p10 must be <= p90"
    for gp in games:
        rows = list(df[df.game_pk == gp].itertuples(index=False))
        assert abs((rows[0].win_prob_p10 + rows[1].win_prob_p90) - 1.0) < 1e-3
        assert abs((rows[0].win_prob_p90 + rows[1].win_prob_p10) - 1.0) < 1e-3

    # Recommendation fields are strings, never null. Moneyline and run line are
    # enabled; totals remains behind its kill switch.
    for col in ("ev_flag", "total_play", "run_line_ev_flag", "high_variance_flag"):
        assert df[col].notna().all(), f"{col} has nulls"
    valid_flags = set(df["team"]) | {"No Play"}
    assert set(df["ev_flag"]) <= valid_flags
    assert set(df["run_line_ev_flag"]) <= valid_flags
    assert (df["total_play"] == "No Play").all(), "totals bypassed its market kill switch"


def test_is_started_freeze_predicate():
    """The freeze lock: a started game is frozen, a future one isn't, TBD isn't."""
    import pandas as pd

    from v2.pipeline.score_games import is_started

    now = pd.Timestamp("2026-06-05 12:00:00", tz="UTC")
    assert is_started(pd.Timestamp("2026-06-05 01:40:00", tz="UTC"), now) is True
    assert is_started(pd.Timestamp("2026-06-05 23:10:00", tz="UTC"), now) is False
    assert is_started(None, now) is False
    assert is_started(pd.NaT, now) is False
    # tz-naive start_time is coerced to UTC, not crashed on
    assert is_started(pd.Timestamp("2026-06-05 01:40:00"), now) is True


def test_market_research_inputs_are_paired_and_pregame():
    from v2.market_model.features import load_feature_games

    games = load_feature_games("2026-05-12", "2026-05-31")

    assert len(games) >= 100
    assert (games["paired_books"] >= 1).all()
    assert (games["max_pair_lag_seconds"] <= 5).all()
    assert (games["market_quote_at"] < games["start_time"]).all()
    assert (games["home_prediction_at"] < games["start_time"]).all()
    assert (games["away_prediction_at"] < games["start_time"]).all()
    assert games["home_market_prob"].between(0, 1, inclusive="neither").all()


def _odds_response(remaining: int) -> Mock:
    response = Mock()
    response.headers = {
        "x-requests-last": "3",
        "x-requests-used": "103",
        "x-requests-remaining": str(remaining),
    }
    response.json.return_value = [{
        "id": "odds-event-1",
        "home_team": "Los Angeles Dodgers",
        "away_team": "San Diego Padres",
        "commence_time": "2026-08-27T02:10:00Z",
        "bookmakers": [{
            "key": "draftkings",
            "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "Los Angeles Dodgers", "price": -145},
                    {"name": "San Diego Padres", "price": 125},
                ]},
                {"key": "spreads", "outcomes": [
                    {"name": "Los Angeles Dodgers", "price": -105, "point": -1.5},
                    {"name": "San Diego Padres", "price": -115, "point": 1.5},
                ]},
                {"key": "totals", "outcomes": [
                    {"name": "Over", "price": -110, "point": 8.5},
                    {"name": "Under", "price": -110, "point": 8.5},
                ]},
            ],
        }],
    }]
    return response


def test_odds_refresh_routes_and_quota_contract(monkeypatch, tmp_path):
    import pipeline
    from backend.data import odds_api
    from v2.pipeline import daily_run

    slate = pd.DataFrame([{
        "game_pk": 1001,
        "game_date": "2026-08-27",
        "home_team": "LAD",
        "away_team": "SDP",
        "start_time": pd.Timestamp("2026-08-27T02:10:00Z"),
    }])
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    monkeypatch.setenv("ODDS_API_STATE_PATH", str(tmp_path / "odds_api_state.json"))
    monkeypatch.setattr(pipeline, "_upcoming_games", lambda *_: slate)
    fresh_odds = Mock(return_value=False)
    monkeypatch.setattr(pipeline, "_has_recent_stored_odds", fresh_odds)
    stored = []
    monkeypatch.setattr(pipeline, "_replace_odds", lambda frame: stored.append(frame.copy()))
    fetched = []
    real_fetch_odds = pipeline.fetch_odds
    monkeypatch.setattr(
        pipeline,
        "fetch_odds",
        lambda game_pks: fetched.append(real_fetch_odds(game_pks)) or fetched[-1],
    )
    provider_get = Mock(side_effect=[_odds_response(397), _odds_response(52)])
    monkeypatch.setattr(odds_api.requests, "get", provider_get)

    monkeypatch.setattr(daily_run, "update_scores_and_schedule", lambda: None)
    monkeypatch.setattr(daily_run, "update_bullpen_daily", lambda: None)
    monkeypatch.setattr(daily_run, "update_weather_for_date", lambda *_: None)
    monkeypatch.setattr(daily_run, "score", lambda *_args, **_kwargs: pd.DataFrame())
    nightly_names = [name for name, _ in pipeline.NIGHTLY_STEPS]
    assert "Odds" not in nightly_names
    monkeypatch.setattr(pipeline, "NIGHTLY_STEPS", [(name, lambda: None) for name in nightly_names])

    def run_daily(*extra_args):
        monkeypatch.setattr(
            "sys.argv",
            ["daily_run", "--date", "2026-08-27", "--n-sims", "1", *extra_args],
        )
        with pytest.raises(SystemExit) as exc:
            daily_run.main()
        assert exc.value.code == 0

    assert pipeline.nightly() == []
    run_daily()
    assert provider_get.call_count == 1

    fresh_odds.return_value = True
    run_daily("--optional-odds-refresh")
    assert provider_get.call_count == 1

    run_daily()
    assert provider_get.call_count == 2
    assert all(call.kwargs["params"]["markets"] == "h2h,spreads,totals" for call in provider_get.call_args_list)

    fresh_odds.return_value = False
    monkeypatch.setenv("ODDS_API_RESERVE_CREDITS", "50")
    run_daily("--optional-odds-refresh")
    assert provider_get.call_count == 2

    quota = odds_api.latest_quota_state()
    assert {key: quota[key] for key in (
        "x-requests-last", "x-requests-used", "x-requests-remaining"
    )} == {"x-requests-last": 3, "x-requests-used": 103, "x-requests-remaining": 52}
    assert quota["retrieved_at"].endswith("+00:00")
    assert fetched[-1].attrs["quota"] == quota
    assert set(("moneyline", "spread", "spread_odds", "total", "total_over_odds", "total_under_odds")) <= set(stored[0])
    assert stored[0]["moneyline"].tolist() == [-145, 125]
    assert stored[0]["total"].tolist() == [8.5, 8.5]


def test_no_game_route_makes_no_provider_request(monkeypatch, tmp_path):
    import pipeline
    from backend.data import odds_api

    monkeypatch.setenv("ODDS_API_STATE_PATH", str(tmp_path / "odds_api_state.json"))
    monkeypatch.setattr(pipeline, "_upcoming_games", lambda *_: pd.DataFrame())
    provider_get = Mock()
    monkeypatch.setattr(odds_api.requests, "get", provider_get)

    assert pipeline.fetch_and_load_odds("2026-08-27") == 0
    provider_get.assert_not_called()
    assert odds_api.latest_quota_state() is None
