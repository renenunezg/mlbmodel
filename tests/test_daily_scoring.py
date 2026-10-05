"""Smoke test: end-to-end scoring on a real date.

Uses the live Supabase (read-only against games + probable_starters + odds) and
the 2026 statcast cache. Skips if posteriors aren't built. Always runs with
write=False so it doesn't touch model_outputs.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from v2.bayesian._common import POSTERIORS_DIR

POSTERIORS_PRESENT = (POSTERIORS_DIR / "batter_skill.nc").exists() and (
    POSTERIORS_DIR / "pitcher_skill.nc"
).exists() and (POSTERIORS_DIR / "park_effects.nc").exists()

CACHE_2026 = Path(__file__).resolve().parents[1] / "cache" / "statcast_2026.parquet"


SMOKE_DATE = os.getenv("MLBMODEL_SMOKE_DATE", str((pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).date()))
N_SIMS = 1000


@pytest.mark.skipif(not POSTERIORS_PRESENT, reason="posteriors not built")
@pytest.mark.skipif(not CACHE_2026.exists(), reason="2026 statcast cache missing")
@pytest.mark.live_read
def test_score_games_end_to_end(monkeypatch):
    from v2.pipeline import score_games

    simulate_game = score_games.simulate_game

    def simulate_with_trained_park(rng, pm, adv, sub_table, inputs, **kwargs):
        # Live MLB team aliases must resolve to the trained Statcast park, not
        # the neutral fallback. Check the actual daily scoring inputs.
        assert inputs.venue in pm.venue_codes, f"unrecognized park: {inputs.venue}"
        return simulate_game(rng, pm, adv, sub_table, inputs, **kwargs)

    monkeypatch.setattr(score_games, "simulate_game", simulate_with_trained_park)

    # Allow an explicitly selected past date to exercise the read-only replay path.
    df = score_games.score(SMOKE_DATE, n_sims=N_SIMS, write=False, seed=0, freeze_started=False,
                           posteriors_dir=Path(os.getenv("MLBMODEL_TEST_POSTERIORS", str(POSTERIORS_DIR))))
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
    for gp in games:
        rows = list(df[df.game_pk == gp].itertuples(index=False))
        context = rows[0].prediction_context
        assert context == rows[1].prediction_context
        assert context["model_version"] == "sim-v3"
        assert context["training_max_date"] < SMOKE_DATE
        assert context["tables_training_max_date"] < SMOKE_DATE
        if context["uncertainty"]["status"] == "below_mc_resolution":
            assert all(pd.isna(r.win_prob_p10) and pd.isna(r.win_prob_p90) for r in rows)
        else:
            assert all(0 <= r.win_prob_p10 <= r.win_prob_p90 <= 1 for r in rows)
            assert abs(rows[0].win_prob_p10 + rows[1].win_prob_p90 - 1) < 1e-3
        for side in ("home", "away"):
            assert abs(sum(context[f"{side}_run_distribution"].values()) - 1) < 1e-6
        joint = context["joint_adjustment"]
        raw = np.array(joint["raw_joint_counts"])
        assert raw[:, 2].sum() == joint["sample_count"]
        for i, side in enumerate(("home", "away")):
            assert context[f"raw_{side}_expected_runs"] == pytest.approx(raw[:, i] @ raw[:, 2] / raw[:, 2].sum())
        if joint["status"] == "adjusted":
            adjusted = np.array(joint["joint_distribution"])
            weights = adjusted[:, 2]
            assert weights.sum() == pytest.approx(1)
            assert joint["target_error"] <= 1e-6
            for i, row in enumerate(rows):
                assert row.expected_runs == pytest.approx(weights @ adjusted[:, i], abs=5.1e-5)
                assert row.win_prob == pytest.approx(weights @ (adjusted[:, i] > adjusted[:, 1-i]), abs=5.1e-5)
        else:
            assert all(r.ev_flag == r.run_line_ev_flag == r.total_play == "No Play" for r in rows)

    # Recommendation fields are strings, never null. Moneyline and run line are
    # enabled; totals remains behind its kill switch.
    for col in ("ev_flag", "total_play", "run_line_ev_flag", "high_variance_flag"):
        assert df[col].notna().all(), f"{col} has nulls"
    valid_flags = set(df["team"]) | {"No Play"}
    assert set(df["ev_flag"]) <= valid_flags
    assert set(df["run_line_ev_flag"]) <= valid_flags
    assert (df["total_play"] == "No Play").all(), "totals bypassed its market kill switch"


def test_postseason_schedule_to_scoring(monkeypatch, tmp_path):
    """Official game types survive ingestion, scoring, and frozen forecast context."""
    from sqlalchemy import create_engine, text

    from backend.data import mlb_api
    from pipeline import _batch_upsert_games
    from v2.data.pa_dataset import OUTCOMES
    from v2.pipeline import score_games
    from v2.simulator import bullpen
    from v2.simulator.posteriors import K_FREE, PosteriorMeans

    games = []
    for gp, game_type in enumerate(("R", "F", "D", "L", "W", "S", "A"), 1):
        games.append({
            "gamePk": gp, "gameType": game_type, "officialDate": "2030-10-01",
            "gameDate": "2030-10-01T23:00:00Z",
            "status": {"abstractGameState": "Preview", "detailedState": "Scheduled"},
            "teams": {
                "home": {"team": {"abbreviation": "BOS"},
                         "probablePitcher": {"id": 10, "fullName": "Home Starter"}},
                "away": {"team": {"abbreviation": "NYY"},
                         "probablePitcher": {"id": 20, "fullName": "Away Starter"}},
            },
        })
    games.extend([
        {**games[1], "gamePk": 8, "status": {"abstractGameState": "Preview", "detailedState": "Cancelled"}},
        {**games[1], "gamePk": 9, "ifNecessary": "Y"},
        {**games[1], "gamePk": 10, "teams": {
            "home": {"team": {"abbreviation": "AL Higher Seed"}},
            "away": {"team": {"abbreviation": "AL Lower Seed"}},
        }},
    ])
    response = Mock()
    response.json.return_value = {"dates": [{"date": "2030-10-01", "games": games}]}
    get = Mock(return_value=response)
    monkeypatch.setattr(mlb_api.requests, "get", get)
    monkeypatch.setattr(mlb_api, "_batch_fetch_handedness", lambda *_: {})
    schedule = mlb_api.fetch_schedule(date(2030, 10, 1))
    assert get.call_args.kwargs["params"]["gameTypes"] == "R,F,D,L,W"
    assert set(schedule.game_pk) == {1, 2, 3, 4, 5, 8, 9}
    assert schedule.set_index("game_pk").loc[9, "status"] == "If Necessary"

    # Exercise the real upsert and scoring SELECT using an isolated database.
    db = create_engine("sqlite://")
    with db.begin() as conn:
        conn.connection.create_function("now", 0, lambda: "2030-09-30")
        conn.execute(text("""CREATE TABLE games (
            game_pk INTEGER PRIMARY KEY, game_date TEXT, game_type TEXT,
            home_team TEXT, away_team TEXT, home_score INTEGER, away_score INTEGER,
            status TEXT, venue TEXT, start_time TEXT, updated_at TEXT)"""))
        _batch_upsert_games(conn, schedule)
        _batch_upsert_games(conn, schedule)  # Refresh uses the same upsert contract.
        conn.execute(text("UPDATE games SET game_type = NULL WHERE game_pk = 1"))
    monkeypatch.setattr(score_games, "engine", db)
    starters = mlb_api.fetch_probable_starters(date(2030, 10, 1), days_ahead=0)
    monkeypatch.setattr(score_games, "fetch_starters", lambda *_: starters)
    monkeypatch.setattr(score_games, "fetch_odds", lambda *_: pd.DataFrame(columns=["game_pk", "team"]))
    monkeypatch.setattr(score_games, "fetch_weather", lambda *_: pd.DataFrame())
    assert [c.game_type for c in score_games.build_contexts("2030-10-01")] == ["F", "D", "L", "W"]
    with db.begin() as conn:
        _batch_upsert_games(conn, schedule.iloc[:1])

    rates = {"K": .22, "BB": .08, "HBP": .01, "1B": .15, "2B": .05, "3B": .005, "HR": .035, "OUT": .45}
    pm = PosteriorMeans(
        intercept=np.array([np.log(rates[o] / rates["OUT"]) for o in OUTCOMES if o != "OUT"]),
        batter_offset=np.zeros((2, K_FREE)), platoon_offset=np.zeros((2, K_FREE)),
        pitcher_offset=np.zeros((2, 2, K_FREE)), park_log=np.zeros(2),
        batter_ids=np.array([1]), pitcher_ids=np.array([10]), venue_codes=np.array(["BOS"]),
    )
    from v2.simulator.baserunner import load_advancement_table, load_out_subtype_table
    from v2.simulator.build_advancement_table import build_advancement, build_out_subtype

    transitions = pd.DataFrame([{
        "state": 0, "outs": 0, "outcome_idx": OUTCOMES.index("OUT"),
        "subtype_key": "field_out", "new_state": 0, "runs": 0, "outs_added": 1,
        "b_q": 0, "p_q": 0, "game_date": "2029-09-30",
    }])
    build_advancement(transitions).to_parquet(tmp_path / "advancement.parquet")
    build_out_subtype(transitions).to_parquet(tmp_path / "out_subtype.parquet")
    monkeypatch.setattr(score_games, "load_advancement_table", lambda: load_advancement_table(tmp_path))
    monkeypatch.setattr(score_games, "load_out_subtype_table", lambda: load_out_subtype_table(tmp_path))
    monkeypatch.setattr(score_games, "N_DRAWS", 3)
    monkeypatch.setattr(score_games, "load_posterior_draws", lambda *_args, **_kwargs: [pm, pm, pm])
    monkeypatch.setattr(score_games, "posterior_provenance", lambda *_: {
        "model_version": "fixture", "training_max_date": "2029-09-30",
    })
    cache = pd.DataFrame(columns=["game_date", "events", "inning_topbot", "home_team", "away_team", "batter", "pitcher", "p_throws"])
    monkeypatch.setattr(score_games, "load_cache_for_year", lambda *_: cache)
    monkeypatch.setattr(score_games, "fetch_lineups_for_games", lambda ids: {
        gp: {"home": list(range(1, 10)), "away": list(range(11, 20))} for gp in ids
    })
    roster_ids = [101, 102, 103, 201, 202, 203, 204, 205]
    workload = pd.DataFrame([
        {"game_date": date(2030, 9, d), "team": team, "pitcher_id": pid, "outs": outs, "role": role}
        for team in ("BOS", "NYY") for d in (20, 24)
        for pid, outs, role in ((10, 15, "SP"), (20, 15, "SP"),
                               (101, 3, "RP"), (102, 3, "RP"), (103, 3, "RP"),
                               (201, 15, "SP"), (202, 15, "SP"), (203, 15, "SP"), (204, 15, "SP"))
    ] + [
        {"game_date": date(2030, 9, 30), "team": team, "pitcher_id": pid, "outs": outs, "role": role}
        for team in ("BOS", "NYY") for pid, outs, role in ((102, 6, "RP"), (203, 15, "SP"), (201, 2, "RP"))
    ])
    monkeypatch.setattr(bullpen, "_load_workload", lambda *_: workload)
    monkeypatch.setattr(mlb_api, "fetch_active_pitchers", lambda *_: [10, 20, *roster_ids, 999])
    boxscore = Mock()
    boxscore.json.return_value = {"teams": {
        side: {"pitchers": [], "bullpen": [starter, *roster_ids, 300], "bench": list(range(401, 417)), "players": {
            f"ID{pid}": {"position": {"abbreviation": "TWP" if pid == 204 else "DH" if pid >= 300 else "P"}}
            for pid in [starter, *roster_ids, 300, *range(401, 417)]
        }} for side, starter in (("home", 10), ("away", 20))
    }}
    get.return_value = boxscore
    monkeypatch.setattr(mlb_api, "fetch_probable_starters", lambda *_args, **_kwargs: pd.DataFrame({"pitcher_id": [202]}))
    daily_writer = Mock()
    monkeypatch.setattr(score_games, "publish_forecasts", daily_writer)

    result = score_games.score("2030-10-01", n_sims=600, write=False)
    assert len(result) == 10
    assert set(result.game_pk) == {1, 2, 3, 4, 5}
    for gp, rows in result.groupby("game_pk"):
        context = rows.iloc[0].prediction_context
        assert context["game_type"] == games[gp - 1]["gameType"]
        assert context["automatic_runner"] is (gp == 1)
        assert np.isfinite(rows.expected_runs).all()
        assert abs(rows.win_prob.sum() - 1) < 1e-6
        assert rows.total_play.eq("No Play").all()
        assert "0" not in context["margin_distribution"]
        queue = context["home_queue"]
        if gp == 1:
            assert queue["relievers"] == [101, 103, 0]
            assert queue["roster_source"] == "active_roster"
            assert queue["availability"][102] == "recent_workload"
            assert queue["availability_assumption"] == "recent_workload"
        else:
            assert queue["relievers"] == [102, 101, 103, 201, 203, 204, 205, 0]
            assert queue["roster_source"] == "game_boxscore"
            assert queue["availability"][102] == "relief"
            assert queue["availability"][202] == "scheduled_starter"
            assert queue["availability"][203] == "emergency_relief"
            assert queue["availability_assumption"] == "full_rest"
            assert queue["workloads"][201] == (3,)  # Recent SP does not become a guessed bulk arm.
            assert 300 not in queue["relievers"] and 999 not in queue["relievers"]
    daily_writer.assert_not_called()

    # No game roster means neutral coverage, never fallback to September's roster.
    get.side_effect = RuntimeError("roster unavailable")
    missing = bullpen.build_queues_live(date(2030, 10, 1), [bullpen.LiveQueueContext(2, "home", "BOS", 10, "F")])[(2, "home")]
    assert missing.relievers == [0] and not missing.usage_known
    assert missing.roster_source == "unknown"
    get.side_effect = None
    for team in boxscore.json.return_value["teams"].values():
        team["bench"].extend([417, 418])
    unconfirmed = bullpen.build_queues_live(date(2030, 10, 1), [bullpen.LiveQueueContext(2, "home", "BOS", 10, "F")])[(2, "home")]
    assert unconfirmed.relievers == [0] and not unconfirmed.usage_known
    for team in boxscore.json.return_value["teams"].values():
        team["bench"] = team["bench"][:-2]

    # A roster change alone triggers a pregame refresh with the same batting order.
    import json

    from v2.pipeline import refresh_lineups

    live_lineup = {"home": list(range(1, 10)), "away": list(range(11, 20))}
    stored_context = json.loads(json.dumps(result[result.game_pk == 2].iloc[0].prediction_context))
    refresh_games = schedule[schedule.game_pk == 2].copy()
    refresh_games["stored_hash"] = score_games.lineup_hash(live_lineup)
    refresh_games["prediction_context"] = [stored_context]
    monkeypatch.setattr(refresh_lineups, "_fetch_scheduled_games", lambda *_: refresh_games)
    monkeypatch.setattr(refresh_lineups, "_starter_map", lambda *_: {2: (10, 20)})
    monkeypatch.setattr(refresh_lineups, "upsert_probable_starters", lambda *_: None)
    monkeypatch.setattr(refresh_lineups, "fetch_and_load_odds", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(refresh_lineups, "fetch_lineup", lambda *_: live_lineup)
    monkeypatch.setattr("backend.data.weather.fetch_weather", lambda *_: None)
    refreshed = Mock()
    monkeypatch.setattr(refresh_lineups, "score", refreshed)
    monkeypatch.setattr("sys.argv", ["refresh_lineups", "--date", "2030-10-01"])
    refresh_lineups.main()
    refreshed.assert_not_called()
    for team in boxscore.json.return_value["teams"].values():
        team["bullpen"].remove(205)
    refresh_lineups.main()
    assert refreshed.call_args.kwargs["game_pks"] == [2]
    assert refreshed.call_args.kwargs["update_season"] is True

    # Reproduce the live regression through provider parsing, refresh dispatch,
    # simulation, and both output writers, with no production writes.
    import pipeline
    from backend.data import odds_api

    for team in boxscore.json.return_value["teams"].values():
        team["bullpen"].append(205)
    response = _odds_response(397)
    event = response.json.return_value[0]
    event.update(home_team="Boston Red Sox", away_team="New York Yankees",
                 commence_time="2030-10-01T23:00:00Z")
    for market in event["bookmakers"][0]["markets"]:
        for outcome in market["outcomes"]:
            outcome["name"] = {"Los Angeles Dodgers": "Boston Red Sox",
                               "San Diego Padres": "New York Yankees"}.get(outcome["name"], outcome["name"])
    # MLB and Odds API share requests; retain each provider's response.
    provider_get = Mock(return_value=response)
    monkeypatch.setattr(odds_api.requests, "get", lambda url, **kwargs:
                        provider_get(url, **kwargs) if url.startswith(odds_api.ODDS_API_BASE) else boxscore)
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    monkeypatch.setenv("ODDS_API_STATE_PATH", str(tmp_path / "odds-state.json"))
    monkeypatch.setattr(pipeline, "_upcoming_games", lambda *_: refresh_games)
    recent = Mock(return_value=False)
    monkeypatch.setattr(pipeline, "_has_recent_stored_odds", recent)
    persisted = [pd.DataFrame(columns=["game_pk", "team"])]
    monkeypatch.setattr(pipeline, "_replace_odds", lambda frame: persisted.__setitem__(0, frame.copy()))
    monkeypatch.setattr(refresh_lineups, "fetch_and_load_odds", pipeline.fetch_and_load_odds)
    monkeypatch.setattr(refresh_lineups, "fetch_odds", lambda *_: persisted[0])
    monkeypatch.setattr(score_games, "fetch_odds", lambda *_: persisted[0])
    monkeypatch.setattr(refresh_lineups, "score", score_games.score)
    monkeypatch.setattr("sys.argv", ["refresh_lineups", "--date", "2030-10-01", "--n-sims", "600"])

    def check_written(priced):
        rows = daily_writer.call_args.args[0]
        assert {r["game_pk"] for r in rows} == {2}
        assert all(pd.notna(r["moneyline"]) == priced for r in rows)
        assert all(pd.notna(r["total"]) == priced for r in rows)
        refresh_games.at[refresh_games.index[0], "prediction_context"] = json.loads(json.dumps(rows[0]["prediction_context"]))
        daily_writer.reset_mock()

    refresh_lineups.main()  # Missing odds arrive without a lineup change.
    assert provider_get.call_count == 1
    check_written(True)
    recent.return_value = True
    refresh_lineups.main()  # Fresh, unchanged offers consume no quota or rescore.
    assert provider_get.call_count == 1
    daily_writer.assert_not_called()

    # A price update already ingested by another path must also trigger scoring.
    persisted[0].loc[persisted[0].team == "BOS", "moneyline"] = -135
    refresh_lineups.main()
    check_written(True)

    persisted[0]["scraped_at"] = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=3)
    recent.return_value = False
    refresh_lineups.main()  # Aging quotes are replaced before scoring.
    assert provider_get.call_count == 2
    check_written(True)

    # Reserve exhaustion expires displayed prices once, then stays stable.
    persisted[0]["scraped_at"] = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=3)
    odds_api._write_state({"x-requests-remaining": 52, "x-requests-last": 3})
    refresh_lineups.main()
    assert provider_get.call_count == 2
    check_written(False)
    refresh_lineups.main()
    daily_writer.assert_not_called()

    # Provider failure must not block a changed lineup or revive stale prices.
    odds_api._write_state({"x-requests-remaining": 397, "x-requests-last": 3})
    provider_get.side_effect = refresh_lineups.RequestException("provider unavailable")
    refresh_games["stored_hash"] = "changed-lineup"
    refresh_lineups.main()
    check_written(False)

    # A prior year's playoff workloads cannot create next season's pitching roles.
    rollover = bullpen.queues_from_workload(
        date(2031, 1, 1), [bullpen.LiveQueueContext(2, "home", "BOS", 10)],
        workload.assign(game_date=date(2030, 12, 31)), {"BOS": [10, *roster_ids]},
    )[(2, "home")]
    assert rollover.relievers == [0] and rollover.team_outs_2d is None

    with db.begin() as conn:
        conn.execute(text("UPDATE games SET start_time = '2000-10-01T23:00:00Z'"))
    assert score_games.score("2030-10-01", n_sims=600, write=True).empty
    daily_writer.assert_not_called()
    db.dispose()


def test_confirmed_pitching_plan_preserves_opener_and_bulk_roles():
    from datetime import date

    from v2.pipeline.score_games import GameContext, build_inputs, top9_batters_by_team
    from v2.simulator.bullpen import LiveQueueContext, PitchingPlan, apply_pitching_plan, queues_from_workload

    cache = pd.DataFrame({"events": ["single"] * 18, "inning_topbot": ["Bot"] * 9 + ["Top"] * 9,
                          "home_team": ["SD"] * 18, "away_team": ["LAD"] * 18, "batter": list(range(1, 19))})
    ctx = GameContext(823932, pd.Timestamp("2026-07-04"), pd.Timestamp("2026-07-05T01:00Z"),
                      "SDP", "LAD", 593974, 808967, "Wandy Peralta", "Yoshinobu Yamamoto", "L", "R", None, None)
    inputs, _, _ = build_inputs(ctx, [], [], top9_batters_by_team(cache), {}, {})
    assert inputs.home_lineup.tolist() == list(range(1, 10))
    assert inputs.away_lineup.tolist() == list(range(10, 19))
    assert inputs.home_queue.outs_samples(593974, 0) == (3,)

    workload = pd.DataFrame([
        {"game_date": date(2026, 7, d), "pitcher_id": pid, "team": "SDP", "outs": outs, "role": role}
        for d in (1, 2) for pid, outs, role in ((593974, 3, "RP"), (656288, 15, "SP"), (999, 3, "RP"))
    ])
    context = LiveQueueContext(823932, "home", "SDP", 593974)
    queue = queues_from_workload(date(2026, 7, 4), [context], workload, {"SDP": [593974, 656288, 999]})[(823932, "home")]
    assert 656288 not in queue.relievers  # A rested starter is not a guessed bulk arm.
    assert queue.starter_role == 1 and not queue.usage_known
    # A fixture report demonstrates the confirmed-plan contract; it is not a historical source claim.
    plan = PitchingPlan(823932, "home", 593974, 656288, 3, 12, "fixture:confirmed-report", "2026-07-04T18:00Z")
    assert plan.valid_for(context, ctx.start_time)
    queue = apply_pitching_plan(queue, plan)
    assert queue.relievers[0] == 656288 and queue.workloads[593974] == (3,) and queue.workloads[656288] == (12,)
    assert queue.usage_known
    from dataclasses import replace
    assert not replace(plan, confirmed_at="2026-07-05T03:00Z").valid_for(context, ctx.start_time)


@pytest.mark.live_read
def test_market_research_inputs_are_paired_and_pregame():
    from v2.market_model.features import load_feature_games

    games = load_feature_games("2026-05-12", "2026-05-31")

    if games.empty:
        pytest.skip("Legacy forecasts have no raw-probability provenance; no valid research sample yet")
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
    cutoff = fresh_odds.call_args.args[1]
    assert pd.Timedelta(minutes=39) < pd.Timestamp.now(tz="UTC") - cutoff < pd.Timedelta(minutes=41)

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
    from v2.pipeline import refresh_lineups

    monkeypatch.setenv("ODDS_API_STATE_PATH", str(tmp_path / "odds_api_state.json"))
    monkeypatch.setattr(pipeline, "_upcoming_games", lambda *_: pd.DataFrame())
    provider_get = Mock()
    monkeypatch.setattr(odds_api.requests, "get", provider_get)

    assert pipeline.fetch_and_load_odds("2026-08-27") == 0
    monkeypatch.setattr(refresh_lineups, "_fetch_scheduled_games", lambda *_: pd.DataFrame())
    monkeypatch.setattr("sys.argv", ["refresh_lineups", "--date", "2026-08-27"])
    refresh_lineups.main()
    provider_get.assert_not_called()
    assert odds_api.latest_quota_state() is None
