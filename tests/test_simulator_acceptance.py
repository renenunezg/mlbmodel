"""Acceptance gates for the v2 plate-appearance and game simulators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_live_probability_feed_endings_and_frozen_forecast_separation(tmp_path, monkeypatch):
    import json
    import sys
    from contextlib import contextmanager
    from copy import deepcopy
    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace

    from v2.live.feed import score_feed
    from v2.live.model import GameState, WinExpectancy

    distributions = np.broadcast_to([.8, .15, .05], (2, 3, 8, 4, 3, 3)).copy()
    distributions[:, :, 2] = [.2, .65, .15]
    model = WinExpectancy(distributions, {"version": "acceptance"})
    model.save(tmp_path / "model.json")
    model = WinExpectancy.load(tmp_path / "model.json")
    feed = {
        "gamePk": 123,
        "metaData": {"timeStamp": "20260930_190000"},
        "gameData": {"game": {"type": "R"}, "status": {"abstractGameState": "Live"}},
        "liveData": {"linescore": {
            "scheduledInnings": 9, "currentInning": 9, "inningState": "End",
            "outs": 3, "balls": 0, "strikes": 3, "offense": {},
            "teams": {"home": {"runs": 2}, "away": {"runs": 2}},
        }, "plays": {"allPlays": []}},
    }
    now = datetime(2026, 9, 30, 19, 0, 30, tzinfo=UTC)
    untouched = deepcopy(feed)
    result = score_feed(feed, model, fetched_at=now)
    assert feed == untouched  # No mutation of source or pregame forecast objects.
    assert result["state"] == {
        "inning": 10, "top": True, "outs": 0, "bases": 2, "home_score": 2,
        "away_score": 2, "balls": 0, "strikes": 0, "automatic_runner": True,
    }
    assert result["home_win_probability"] + result["away_win_probability"] == 1
    # Identical staffs starting the 10th with identical automatic runners are 50/50.
    # The bottom half must receive its runner too, not just the observed top half.
    assert result["home_win_probability"] == pytest.approx(.5)
    assert not result["stale"]
    feed["gameData"]["game"]["type"] = "D"
    postseason = score_feed(feed, model, fetched_at=now)
    assert postseason["state"]["bases"] == 0 and not postseason["state"]["automatic_runner"]

    ls = feed["liveData"]["linescore"]
    # Home ahead after the top ninth: no bottom half should be simulated.
    ls.update(inningState="Middle")
    ls["teams"]["home"]["runs"] = 3
    assert score_feed(feed, model, fetched_at=now)["home_win_probability"] == 1
    # The same lead before the third out is not a finished game.
    ls.update(inningState="Top", outs=2, strikes=2)
    assert 0 < score_feed(feed, model, fetched_at=now)["home_win_probability"] < 1
    # Walkoff before three outs and an away win after the bottom ninth.
    ls.update(inningState="Bottom", outs=0)
    assert score_feed(feed, model, fetched_at=now)["home_win_probability"] == 1
    ls.update(inningState="End", outs=3)
    ls["teams"]["away"]["runs"] = 4
    assert score_feed(feed, model, fetched_at=now)["home_win_probability"] == 0
    feed["gameData"]["status"]["abstractGameState"] = "Final"
    assert score_feed(feed, model, fetched_at=now)["home_win_probability"] == 0
    ls["currentInning"] = 7
    called = score_feed(feed, model, fetched_at=now)
    assert called["history"][-1]["label"] == "Final"
    assert called["history"][-1]["home_win_probability"] == 0

    feed["gameData"]["status"]["abstractGameState"] = "Live"
    ls.update(inningState="Top", outs=0, balls=4)
    invalid = score_feed(feed, model, fetched_at=now)
    assert invalid["home_win_probability"] is None and invalid["unavailable_reason"]
    ls["balls"] = 0
    assert score_feed(feed, model, fetched_at=datetime(2026, 9, 30, 19, 5, tzinfo=UTC))["stale"]
    del ls["outs"]
    assert score_feed(feed, model, fetched_at=now)["home_win_probability"] is None
    with pytest.raises(ValueError, match="integers"):
        GameState(9, True, 0, 0, 1.5, 0)

    # Exercise the publication path without a database: suppress clock-only
    # updates, retain freshness, and retry a batch whose commit failed.
    from v2.live.publish import publish_snapshots

    commits = []
    fail_commit = False

    @contextmanager
    def transaction():
        batch = []

        def execute(statement, params):
            assert "INSERT INTO live_win_probability" in str(statement)
            assert "model_outputs" not in str(statement)
            batch.extend(json.loads(params["snapshots"]))

        yield SimpleNamespace(execute=execute)
        if fail_commit:
            raise RuntimeError("commit failed")
        commits.append(batch)

    monkeypatch.setitem(sys.modules, "backend.db", SimpleNamespace(engine=SimpleNamespace(begin=transaction)))
    published = {}

    def packet(second):
        stamp = (now + timedelta(seconds=second)).isoformat()
        return {**deepcopy(result), "fetched_at": stamp, "source_timestamp": stamp}

    first = packet(0)
    assert publish_snapshots([first], published, now=0) == {123}
    assert publish_snapshots([packet(30)], published, now=30) == set()
    assert len(commits) == 1
    assert publish_snapshots([packet(60)], published, now=60) == {123}
    assert commits[-1][0]["payload"]["fetched_at"] == packet(60)["fetched_at"]
    changed = packet(61)
    changed["state"]["balls"] = 1
    assert publish_snapshots([changed], published, now=61) == {123}
    assert publish_snapshots([packet(59)], published, now=62) == set()  # Out-of-order upstream response.
    corrected = deepcopy(changed)
    corrected["history"][0]["description"] = "Corrected play description"
    fail_commit = True
    with pytest.raises(RuntimeError, match="commit failed"):
        publish_snapshots([corrected], published, now=63)
    fail_commit = False
    assert publish_snapshots([corrected], published, now=64) == {123}
    frozen_source = {**corrected, "fetched_at": packet(125)["fetched_at"], "stale": True}
    assert publish_snapshots([frozen_source], published, now=125) == {123}
    assert commits[-1][0]["source_timestamp"] == corrected["source_timestamp"]
    final = {**packet(130), "abstract_state": "Final", "home_win_probability": 1., "away_win_probability": 0.}
    assert publish_snapshots([final], published, now=130) == {123}
    assert publish_snapshots([final], published, now=240) == set()  # Finals need no heartbeat.


def test_simulator_uses_the_pitcher_intercept(monkeypatch):
    from v2.simulator.posteriors import K_FREE, _assemble

    intercept = np.arange(K_FREE, dtype=float) * 0.1
    pm = _assemble(
        intercept=intercept,
        sigma_batter=np.ones(K_FREE),
        z_batter=np.zeros((3, K_FREE)),
        sigma_platoon=np.ones(K_FREE),
        z_platoon=np.zeros((3, K_FREE)),
        sigma_pitcher=np.ones((2, K_FREE)),
        z_pitcher=np.zeros((2, K_FREE)),
        park_log_real=np.zeros(4),
        batter_ids=np.array([10, 20, 30], dtype=np.int64),
        pitcher_ids=np.array([100, 200], dtype=np.int64),
        venue_codes=np.array(["AAA", "BBB", "CCC", "DDD"]),
    )

    np.testing.assert_allclose(pm.intercept, intercept)
    assert not hasattr(pm, "intercept_diff")

    from collections import Counter

    from v2.simulator.bullpen import BullpenQueue
    from v2.simulator.game_sim import GameInputs, simulate_game
    from v2.simulator.gb_quartiles import GBQuartiles

    seen = Counter()

    def record_pa(pm, batters, pitchers, *args):
        seen.update(int(p) for p in pitchers)
        logits = np.full((len(pitchers), 8), -100.)
        logits[:, 0] = 100.  # deterministic strikeout, isolates pitcher usage
        return logits

    class OutsOnly:
        calls = 0

        def sample(self, rng, state, outs, outcomes, subtype):
            self.calls += 1
            if self.calls == 58:  # Bottom 10th: walkoff HR after nine scoreless innings.
                return np.zeros(len(state), int), 1 + (state == 2).astype(int), np.zeros(len(state), int)
            return np.zeros(len(state), int), np.zeros(len(state), int), np.ones(len(state), int)

    monkeypatch.setattr("v2.simulator.game_sim.pa_logits_batch", record_pa)
    inputs = GameInputs(np.arange(1, 10), np.arange(11, 20),
                        BullpenQueue(593974, [656288, 0], starter_role=1,
                                     workloads={593974: (3,), 656288: (12,)}, roles={656288: 0}),
                        BullpenQueue(200, [0]), "AAA", {}, {})
    for automatic_runner in (True, False):
        seen.clear()
        inputs.automatic_runner = automatic_runner
        h, a = simulate_game(np.random.default_rng(0), pm, OutsOnly(), None, inputs,
                             n_sims=1, form_sigma=0, gbq=GBQuartiles({}, {}))
        assert seen[593974] == 3 and seen[656288] == 12
        assert h.tolist() == [2 if automatic_runner else 1] and a.tolist() == [0]

    unfinished = OutsOnly()
    unfinished.calls = 100  # Never reaches the scripted walkoff.
    with pytest.raises(RuntimeError, match="simulations unfinished"):
        simulate_game(np.random.default_rng(0), pm, unfinished, None, inputs,
                      n_sims=1, form_sigma=0, gbq=GBQuartiles({}, {}))


def test_advancement_conserves_runners_and_uncertainty_separates_mc(tmp_path):
    from v2.data.pa_dataset import OUTCOMES
    from v2.simulator.baserunner import (
        N_OUTCOMES,
        N_SUBTYPES,
        OutSubtypeTable,
        legal_transitions,
        load_advancement_table,
    )
    from v2.simulator.build_advancement_table import build_advancement
    from v2.simulator.uncertainty import win_probability_uncertainty

    # The reproduced failure: bases-loaded singles contaminated a thin empty-base cell.
    rows = pd.DataFrame([
        {"state": 0, "outs": 0, "outcome_idx": OUTCOMES.index("1B"), "subtype_key": "_NA_",
         "new_state": 1, "runs": 0, "outs_added": 0},
        *[{"state": 7, "outs": 0, "outcome_idx": OUTCOMES.index("1B"), "subtype_key": "_NA_",
           "new_state": 3, "runs": 2, "outs_added": 0}] * 100,
    ])
    table = build_advancement(rows)
    assert legal_transitions(table).all()
    table.to_parquet(tmp_path / "advancement.parquet")
    adv = load_advancement_table(tmp_path)
    n = 2000
    new_state, runs, added = adv.sample(np.random.default_rng(0), np.zeros(n, int),
                                       np.zeros(n, int), np.full(n, OUTCOMES.index("1B")), np.zeros(n, int))
    assert (new_state == 1).all() and (runs == 0).all() and (added == 0).all()
    # Exercise every stored transition, not just the most common game states.
    assert not ((table.state == 0) & (table.new_state.map(int.bit_count) + table.runs > 1)).any()

    # Batched sampling must preserve the original draws, including exact CDF ties,
    # duplicate probabilities, single-entry groups, and an empty batch.
    lengths = np.tile([1, 4], 192)
    sub = OutSubtypeTable(np.r_[0, lengths.cumsum()], np.tile([1., 0., .4, .4, 1.], 192),
                          np.tile([1, 1, 2, 3, 4], 192))
    for lookup, bases, values in (
        (adv, (3, N_OUTCOMES, N_SUBTYPES), (adv.new_state, adv.runs, adv.outs_added)),
        (sub, (3, 4, 4), (sub.subtype,)),
    ):
        keys = np.tile(np.flatnonzero(np.diff(lookup.starts)), 4)
        args = np.unravel_index(keys, (8, *bases))
        starts, ends = lookup.starts[keys], lookup.starts[keys + 1]
        reference_rng, batch_rng = np.random.default_rng(7), np.random.default_rng(7)
        draws = reference_rng.random(len(keys))

        def check_sample(rng, draws):
            indices = np.array([min(s + np.searchsorted(lookup.cdf[s:e], u), e - 1)
                                for s, e, u in zip(starts, ends, draws)])
            actual = lookup.sample(rng, *args)
            actual = actual if isinstance(actual, tuple) else (actual,)
            for observed, source in zip(actual, values):
                np.testing.assert_array_equal(observed, source[indices])
                assert observed.dtype == np.int64

        check_sample(batch_rng, draws)
        assert reference_rng.bit_generator.state == batch_rng.bit_generator.state

        class FixedDraws:
            def random(self, size):
                assert size == len(draws)
                return draws

        boundary = lookup.cdf[starts]
        for draws in (boundary, np.nextafter(boundary, 0), np.nextafter(boundary, 1),
                      np.zeros(len(keys)), np.ones(len(keys))):
            check_sample(FixedDraws(), draws)
        empty = lookup.sample(batch_rng, *(np.array([], dtype=int) for _ in range(4)))
        assert all(a.size == 0 for a in (empty if isinstance(empty, tuple) else (empty,)))

    rng = np.random.default_rng(10)
    unresolved = 0
    for _ in range(200):
        p = rng.binomial(333, .5, size=30) / 333
        report = win_probability_uncertainty(p.tolist(), 333)
        unresolved += report["status"] == "below_mc_resolution"
        if report["status"] == "below_mc_resolution":
            assert report["p10"] is None and report["p90"] is None
    assert unresolved >= 180
    signal = win_probability_uncertainty(np.linspace(.2, .8, 30).tolist(), 10000)
    assert signal["status"] == "resolved" and signal["p10"] < .5 < signal["p90"]
    assert signal["parameter_variance_estimate"] < signal["between_draw_variance"]


def test_chronological_acceptance_rejects_hindsight_forecasts():
    from v2.market_model.acceptance import compare_forecasts
    from v2.market_model.residual import validated_forecasts

    rows = []
    for i in range(100):
        day = pd.Timestamp("2026-04-01") + pd.Timedelta(days=i)
        before = (day + pd.Timedelta(hours=15)).tz_localize("UTC").isoformat()
        context = {
            "forecast_at": before, "inputs_as_of": before,
            "training_max_date": "2026-03-31", "tables_training_max_date": "2025-09-28",
            "model_version": "test-model", "raw_home_win_prob": .5, "opener": i < 20,
            "home_run_distribution": {"3": .5, "5": .5},
            "away_run_distribution": {"3": .5, "5": .5},
            "margin_distribution": {"-2": .5, "2": .5},
        }
        rows.append({"game_pk": i, "game_date": day, "home_team": "LAD", "away_team": "SDP",
                     "home_score": 5 if i % 2 else 3, "away_score": 3 if i % 2 else 5,
                     "start_time": (day + pd.Timedelta(hours=19)).tz_localize("UTC"),
                     "home_prediction_at": before, "away_prediction_at": before,
                     "prediction_context": context, "away_prediction_context": context})
    frame = pd.DataFrame(rows)
    report = compare_forecasts(frame, frame)
    assert report["all"]["games"] == 100 and report["opener"]["games"] == 20
    assert report["all_pass"]
    contaminated = frame.copy(deep=True)
    for index, row in contaminated.iterrows():
        context = {**row.prediction_context, "training_max_date": "2026-12-31"}
        contaminated.at[index, "prediction_context"] = context
        contaminated.at[index, "away_prediction_context"] = context
    assert validated_forecasts(contaminated).empty
    with pytest.raises(ValueError, match="verified pregame provenance"):
        compare_forecasts(contaminated, frame)
    late = frame.copy(deep=True)
    late["home_prediction_at"] = late.start_time + pd.Timedelta(seconds=1)
    assert validated_forecasts(late).empty
    missing = frame.iloc[:1].copy()
    context = {**missing.iloc[0].prediction_context, "training_max_date": None}
    missing.at[0, "prediction_context"] = context
    missing.at[0, "away_prediction_context"] = context
    assert validated_forecasts(missing).empty
    extreme = frame.iloc[:1].copy()
    context = {**extreme.iloc[0].prediction_context, "raw_home_win_prob": 0.0}
    extreme.at[0, "prediction_context"] = context
    extreme.at[0, "away_prediction_context"] = context
    assert len(validated_forecasts(extreme)) == 1  # Do not hide overconfident errors.


def test_postseason_probability_mass_and_completed_series():
    """Protect bracket topology, exact series math, and absorbing completed results."""
    from v2.postseason.bracket import forecast, series_outcomes

    teams = [dict(code=f'{league}{seed}', league=league, seed=seed,
                  ws_rank=(seed - 1) * 2 + (1 if league == 'AL' else 2))
             for league in ('AL', 'NL') for seed in range(1, 7)]
    result = forecast(teams, lambda a, b, home, n: [.5] * n)
    by_team = {o['team']: o for o in result['odds']}
    assert len(result['nodes']) == 11
    assert sum(o['champion'] for o in result['odds']) == pytest.approx(1)
    assert by_team['AL1']['champion'] == pytest.approx(.125)
    assert by_team['AL4']['champion'] == pytest.approx(.0625)
    for length in (3, 5, 7):
        outcomes = series_outcomes([.5] * length)
        assert sum(o['probability'] for o in outcomes) == pytest.approx(1)
        assert sum(o['probability'] for o in outcomes if o['wins'][0] > o['wins'][1]) == pytest.approx(.5)
        assert all(max(o['wins']) == length // 2 + 1 for o in outcomes)
    # Two remaining games: 0.8 * 0.7 to come back from 0-1; the played p is ignored.
    outcomes = series_outcomes([.01, .8, .7], (0, 1))
    assert sum(o['probability'] for o in outcomes if o['wins'][0] == 2) == pytest.approx(.56)
    assert series_outcomes([.1] * 3, (2, 1)) == [{'wins': [2, 1], 'probability': 1.}]
    locked = forecast(teams, lambda a, b, home, n: [.5] * n,
                      {'AL-WC45': {'AL4': 0, 'AL5': 2}, 'AL-DS1': {'AL1': 0, 'AL5': 3}})
    odds = {o['team']: o for o in locked['odds']}
    assert odds['AL4']['champion'] == odds['AL1']['champion'] == 0
    assert odds['AL5']['CS'] == 1
    assert odds['AL5']['champion'] == pytest.approx(.25)
    with pytest.raises(ValueError):
        series_outcomes([float('nan')] * 3)
    with pytest.raises(ValueError):
        forecast(teams, lambda a, b, home, n: [.5] * n, {'AL-WC45': {'NL4': 2, 'NL5': 1}})
