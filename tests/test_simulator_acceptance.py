"""Acceptance gates for the v2 plate-appearance and game simulators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


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
        def sample(self, rng, state, outs, outcomes, subtype):
            return np.zeros(len(state), int), np.zeros(len(state), int), np.ones(len(state), int)

    monkeypatch.setattr("v2.simulator.game_sim.pa_logits_batch", record_pa)
    inputs = GameInputs(np.arange(1, 10), np.arange(11, 20),
                        BullpenQueue(593974, [656288, 0], starter_role=1,
                                     workloads={593974: (3,), 656288: (12,)}, roles={656288: 0}),
                        BullpenQueue(200, [0]), "AAA", {}, {})
    simulate_game(np.random.default_rng(0), pm, OutsOnly(), None, inputs,
                  n_sims=1, form_sigma=0, gbq=GBQuartiles({}, {}))
    assert seen[593974] == 3 and seen[656288] == 12


def test_advancement_conserves_runners_and_uncertainty_separates_mc(tmp_path):
    from v2.data.pa_dataset import OUTCOMES
    from v2.simulator.baserunner import legal_transitions, load_advancement_table
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
