from datetime import date
from unittest.mock import Mock

import pandas as pd

from backend.data.mlb_api import fetch_schedule
from backend.evaluate_model import (
    _evaluation_update_values,
    _merge_predictions_with_results,
)
from pipeline import _batch_upsert_games


class RecordingConnection:
    def __init__(self):
        self.statements = []

    def execute(self, statement, params):
        self.statements.append((str(statement), params))


def test_schedule_upsert_moves_rescheduled_game():
    conn = RecordingConnection()
    schedule = pd.DataFrame([{
        "game_pk": 123456,
        "game_date": date(2026, 6, 3),
        "start_time": "2026-06-03T17:35:00Z",
        "home_team": "BOS",
        "away_team": "NYY",
        "home_score": None,
        "away_score": None,
        "status": "Scheduled",
        "venue": "Fenway Park",
    }])

    _batch_upsert_games(conn, schedule)

    sql, params = conn.statements[0]
    assert "game_date = EXCLUDED.game_date" in sql
    assert "start_time = EXCLUDED.start_time" in sql
    assert params["game_date"] == "2026-06-03"
    assert params["start_time"] == "2026-06-03T17:35:00Z"


def test_schedule_uses_rescheduled_date_and_start(monkeypatch):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "dates": [{
            "date": "2026-06-01",
            "games": [{
                "gamePk": 123456,
                "officialDate": "2026-06-03",
                "gameDate": "2026-06-01T20:10:00Z",
                "rescheduleDate": "2026-06-03T17:35:00Z",
                "teams": {
                    "home": {"team": {"abbreviation": "BOS"}},
                    "away": {"team": {"abbreviation": "NYY"}},
                },
                "status": {"abstractGameState": "Scheduled"},
                "venue": {"name": "Fenway Park"},
            }],
        }],
    }
    monkeypatch.setattr("backend.data.mlb_api.requests.get", Mock(return_value=response))

    schedule = fetch_schedule(date(2026, 6, 1))

    assert schedule.loc[0, "game_date"] == "2026-06-03"
    assert schedule.loc[0, "start_time"] == "2026-06-03T17:35:00Z"
    assert schedule.loc[0, "away_team"] == "NYY"


def test_evaluation_rejects_prediction_from_original_date():
    predictions = pd.DataFrame([
        {"game_pk": 123456, "date": "2026-06-01", "team": "NYY", "win_prob": 0.6},
        {"game_pk": 123456, "date": "2026-06-03", "team": "NYY", "win_prob": 0.4},
    ])
    results = pd.DataFrame([{
        "game_pk": 123456,
        "game_date": "2026-06-03",
        "team": "NYY",
        "actual_runs": 0,
        "winning_team": "BOS",
        "actual_margin": -10,
    }])

    merged = _merge_predictions_with_results(predictions, results)

    assert pd.to_datetime(merged["date"]).dt.date.tolist() == [date(2026, 6, 3)]


def test_evaluation_clears_accuracy_when_bet_count_returns_to_zero():
    updates = _evaluation_update_values({
        "date": date(2026, 6, 3),
        "eval_window": "day",
        "ml_predictions": 0,
        "ml_accuracy": None,
    })

    assert updates == {"ml_predictions": 0, "ml_accuracy": None}
