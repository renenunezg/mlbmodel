from datetime import date
from unittest.mock import Mock

import pandas as pd

from backend.data.mlb_api import fetch_probable_starters, fetch_schedule, fetch_schedule_range


def test_schedule_uses_rescheduled_date_and_start(monkeypatch):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "dates": [{
            "date": "2026-06-01",
            "games": [{
                "gamePk": 123456,
                "gameType": "F",
                "officialDate": "2026-06-03",
                "gameDate": "2026-06-01T20:10:00Z",
                "rescheduleDate": "2026-06-03T17:35:00Z",
                "teams": {
                    "home": {"team": {"abbreviation": "BOS"},
                             "probablePitcher": {"id": 1, "fullName": "Fixture Starter"}},
                    "away": {"team": {"abbreviation": "NYY"}},
                },
                "status": {"abstractGameState": "Scheduled"},
                "venue": {"name": "Fenway Park"},
            }],
        }],
    }
    monkeypatch.setattr("backend.data.mlb_api.requests.get", Mock(return_value=response))
    monkeypatch.setattr("backend.data.mlb_api._batch_fetch_handedness", lambda *_: {})

    schedule = fetch_schedule(date(2026, 6, 1))

    assert schedule.loc[0, "game_date"] == "2026-06-03"
    assert schedule.loc[0, "start_time"] == "2026-06-03T17:35:00Z"
    assert schedule.loc[0, "away_team"] == "NYY"
    assert schedule.loc[0, "game_type"] == "F"
    ranged = fetch_schedule_range(date(2026, 6, 1), date(2026, 6, 3))
    pd.testing.assert_frame_equal(schedule, ranged)
    starters = fetch_probable_starters(date(2026, 6, 1))
    assert starters.loc[0, "game_date"] == "2026-06-03"
