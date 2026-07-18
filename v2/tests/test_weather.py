"""Unit tests for the wind-string parser and dome handling."""
from __future__ import annotations

from backend.data.weather import _parse_weather, parse_wind


def test_parse_wind_direction_signs():
    """Speed + enum + sign across the direction families that matter."""
    assert parse_wind("8 mph, Out To LF") == (8, "OUT_TO_LF", 0.7)
    assert parse_wind("12 mph, Out To CF")[2] > 0
    assert parse_wind("10 mph, In From CF")[2] < 0
    assert parse_wind("9 mph, L To R")[2] == 0.0   # crosswind neutral
    assert parse_wind("3 mph, Varies")[2] == 0.0


def test_parse_wind_none_and_empty():
    assert parse_wind(None) == (0, "NONE", 0.0)
    assert parse_wind("") == (0, "NONE", 0.0)


def test_cf_stronger_than_gap():
    assert parse_wind("10 mph, Out To CF")[2] > parse_wind("10 mph, Out To LF")[2] > 0


def test_dome_zeroes_wind():
    row = _parse_weather({"condition": "Roof Closed", "temp": "72", "wind": "0 mph, None"})
    assert row["is_dome"] and row["wind_speed_mph"] == 0 and row["wind_out_component"] == 0.0


def test_outdoor_parsed_normally():
    row = _parse_weather({"condition": "Partly Cloudy", "temp": "76", "wind": "8 mph, Out To LF"})
    assert not row["is_dome"]
    assert row["temp_f"] == 76 and row["wind_dir_enum"] == "OUT_TO_LF"


def test_bad_temp_is_none():
    assert _parse_weather({"condition": "Clear", "temp": "", "wind": "5 mph, Calm"})["temp_f"] is None


def test_weather_shift_direction():
    """Wind-out + heat lift HR/2B logits; K/BB/OUT untouched; calm = no-op."""
    import numpy as np
    from v2.data.pa_dataset import OUTCOMES
    from v2.simulator.weather_effects import apply_weather_shift

    base = np.zeros((1, len(OUTCOMES)))
    i = {o: k for k, o in enumerate(OUTCOMES)}
    shifted = apply_weather_shift(base.copy(), np.array([15.0]), np.array([20.0]))
    assert shifted[0, i["HR"]] > 0
    assert shifted[0, i["2B"]] > 0
    assert shifted[0, i["K"]] == 0 and shifted[0, i["OUT"]] == 0
    calm = apply_weather_shift(np.zeros((1, len(OUTCOMES))), np.array([0.0]), np.array([0.0]))
    assert not calm.any()


def test_shift_vector_is_single_source_of_truth():
    """weather_shift_vector equals the per-row shift apply_weather_shift adds."""
    import numpy as np
    from v2.data.pa_dataset import OUTCOMES
    from v2.simulator.weather_effects import apply_weather_shift, weather_shift_vector

    vec = weather_shift_vector(15.0, 20.0)
    added = apply_weather_shift(np.zeros((1, len(OUTCOMES))), np.array([15.0]), np.array([20.0]))[0]
    assert np.allclose(vec, added)
    assert not weather_shift_vector(0.0, 0.0).any()


def _ctx(**wx):
    import pandas as pd
    from v2.pipeline.score_games import GameContext
    base = dict(
        game_pk=1, game_date=pd.Timestamp("2026-06-21"), start_time=None,
        home_team="CHC", away_team="LAD",
        home_starter_id=1, away_starter_id=2,
        home_starter_name=None, away_starter_name=None,
        home_starter_throws="R", away_starter_throws="R",
        home_odds=None, away_odds=None,
    )
    base.update(wx)
    return GameContext(**base)


def test_weather_scalars_gating(monkeypatch):
    """Disabled, dome, and missing weather all collapse to (0, 0); live values pass through."""
    import v2.pipeline.score_games as sg

    full = dict(wind_speed_mph=15.0, wind_out_component=1.0, temp_f=90.0, is_dome=False)

    monkeypatch.setattr(sg, "WEATHER_ENABLED", False)
    assert sg.weather_scalars(_ctx(**full)) == (0.0, 0.0)

    monkeypatch.setattr(sg, "WEATHER_ENABLED", True)
    assert sg.weather_scalars(_ctx(**{**full, "is_dome": True})) == (0.0, 0.0)
    assert sg.weather_scalars(_ctx()) == (0.0, 0.0)  # all weather None
    assert sg.weather_scalars(_ctx(**full)) == (15.0, 20.0)


def test_lineups_from_cache_uses_first_nine_distinct_batters():
    import pandas as pd
    from v2.evaluation.weather_ab import _lineups_from_cache

    rows = []
    for side, half, offset in (("away", "Top", 100), ("home", "Bot", 200)):
        for ab in range(1, 11):
            rows.append({
                "game_pk": 1,
                "at_bat_number": ab * 2 + (half == "Bot"),
                "pitch_number": 1,
                "events": "field_out",
                "inning_topbot": half,
                "batter": offset + ab,
            })
    rows.reverse()

    lineups = _lineups_from_cache(pd.DataFrame(rows))

    assert lineups[1]["away"] == list(range(101, 110))
    assert lineups[1]["home"] == list(range(201, 210))


def test_totals_ledger_excludes_scoreless_finals():
    import numpy as np
    import pandas as pd
    from v2.evaluation.weather_ab import _totals_ledger

    games = pd.DataFrame({
        "actual_total": [10.0, np.nan],
        "line": [8.5, 8.5],
        "over_odds": [100.0, 100.0],
        "under_odds": [-110.0, -110.0],
        "p_over_on": [0.6, 0.6],
    })

    with np.errstate(divide="raise", invalid="raise"):
        ledger = _totals_ledger(games, "on")

    assert len(ledger) == 1
    assert ledger.iloc[0]["won"]
