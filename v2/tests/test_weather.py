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
