"""Per-game weather from the MLB Stats API live feed.

`gameData.weather` carries {condition, temp, wind} where wind is a
park-relative string like "8 mph, Out To LF" - the direction is already
relative to the ballpark, so no lat/lon or park-orientation math is needed.

Indoor games report condition "Roof Closed" / "Dome" with wind "0 mph, None";
those get is_dome=True and a zero wind component.
"""
from __future__ import annotations

import re
import time
from datetime import date

import requests
from sqlalchemy import text

from backend.db import engine

FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"

# signed out-component: + blows runs out, - holds them in. CF dead-center is
# the strongest axis; gap winds (LF/RF) are partial; crosswinds are neutral.
_DIR_COMPONENT = {
    "OUT_TO_CF": 1.0,
    "OUT_TO_LF": 0.7,
    "OUT_TO_RF": 0.7,
    "IN_FROM_CF": -1.0,
    "IN_FROM_LF": -0.7,
    "IN_FROM_RF": -0.7,
    "L_TO_R": 0.0,
    "R_TO_L": 0.0,
    "VARIABLE": 0.0,
    "CALM": 0.0,
    "NONE": 0.0,
}

_FIELD = {"lf": "LF", "left": "LF", "cf": "CF", "center": "CF", "rf": "RF", "right": "RF"}


def parse_wind(wind: str | None) -> tuple[int, str, float]:
    """('8 mph, Out To LF') -> (speed_mph, dir_enum, out_component)."""
    if not wind:
        return 0, "NONE", 0.0
    w = wind.strip().lower()
    m = re.search(r"(\d+)\s*mph", w)
    speed = int(m.group(1)) if m else 0

    dir_part = w.split(",", 1)[1].strip() if "," in w else w
    enum = "NONE"
    if "calm" in dir_part:
        enum = "CALM"
    elif "varies" in dir_part or "variable" in dir_part:
        enum = "VARIABLE"
    elif "l to r" in dir_part:
        enum = "L_TO_R"
    elif "r to l" in dir_part:
        enum = "R_TO_L"
    elif "out to" in dir_part or "in from" in dir_part:
        prefix = "OUT_TO" if "out to" in dir_part else "IN_FROM"
        for token, code in _FIELD.items():
            if token in dir_part:
                enum = f"{prefix}_{code}"
                break
    comp = _DIR_COMPONENT.get(enum, 0.0)
    # crosswind / calm / variable / none carry no run signal regardless of speed
    return speed, enum, comp


def _parse_weather(gd_weather: dict) -> dict:
    condition = (gd_weather.get("condition") or "").strip()
    raw_wind = gd_weather.get("wind")
    temp_s = (gd_weather.get("temp") or "").strip()
    temp_f = int(temp_s) if temp_s.lstrip("-").isdigit() else None

    is_dome = bool(re.search(r"roof|dome", condition, re.I))
    speed, enum, comp = parse_wind(raw_wind)
    if is_dome:
        speed, comp = 0, 0.0
        if enum not in ("NONE", "CALM"):
            enum = "NONE"
    return {
        "wind_speed_mph": speed,
        "wind_dir_raw": raw_wind,
        "wind_dir_enum": enum,
        "wind_out_component": comp,
        "temp_f": temp_f,
        "condition": condition or None,
        "is_dome": is_dome,
    }


_UPSERT = text("""
    INSERT INTO weather (game_pk, wind_speed_mph, wind_dir_raw, wind_dir_enum,
                         wind_out_component, temp_f, condition, is_dome, updated_at)
    VALUES (:game_pk, :wind_speed_mph, :wind_dir_raw, :wind_dir_enum,
            :wind_out_component, :temp_f, :condition, :is_dome, NOW())
    ON CONFLICT (game_pk) DO UPDATE SET
      wind_speed_mph     = EXCLUDED.wind_speed_mph,
      wind_dir_raw       = EXCLUDED.wind_dir_raw,
      wind_dir_enum      = EXCLUDED.wind_dir_enum,
      wind_out_component = EXCLUDED.wind_out_component,
      temp_f             = EXCLUDED.temp_f,
      condition          = EXCLUDED.condition,
      is_dome            = EXCLUDED.is_dome,
      updated_at         = NOW()
""")


def fetch_weather(game_pk: int) -> dict | None:
    """Fetch + upsert one game's weather. Idempotent. Returns the parsed row."""
    try:
        r = requests.get(FEED_URL.format(pk=game_pk), timeout=20)
        r.raise_for_status()
        gd = r.json().get("gameData", {})
        wx = gd.get("weather") or {}
        if not wx:
            return None
        row = _parse_weather(wx)
    except (requests.RequestException, ValueError) as e:
        print(f"  fetch_weather({game_pk}) failed: {e}")
        return None
    with engine.begin() as conn:
        conn.execute(_UPSERT, {"game_pk": int(game_pk), **row})
    return row


def update_weather_for_date(d: date) -> int:
    """Fetch + upsert weather for every game scheduled on date d. Idempotent."""
    import pandas as pd

    games = pd.read_sql(
        text("SELECT game_pk FROM games WHERE game_date = :d"),
        engine, params={"d": d},
    )
    n = sum(fetch_weather(int(gp)) is not None for gp in games["game_pk"])
    print(f"  update_weather_for_date {d}: {n}/{len(games)} games")
    return n


def backfill_weather(start: date, end: date, sleep: float = 0.05) -> int:
    """Populate weather for every game in [start, end]. Idempotent."""
    import pandas as pd

    games = pd.read_sql(
        text("SELECT game_pk FROM games WHERE game_date BETWEEN :s AND :e"),
        engine, params={"s": start, "e": end},
    )
    n = 0
    for gp in games["game_pk"].astype(int):
        if fetch_weather(gp) is not None:
            n += 1
        time.sleep(sleep)
    print(f"  backfill_weather {start}..{end}: {n}/{len(games)} games populated")
    return n
