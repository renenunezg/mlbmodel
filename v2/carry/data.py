"""Inputs for the carry study: regular-season Statcast pitches from the local caches, the venue and elevation of every
game (MLB Stats API, cached under cache/carry/), game-time temperature from the weather table, and hitter names."""
from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from v2.carry.physics import FT, air_density

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
STATCAST_DIR = ROOT / "cache"
CACHE_DIR = ROOT / "cache" / "carry"
STATS_API = "https://statsapi.mlb.com/api/v1"
SPORT_IDS = {"mlb": 1, "aaa": 11, "fsl": 14}    # the Florida State League is one of the Single-A (sport 14) leagues

COLUMNS = [
    "game_pk", "game_date", "game_type", "at_bat_number", "pitch_number", "batter", "stand", "p_throws", "description",
    "events", "inning_topbot", "home_team", "away_team", "hc_x", "hc_y", "hit_distance_sc", "launch_speed", "launch_angle",
    "plate_x", "plate_z", "pitch_type", "vx0", "vy0", "vz0", "ax", "ay", "az", "bat_speed", "attack_angle",
    "attack_direction", "swing_path_tilt", "swing_length",
]
SWINGS = {"hit_into_play", "foul", "foul_tip", "swinging_strike", "swinging_strike_blocked", "foul_bunt", "missed_bunt"}
WHIFFS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
NOT_AB = {"walk", "intent_walk", "hit_by_pitch", "sac_fly", "sac_bunt", "catcher_interf", "sac_fly_double_play",
          "sac_bunt_double_play"}
TOTAL_BASES = {"single": 1, "double": 2, "triple": 3, "home_run": 4}


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def load_pitches(season: int, level: str = "mlb") -> pd.DataFrame:
    name = f"statcast_{season}.parquet" if level == "mlb" else f"statcast_{level}_{season}.parquet"
    frame = pd.read_parquet(STATCAST_DIR / name, columns=COLUMNS)
    frame = frame[frame.game_type == "R"].copy()
    for col in ("vx0", "vy0", "vz0", "ax", "ay", "az", "plate_x", "plate_z", "launch_speed", "launch_angle",
                "hit_distance_sc", "bat_speed", "attack_angle", "attack_direction", "swing_path_tilt", "swing_length",
                "hc_x", "hc_y"):
        frame[col] = frame[col].astype(float)
    frame["season"] = season
    frame["level"] = level
    frame["batting_team"] = np.where(frame.inning_topbot == "Bot", frame.home_team, frame.away_team)
    return frame


def venues(season: int, level: str = "mlb") -> pd.DataFrame:
    """game_pk -> venue_id, venue_name, elevation_ft for the level's regular-season games."""
    path = CACHE_DIR / (f"venues_{season}.parquet" if level == "mlb" else f"venues_{level}_{season}.parquet")
    if path.exists():
        return pd.read_parquet(path)
    sched = _get(f"{STATS_API}/schedule?sportId={SPORT_IDS[level]}&season={season}&gameType=R"
                 "&fields=dates,games,gamePk,venue,id,name")
    games = pd.DataFrame([(g["gamePk"], g["venue"]["id"], g["venue"]["name"]) for d in sched["dates"] for g in d["games"]],
                         columns=["game_pk", "venue_id", "venue_name"]).drop_duplicates("game_pk")
    ids = ",".join(str(v) for v in games.venue_id.unique())
    info = _get(f"{STATS_API}/venues?venueIds={ids}&hydrate=location")["venues"]
    elevation = {v["id"]: v.get("location", {}).get("elevation") for v in info}
    # Most minor-league venues carry coordinates but no elevation. Open-Meteo's terrain elevation (90 m grid) matches the
    # published MLB park elevations within 50 ft (Coors exactly), well under a 0.2% change in air density.
    coords = {v["id"]: v.get("location", {}).get("defaultCoordinates") for v in info}
    missing = [i for i, e in elevation.items() if e is None and coords.get(i)]
    if missing:
        lat = ",".join(str(coords[i]["latitude"]) for i in missing)
        lon = ",".join(str(coords[i]["longitude"]) for i in missing)
        meters = _get(f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}")["elevation"]
        elevation.update({i: round(m / FT) for i, m in zip(missing, meters)})
    games["elevation_ft"] = games.venue_id.map(elevation).astype(float)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    games.to_parquet(path)
    return games


def weather() -> pd.DataFrame:
    from backend.db import engine
    return pd.read_sql(text("SELECT game_pk, temp_f FROM weather"), engine)


def hitter_names(ids: list[int]) -> dict[int, str]:
    path = CACHE_DIR / "people.parquet"
    known = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["player_id", "name"])
    missing = sorted(set(ids) - set(known.player_id))
    rows = []
    for start in range(0, len(missing), 150):
        chunk = ",".join(str(i) for i in missing[start:start + 150])
        rows += [(p["id"], p["fullName"]) for p in _get(f"{STATS_API}/people?personIds={chunk}")["people"]]
    if rows:
        known = pd.concat([known, pd.DataFrame(rows, columns=["player_id", "name"])], ignore_index=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        known.to_parquet(path)
    return dict(zip(known.player_id.astype(int), known.name))


def batted_balls(pitches: pd.DataFrame, temps: pd.DataFrame) -> pd.DataFrame:
    """One row per ball in play with distance: spray (+ toward right field), pull (+ to the pull side), venue and air
    density (null where the venue's elevation or the game's temperature is unknown; the weather table covers MLB games
    only), and the pitch velocity at the plate."""
    bip = pitches[pitches.description.eq("hit_into_play") & pitches.launch_speed.notna() & pitches.launch_angle.notna()
                  & pitches.hit_distance_sc.notna() & pitches.hc_x.notna()].copy()
    bip["spray"] = np.degrees(np.arctan2(bip.hc_x - 125.42, 198.27 - bip.hc_y))
    bip["pull"] = np.where(bip.stand == "R", -1, 1) * bip.spray
    bip["month"] = pd.to_datetime(bip.game_date).dt.month
    season, level = int(bip.season.iloc[0]), str(bip.level.iloc[0])
    bip = bip.merge(venues(season, level), on="game_pk", how="left").merge(temps, on="game_pk", how="left")
    bip["air_density"] = air_density(bip.elevation_ft, bip.temp_f)
    # Statcast kinematics are at y = 50 ft; step each pitch to the front of the plate.
    t = (-bip.vy0 - np.sqrt(bip.vy0 ** 2 - 2 * bip.ay * (50 - 17 / 12))) / bip.ay
    for axis in ("x", "y", "z"):
        bip[f"pv{axis}"] = (bip[f"v{axis}0"] + bip[f"a{axis}"] * t) * FT
    return bip.reset_index(drop=True)


def hitter_counts(pitches: pd.DataFrame) -> pd.DataFrame:
    """Per hitter-season: plate-appearance outcomes, swings and whiffs, and the team with the most plate appearances."""
    pa = pitches[pitches.events.notna() & (pitches.events != "truncated_pa")].drop_duplicates(["game_pk", "at_bat_number"])
    ev = pa.events
    pa = pa.assign(pa=1, ab=~ev.isin(NOT_AB), h=ev.isin(TOTAL_BASES), tb=ev.map(TOTAL_BASES).fillna(0), hr=ev.eq("home_run"),
                   bb=ev.isin(["walk", "intent_walk"]), hbp=ev.eq("hit_by_pitch"), sf=ev.isin(["sac_fly", "sac_fly_double_play"]),
                   so=ev.isin(["strikeout", "strikeout_double_play"]))
    keys = ["batter", "season"]
    counts = pa.groupby(keys)[["pa", "ab", "h", "tb", "hr", "bb", "hbp", "sf", "so"]].sum().astype(int)
    swings = pitches[pitches.description.isin(SWINGS)]
    counts["swings"] = swings.groupby(keys).size()
    counts["whiffs"] = swings[swings.description.isin(WHIFFS)].groupby(keys).size()
    counts["team"] = pa.groupby(keys).batting_team.agg(lambda s: s.value_counts().index[0])
    counts["bats"] = pa.groupby(keys).stand.agg(lambda s: "S" if s.nunique() > 1 else s.iloc[0])
    counts[["swings", "whiffs"]] = counts[["swings", "whiffs"]].fillna(0).astype(int)
    return counts.reset_index()
