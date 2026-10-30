"""Tracked pitches at any level in one frame: shape readings, location and run value per pitch.

MLB (``cache/statcast_{season}.parquet``), Triple-A (``statcast_aaa_``), the Florida State League
(``statcast_fsl_``) and spring training (``statcast_spring_``) share the Savant column layout, so a pitch reads the same at every level.
The Arizona Fall League (``afl``) and the Dominican winter league (``lidom``) have no Savant cache and
are read from the Stats API feed cache instead (``feeds.py``, sport 17), whose fields are the ones
Savant derives its columns from: the feed's induced vertical break is Savant's ``pfx_z`` in inches, its
horizontal break is ``-pfx_x`` in inches, and the release point is reconstructed from the 50-foot state
to within 0.005 ft (checked on an MLB game, 2026-10-30). Feed pitches carry no run value. Shape
readings are in the pitcher's frame: Savant's catcher-view horizontal break and release side are
negative to a right-hander's arm side, so they are flipped for right-handers and arm side reads
positive for everyone; the spin axis is mirrored for left-handers. Each pitch also carries its gap from the pitcher's primary
fastball that season and level (the more thrown of ``FF`` and ``SI``), and its location in the
batter's frame for the command model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.data.game_types import regular_season_pitches
from v2.players.inputs import STATCAST_DIR
from v2.prospects import feeds

FAMILY = {
    **dict.fromkeys(("FF", "SI", "FC"), "fastball"),
    **dict.fromkeys(("SL", "ST", "SV", "CU", "KC", "CS"), "breaking"),
    **dict.fromkeys(("CH", "FS", "FO", "SC"), "offspeed"),
}
FAMILIES = ("fastball", "breaking", "offspeed")
PRIMARY_FASTBALLS = ("FF", "SI")
# Arm angle is left out: Savant lacks it on most 2021-2022 minor-league pitches, and release side and height carry the slot.
SHAPE = ["velo", "ivb", "hb", "spin", "axis", "ext", "rel_x", "rel_z", "velo_diff", "ivb_diff", "hb_diff"]
LOCATION = ["px", "pz", "balls", "strikes", "same_side", "family_code"]
COUNT = ["balls", "strikes", "same_side", "family_code"]
REQUIRED = ("release_speed", "pfx_z", "release_spin_rate")
MIN_TRACKED_SHARE = 0.05  # Triple-A 2022 was tracked in some parks only (37%); a level-season under this share is a broken cache
SOURCE_COLUMNS = [
    "game_pk", "game_date", "game_type", "pitcher", "batter", "stand", "p_throws", "pitch_type", "pitch_number", "at_bat_number",
    "balls", "strikes", "release_speed", "pfx_x", "pfx_z", "release_spin_rate", "spin_axis", "release_extension", "release_pos_x",
    "release_pos_z", "arm_angle", "plate_x", "plate_z", "sz_top", "sz_bot", "delta_run_exp", "launch_speed", "launch_angle",
    "description",
]
BATTED = ["bip", "ls", "la"]
PITCH_COLUMNS = ["pitcher", "season", "level", "game_pk", "pitch_type", "family", "rv", *SHAPE, *LOCATION, *BATTED]


TRACKED_SINCE = {"aaa": 2022, "fsl": 2021, "spring": 2021, "afl": 2024, "lidom": 2024}  # lidom: from December 2024, AFL: from mid-October 2024
LEVEL_FEED = {"aaa": (11, None), "fsl": (14, 123)}  # Savant level -> (sport id, league id) in the prospect feed cache
FEED_LEVELS = {"afl": (17, feeds.AFL_LEAGUE_ID), "lidom": (17, feeds.LIDOM_LEAGUE_ID)}  # levels read from the feed cache itself
FEED_STUFF_COLUMNS = ["bat_side", "pitch_hand", "balls", "strikes", "spin_axis", "release_x", "release_z", "plate_x", "plate_z",
                      "sz_top", "sz_bot", "launch_speed", "launch_angle"]  # written by feeds.py for sport 17 only
CACHE_LAG_DAYS = 5  # Savant may trail the schedule's last games by a few days


def path(season: int, level: str):
    if level in FEED_LEVELS:
        raise ValueError(f"{level} pitches live in the feed cache, not a Savant file")
    return STATCAST_DIR / (f"statcast_{season}.parquet" if level == "mlb" else f"statcast_{level}_{season}.parquet")


def available(season: int, level: str) -> bool:
    """Whether a level-season has pitches to load: a Savant cache file, or feed games of the league with tracked pitches."""
    if level in FEED_LEVELS:
        return not _feed_games(season, level).empty
    return path(season, level).exists()


def _feed_games(season: int, level: str) -> pd.DataFrame:
    sport_id, league_id = FEED_LEVELS[level]
    games = feeds.read_kind(sport_id, season, "games")
    if games.empty:
        return games
    return games[games["league_id"].eq(league_id) & (games["tracked_share"] >= feeds.MIN_TRACKED_SHARE)]


def _feed_frame(season: int, level: str) -> pd.DataFrame:
    """A feed level-season's pitch rows in the Savant column layout (``SOURCE_COLUMNS``); run value is null."""
    games = _feed_games(season, level)
    if games.empty:
        raise FileNotFoundError(f"No tracked {level} games cached for {season}; run python -m v2.prospects.feeds --years {season} --levels 17")
    raw = feeds.read_kind(FEED_LEVELS[level][0], season, "pitches")
    raw = raw[raw["game_pk"].isin(games["game_pk"])]
    missing = [c for c in FEED_STUFF_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{level} {season} pitch rows lack {missing}; the cache predates the stuff columns, rebuild cache/prospects/feeds/17/{season}")
    out = pd.DataFrame({
        "game_pk": raw["game_pk"], "game_date": raw["game_pk"].map(games.set_index("game_pk")["date"]), "game_type": "R",
        "pitcher": raw["pitcher"], "batter": np.nan, "stand": raw["bat_side"], "p_throws": raw["pitch_hand"], "pitch_type": raw["pitch_type"],
        "pitch_number": np.nan, "at_bat_number": np.nan, "balls": raw["balls"], "strikes": raw["strikes"],
        "release_speed": raw["start_speed"], "pfx_x": -raw["hb"] / 12.0, "pfx_z": raw["ivb"] / 12.0, "release_spin_rate": raw["spin_rate"],
        "spin_axis": raw["spin_axis"], "release_extension": raw["extension"], "release_pos_x": raw["release_x"], "release_pos_z": raw["release_z"],
        "arm_angle": np.nan, "plate_x": raw["plate_x"], "plate_z": raw["plate_z"], "sz_top": raw["sz_top"], "sz_bot": raw["sz_bot"],
        "delta_run_exp": np.nan, "launch_speed": raw["launch_speed"], "launch_angle": raw["launch_angle"],
        "description": np.where(raw["code"].isin(feeds.IN_PLAY_CODES), "hit_into_play", "other"),
    })
    return out[SOURCE_COLUMNS]


def _savant_frame(season: int, level: str) -> pd.DataFrame:
    src = path(season, level)
    if not src.exists():
        raise FileNotFoundError(f"Missing {src}")
    raw = pd.read_parquet(src, columns=SOURCE_COLUMNS)
    if level == "spring":
        return raw[raw["game_type"].eq("S") & pd.to_datetime(raw["game_date"]).dt.year.eq(season)]
    return regular_season_pitches(raw, season)


def load(season: int, level: str) -> pd.DataFrame:
    """Graded-type regular-season pitches of one level and season, normalized; excluded types are dropped."""
    raw = _feed_frame(season, level) if level in FEED_LEVELS else _savant_frame(season, level)
    tracked = raw[list(REQUIRED)].notna().all(axis=1)
    if tracked.mean() < MIN_TRACKED_SHARE:
        raise ValueError(f"{level} {season}: only {tracked.mean():.1%} of pitches carry the required readings")
    p = raw[raw["pitch_type"].isin(FAMILY) & tracked].copy()  # pitches from untracked games carry no shape
    for c in SOURCE_COLUMNS[9:-1]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    lefty = p["p_throws"].eq("L")
    sign = np.where(lefty, 1.0, -1.0)  # arm side positive
    out = pd.DataFrame({
        "pitcher": p["pitcher"].astype("int64"), "season": season, "level": level, "game_pk": p["game_pk"].astype("int64"),
        "pitch_type": p["pitch_type"], "family": p["pitch_type"].map(FAMILY), "rv": p["delta_run_exp"],
        "velo": p["release_speed"], "ivb": p["pfx_z"] * 12.0, "hb": p["pfx_x"] * 12.0 * sign, "spin": p["release_spin_rate"],
        "axis": np.where(lefty, 360.0 - p["spin_axis"], p["spin_axis"]), "ext": p["release_extension"],
        "rel_x": p["release_pos_x"] * sign, "rel_z": p["release_pos_z"], "arm_angle": p["arm_angle"],
        "px": p["plate_x"] * np.where(p["stand"].eq("R"), 1.0, -1.0),
        "pz": (p["plate_z"] - p["sz_bot"]) / (p["sz_top"] - p["sz_bot"]),
        "balls": p["balls"], "strikes": p["strikes"], "same_side": (p["stand"] == p["p_throws"]).astype(float),
        "family_code": p["pitch_type"].map(FAMILY).map({f: i for i, f in enumerate(FAMILIES)}).astype(float),
        "bip": p["description"].eq("hit_into_play"), "ls": p["launch_speed"], "la": p["launch_angle"],
    }, index=p.index)
    primary = out[out["pitch_type"].isin(PRIMARY_FASTBALLS)].groupby(["pitcher", "pitch_type"]).size().reset_index(name="n")
    primary = primary.sort_values("n").groupby("pitcher").tail(1).set_index("pitcher")["pitch_type"]
    fb = out[out["pitch_type"].eq(out["pitcher"].map(primary))].groupby("pitcher")[["velo", "ivb", "hb"]].mean()
    for c in ("velo", "ivb", "hb"):
        out[f"{c}_diff"] = out[c] - out["pitcher"].map(fb[c])
    return out[PITCH_COLUMNS].reset_index(drop=True)


def require_complete(season: int, level: str) -> None:
    """Fail fast when a tracked minor-league season's Savant cache is missing or stops well before the level's last game.

    The scoring job builds the cache a time-boxed slice per night, so a half-built season must stop the
    board rather than publish stuff (and grades) from part of a season.
    """
    if season < TRACKED_SINCE[level] or level in FEED_LEVELS:  # feed levels are their own schedule; a partial season is still a valid read
        return
    src = path(season, level)
    if not src.exists():
        raise FileNotFoundError(f"Missing {src}; run python -m v2.data.build_aaa_cache --level {level} --years {season}")
    if level not in LEVEL_FEED:  # spring training has no prospect feed to compare against
        return
    sport_id, league_id = LEVEL_FEED[level]
    games = feeds.read_kind(sport_id, season, "games")
    if league_id is not None:
        games = games[games["league_id"].eq(league_id)]
    if games.empty:
        return
    last_game = pd.to_datetime(games["date"]).max()
    last_pitch = pd.to_datetime(pd.read_parquet(src, columns=["game_date"])["game_date"]).max()
    if last_pitch < last_game - pd.Timedelta(days=CACHE_LAG_DAYS):
        raise ValueError(f"{src.name} ends {last_pitch.date()} but {level} played through {last_game.date()}; the cache is still building")
