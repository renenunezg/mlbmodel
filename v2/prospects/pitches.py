"""Savant pitches at any tracked level in one frame: shape readings, location and run value per pitch.

MLB (``cache/statcast_{season}.parquet``), Triple-A (``statcast_aaa_``) and the Florida State League
(``statcast_fsl_``) share the Savant column layout, so a pitch reads the same at every level. Shape
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


def path(season: int, level: str):
    return STATCAST_DIR / (f"statcast_{season}.parquet" if level == "mlb" else f"statcast_{level}_{season}.parquet")


def load(season: int, level: str) -> pd.DataFrame:
    """Graded-type regular-season pitches of one level and season, normalized; excluded types are dropped."""
    src = path(season, level)
    if not src.exists():
        raise FileNotFoundError(f"Missing {src}")
    raw = regular_season_pitches(pd.read_parquet(src, columns=SOURCE_COLUMNS), season)
    tracked = raw[list(REQUIRED)].notna().all(axis=1)
    if tracked.mean() < MIN_TRACKED_SHARE:
        raise ValueError(f"{src.name}: only {tracked.mean():.1%} of pitches carry the required readings")
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
