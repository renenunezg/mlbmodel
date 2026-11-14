"""The pitch-value frame with what the hitter's side needs: location, pitch shape, the hitter's action and its result.

Rows and order are exactly ``v2.pitch_value.pitches.load``; the extra Statcast columns ride along with it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from v2.data.pa_dataset import EVENT_TO_OUTCOME, NON_PA_EVENTS
from v2.pitch_value import pitches as pitch_frame
from v2.pitch_value.value import thrown

EXTRA = ("plate_x", "plate_z", "sz_top", "sz_bot", "release_speed", "pfx_x", "pfx_z", "hc_x", "hc_y",
         "woba_value", "woba_denom")
# Rows with no pitch for the hitter to act on: the automatic balls of an intentional walk and clock violations.
NO_PITCH = ("automatic_ball", "automatic_strike")
# What a pitch became, by the hitter's action. A caught foul tip is a swinging strike in every count.
TAKE_RESULTS = {"ball": "ball", "blocked_ball": "ball", "pitchout": "ball", "called_strike": "called_strike",
                "hit_by_pitch": "hit_by_pitch"}
SWING_RESULTS = {"swinging_strike": "whiff", "swinging_strike_blocked": "whiff", "missed_bunt": "whiff",
                 "foul_tip": "whiff", "bunt_foul_tip": "whiff", "swinging_pitchout": "whiff",
                 "foul": "foul", "foul_bunt": "foul", "foul_pitchout": "foul", "hit_into_play": "in_play"}
PITCH_GROUPS = {
    "FF": "fastball", "SI": "fastball", "FC": "fastball", "FA": "fastball",
    "SL": "breaking", "ST": "breaking", "SV": "breaking", "CU": "breaking", "KC": "breaking", "CS": "breaking",
    "CH": "offspeed", "FS": "offspeed", "FO": "offspeed", "SC": "offspeed",
}
# Savant's attack zones: half-widths in feet, and heights as a share of half the strike zone's height.
ZONE_HALF_WIDTHS = (6.7 / 12, 13.3 / 12, 20.0 / 12)
ZONE_HEIGHTS = (0.67, 1.33, 2.0)
ZONES = ("heart", "shadow", "chase", "waste")
# Directions cut the plate and the hitter's own strike zone in thirds and run on past their edges: inside and away
# are the outer thirds of the plate's 17 inches, up and down the outer thirds of his zone. A pitch can be in one of
# each pair (up and in), so the two families cross each other and neither sums with the attack zones.
PLATE_THIRD = 17.0 / 6.0 / 12.0
HEIGHT_THIRD = 1.0 / 3.0
SIDES = ("inside", "away")
HEIGHTS = ("up", "down")
# Every count is in exactly one situation, so a hitter's situations sum to his total. A full count is a
# two-strike count: the strike that ends the at-bat outweighs the ball that would.
SITUATIONS = ("first_pitch", "ahead", "two_strikes", "behind_even")


def attack_zone(frame: pd.DataFrame) -> pd.Series:
    """heart, shadow, chase or waste; missing where the pitch was not tracked."""
    half = (frame["sz_top"] - frame["sz_bot"]) / 2.0
    height = ((frame["plate_z"] - (frame["sz_top"] + frame["sz_bot"]) / 2.0) / half).abs()
    width = frame["plate_x"].abs()
    index = np.maximum(np.digitize(width.to_numpy(), ZONE_HALF_WIDTHS, right=True),
                       np.digitize(height.to_numpy(), ZONE_HEIGHTS, right=True))
    zone = pd.Series(np.array(ZONES, dtype=object)[index], index=frame.index)
    return zone.where(width.notna() & height.notna())


def side(frame: pd.DataFrame) -> pd.Series:
    """inside or away of the middle third of the plate, from the hitter's side of it; missing in the middle third and
    where the pitch was not tracked."""
    toward = frame["plate_x"] * np.where(frame["stand"].eq("L"), 1.0, -1.0)
    index = np.select([toward > PLATE_THIRD, toward < -PLATE_THIRD], [0, 1], default=2)
    return pd.Series(np.array([*SIDES, None], dtype=object)[index], index=frame.index)


def height(frame: pd.DataFrame) -> pd.Series:
    """up or down of the middle third of the hitter's strike zone; missing in the middle third and where the pitch
    was not tracked."""
    half = (frame["sz_top"] - frame["sz_bot"]) / 2.0
    above = (frame["plate_z"] - (frame["sz_top"] + frame["sz_bot"]) / 2.0) / half
    index = np.select([above > HEIGHT_THIRD, above < -HEIGHT_THIRD], [0, 1], default=2)
    return pd.Series(np.array([*HEIGHTS, None], dtype=object)[index], index=frame.index)


def situation(state: pd.Series) -> pd.Series:
    """first_pitch (0-0), two_strikes, ahead (more balls than strikes, under two strikes) or behind_even (0-1, 1-1)."""
    balls, strikes = pitch_frame.state_parts(state.to_numpy())[:2]
    index = np.select([strikes == 2, balls > strikes, (balls == 0) & (strikes == 0)], [2, 1, 0], default=3)
    return pd.Series(np.array(SITUATIONS, dtype=object)[index], index=state.index)


def load(seasons: list[int]) -> pd.DataFrame:
    frame = pitch_frame.load(seasons, extra=EXTRA)
    description = frame["description"]
    result = description.map(TAKE_RESULTS | SWING_RESULTS)
    frame["seen"] = (thrown(frame) & ~description.isin(NO_PITCH)).to_numpy()
    unknown = sorted(description[frame["seen"] & result.isna()].unique())
    if unknown:
        raise ValueError(f"pitch descriptions with no take or swing result: {unknown}")
    frame["result"] = result
    frame["swing"] = description.isin(SWING_RESULTS).to_numpy()
    frame["bat_team"] = frame["away_team"].where(frame["top"], frame["home_team"])
    frame["outcome"] = frame["events"].where(~frame["events"].isin(NON_PA_EVENTS)).map(EVENT_TO_OUTCOME)
    frame["pitch_group"] = frame["pitch_type"].map(PITCH_GROUPS)
    frame["zone"] = attack_zone(frame)
    frame["side"] = side(frame)
    frame["height"] = height(frame)
    frame["situation"] = situation(frame["state"])
    return frame
