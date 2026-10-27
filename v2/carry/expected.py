"""League-expected distance for a ball in the air, and carry = actual minus expected.

Two expectations, each cross-fitted by game (a ball never informs its own prediction):
  raw: exit velocity, launch angle and season, the comparison the public usually quotes;
  full: adds spray, handedness, ballpark-season and month, so a hitter's park and summer air do not count as his carry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

AIR_LAUNCH = (10.0, 50.0)
AIR_MIN_EXIT = 90.0
RAW = ["launch_speed", "launch_angle", "season"]
FULL = RAW + ["pull", "right_handed", "park", "month"]


def air_balls(bip: pd.DataFrame) -> pd.DataFrame:
    air = bip[bip.launch_angle.between(*AIR_LAUNCH) & (bip.launch_speed >= AIR_MIN_EXIT)].copy()
    air["right_handed"] = (air.stand == "R").astype(int)
    park = air.venue_id.astype("Int64").astype(str) + "-" + air.season.astype(str)
    air["park"] = park.astype("category").cat.codes
    return air.reset_index(drop=True)


def _cross_fit(frame: pd.DataFrame, features: list[str], categorical: list[str]) -> np.ndarray:
    fold = (frame.game_pk % 2).to_numpy()
    out = np.empty(len(frame))
    mask = [f in categorical for f in features]
    for k in (0, 1):
        model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, max_leaf_nodes=63, categorical_features=mask,
                                              random_state=0)
        model.fit(frame.loc[fold != k, features], frame.loc[fold != k, "hit_distance_sc"])
        out[fold == k] = model.predict(frame.loc[fold == k, features])
    return out


def add_carry(air: pd.DataFrame) -> pd.DataFrame:
    if air.park.nunique() > 255:
        raise ValueError(f"{air.park.nunique()} ballpark-seasons exceed the 255 categories the model supports")
    air = air.copy()
    air["carry_raw"] = air.hit_distance_sc - _cross_fit(air, RAW, [])
    air["carry"] = air.hit_distance_sc - _cross_fit(air, FULL, ["park"])
    return air
