"""Keep prediction coverage separate from the regular-season training population."""
from __future__ import annotations

import pandas as pd

POSTSEASON_GAME_TYPES = ("F", "D", "L", "W")
PREDICTION_GAME_TYPES = ("R", *POSTSEASON_GAME_TYPES)


def regular_season_pitches(frame: pd.DataFrame, year: int | None = None) -> pd.DataFrame:
    """Reject unclassified caches; select regular-season pitches before any fitting."""
    if frame.empty:
        return frame.copy()
    if "game_type" not in frame:
        raise ValueError("Statcast data lacks game_type; rebuild the cache before training")
    keep = frame["game_type"].eq("R")
    if year is not None:
        keep &= pd.to_datetime(frame["game_date"]).dt.year.eq(year)
    return frame.loc[keep].copy()
