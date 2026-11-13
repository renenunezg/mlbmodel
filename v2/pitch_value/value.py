"""Pitch value: expectancy after the pitch, minus expectancy before it, plus the runs that scored on it.

Runs are the batting team's, so a negative value is a pitch that helped the pitcher.
"""
from __future__ import annotations

import pandas as pd

from v2.pitch_value import expectancy

# An intentional walk is a row in the feed but no pitch is thrown and the call is the manager's.
NOT_THROWN = ("intent_walk",)


def pitch_values(pitches: pd.DataFrame, table: pd.DataFrame) -> pd.Series:
    runs = expectancy.lookup(table)
    value = runs[pitches["next_state"].to_numpy()] - runs[pitches["state"].to_numpy()] + pitches["runs"].to_numpy()
    return pd.Series(value, index=pitches.index, name="value")


def thrown(pitches: pd.DataFrame) -> pd.Series:
    return ~pitches["events"].isin(NOT_THROWN)
