"""The pitch values of ``v2.pitch_value`` read from the hitter's side, and the park and season taken out of them.

Runs are the batting team's, so positive is good for the hitter. Nothing here adjusts for the pitcher: the
pitch difficulty model (``difficulty.py``) does that for the components.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from v2.pitch_value import batted, context, expectancy, neutral, value

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Values:
    table: pd.DataFrame        # the 288-state run expectancy
    contact: batted.Contact    # the league contact model
    frame: pd.DataFrame        # raw, expected, neutral: one row per pitch


def build(pitches: pd.DataFrame) -> Values:
    table = expectancy.fit(pitches)
    raw = value.pitch_values(pitches, table)
    contact = batted.fit(pitches, table)
    return Values(table=table, contact=contact, frame=pd.DataFrame({
        "raw": raw,
        "expected": batted.expected_values(pitches, table, raw, contact),
        "neutral": neutral.neutral_values(pitches, table, raw, contact),
    }))


def environment(pitches: pd.DataFrame) -> pd.Series:
    """Each season's runs per pitch relative to all seasons."""
    return pitches.groupby("season")["runs"].mean() / pitches["runs"].mean()


def _park_effect(pitches: pd.DataFrame, series: pd.Series) -> np.ndarray:
    """Runs per pitch the park adds to ``series``; zero when the parks cannot be told apart from noise."""
    try:
        parks = context.park_effects(pitches, series)
    except ValueError as error:
        log.info(f"{series.name}: no park effect removed ({error})")
        return np.zeros(len(series))
    return context.park(pitches).map(parks.set_index("park")["effect"]).to_numpy()


def adjusted(pitches: pd.DataFrame, series: pd.Series, where: pd.Series, scale: pd.Series) -> pd.Series:
    """``series`` over the pitches in ``where`` with the park removed, each season centred on the league
    average and put on a common run scale; zero on every other pitch."""
    sub, part = pitches[where], series[where]
    part = part - _park_effect(sub, part)
    part = (part - part.groupby(sub["season"]).transform("mean")) / sub["season"].map(scale)
    return part.reindex(series.index, fill_value=0.0).rename(series.name)
