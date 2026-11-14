"""A hitter's value on a pitch, split into the parts he controls.

- decision: the value of what he chose (swing or take) minus the value of the league's mix at that pitch;
- contact: on a swing, the value of what he made of it (a whiff, a foul, or a ball in play at the league's
  quality for that pitch) minus the value of an average swing at it;
- quality: on a ball in play, its expected value from how it was hit minus the league's ball in play from
  that pitch;
- call: on a take, what the pitch was called (or a hit batter) against what the league model expected.
  The umpire's, not the hitter's, so it is reported apart and not credited.

The four sum, pitch by pitch, to the context-neutral value minus the pitch's league-average value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COMPONENTS = ("decision", "contact", "quality")
SUM_TOLERANCE = 1e-9


def split(pitches: pd.DataFrame, neutral: pd.Series, league: pd.DataFrame) -> pd.DataFrame:
    """decision, contact, quality and call for every pitch a hitter saw; zero on every other row."""
    seen = pitches["seen"].to_numpy()
    swing = seen & pitches["swing"].to_numpy()
    take = seen & ~swing
    in_play = swing & pitches["result"].eq("in_play").to_numpy()
    value = neutral.to_numpy()
    made = np.where(in_play, league["in_play"], value)
    out = pd.DataFrame({
        "decision": np.where(swing, league["swing"], league["take"]) - league["league"],
        "contact": np.where(swing, made - league["swing"], 0.0),
        "quality": np.where(in_play, value - league["in_play"], 0.0),
        "call": np.where(take, value - league["take"], 0.0),
    }, index=pitches.index)
    out[~seen] = 0.0
    gap = np.abs(out.sum(axis=1).to_numpy() - np.where(seen, value - league["league"], 0.0)).max()
    if not gap <= SUM_TOLERANCE:
        raise ValueError(f"components miss the neutral value minus the league value by up to {gap:.3g} runs")
    return out
