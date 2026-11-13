"""Context-neutral pitch value: what a pitch was worth on average, whatever the bases and outs.

The contextual value charges a strikeout with the bases loaded more than one with the bases empty, so a
season total carries the order events happened in. Here each pitch is worth the average of its contextual
value over the bases and outs the league sees:

- a pitch not put in play is worth the league mean for its count and call (a 1-2 swinging strike, a 3-1
  ball), which also averages out the steals, wild pitches and pickoffs that land on a pitch;
- a tracked ball in play is worth its expected value in each of the 24 base-out states, weighted by how
  often the league's balls in play come in each.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from v2.pitch_value import batted, expectancy
from v2.pitch_value.batted import N_BASE_OUT


def neutral_values(pitches: pd.DataFrame, table: pd.DataFrame, raw: pd.Series, contact: batted.Contact) -> pd.Series:
    state = pitches["state"].to_numpy()
    count = state // N_BASE_OUT
    hit = batted.tracked(pitches).to_numpy()
    out = raw.groupby([count, pitches["description"]]).transform("mean").to_numpy(copy=True)

    balls = pitches[hit]
    share = np.bincount(state[hit] % N_BASE_OUT, minlength=N_BASE_OUT) / hit.sum()
    lookup = expectancy.lookup(table)
    value = np.zeros(len(balls))
    for base_out in np.flatnonzero(share):
        swapped = count[hit] * N_BASE_OUT + base_out
        value += share[base_out] * (batted.expected_after(balls, swapped, contact) - lookup[swapped])
    out[hit] = value
    return pd.Series(out, index=raw.index, name="value")
