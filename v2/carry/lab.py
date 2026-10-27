"""The swing lab's carry grids: rerun each hitter's own balls through the collision with the swing shifted, holding every
launch fixed, and convert the change in spin effect to feet of carry with the study's calibration slope."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from v2.carry.swing import solve

log = logging.getLogger(__name__)

STEPS = list(range(-6, 7, 2))    # degrees; the grid is tilt change (outer) by attack-angle change (inner)
MIN_BALLS = 25


def grids(balls: pd.DataFrame, base: pd.DataFrame, typical: pd.Series, slope: float) -> tuple[dict[int, list[int]], list[float]]:
    """Per hitter with MIN_BALLS spin balls: carry change in tenths of a foot, 49 values. Also the generic hitter: the
    same change averaged over every ball in the league."""
    hitters = balls.batter.value_counts()
    keep = balls.batter.isin(hitters[hitters >= MIN_BALLS].index)
    tilt_rel = balls.tilt_rel.to_numpy()
    per_hitter: dict[int, list[int]] = {}
    generic: list[float] = []
    for d_tilt in STEPS:
        for d_attack in STEPS:
            if d_tilt == 0 and d_attack == 0:
                change = pd.Series(0.0, index=balls.index)
            else:
                shifted, _ = solve(balls, tilt_rel, d_tilt, d_attack, typical)
                change = slope * (shifted.spin_effect - base.spin_effect)
            generic.append(round(float(change.mean()), 2))
            for pid, val in change[keep].groupby(balls.batter[keep]).mean().items():
                per_hitter.setdefault(int(pid), []).append(int(round(val * 10)))
        log.info(f"lab grid: tilt change {d_tilt:+d} done")
    return per_hitter, generic
