"""How much of a player's gap on contact is his own.

A hitter's balls in play do better or worse than their exit velocity, launch angle and his speed say they should,
and a pitcher's leave the park more or less often than they should. Part of that gap repeats and part is luck.
``earned`` credits the part his own record supports: his gap rate over his seasons to date, earlier seasons
counting for less, shrunk toward zero by ``k`` phantom balls, times the balls of the season being valued.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Shrink:
    k: float       # phantom balls with no gap, added to his weighted balls
    decay: float   # weight of a season relative to the one after it: 0 reads the latest season alone, 1 weighs all alike


def earned(frame: pd.DataFrame, player: str, gap: str, balls: str, shrink: Shrink | None) -> pd.Series:
    """Per row (one player-season of ``frame``, which has a ``season`` column): the shrunk gap rate over that
    player's seasons up to and including the row's, times the row's balls. No ``shrink`` credits nothing."""
    if shrink is None:
        return pd.Series(0.0, index=frame.index)
    rows = frame[[player, "season", gap, balls]].reset_index(drop=True).rename_axis("row").reset_index()
    past = rows.merge(rows, on=player, suffixes=("", "_past"))
    past = past[past["season_past"] <= past["season"]]
    weight = shrink.decay ** (past["season"] - past["season_past"])
    to_date = pd.DataFrame({"gap": weight * past[f"{gap}_past"], "balls": weight * past[f"{balls}_past"]}).groupby(past["row"]).sum()
    return pd.Series((to_date["gap"] / (to_date["balls"] + shrink.k) * rows[balls]).to_numpy(), index=frame.index)


# Pinned from `python -m v2.xwar.validate` on 2026-11-16: development pairs only (first seasons 2021 to 2023).
# Hitters: the gap per tracked ball repeats at +.248 [+.163, +.330] on 596 pairs with 200+ balls in both seasons,
# so the gate passes. The fit gives k 1217.1 balls and a season weight of .322. That leads every season weighted
# alike by +.029 [-.001, +.059] in correlation with the next season's gap, and is not told apart from the latest
# season alone (+.007 [-.011, +.024]): the gap fades (r .25 a season on, .15 two seasons on).
HITTER: Shrink | None = Shrink(k=1217.1, decay=0.322)
# Pitchers: home runs against expected home runs repeat at +.089 [-.014, +.189] on 354 pairs, so the gate fails
# and a pitcher keeps none of his gap.
PITCHER: Shrink | None = None
