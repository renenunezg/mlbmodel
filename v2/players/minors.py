"""Triple-A pseudo-counts: the prior a thin MLB record regresses toward.

A hitter's Triple-A plate appearances, translated to MLB outcome rates and
season-weighted like the MLB window, shape the prior mean of his shrunk rates
(see ``inputs.prior_rates``). They never count as sample: an established MLB
record is untouched, a rookie's regresses toward his own minor-league record
instead of the position-group mean. The scale is chosen with the window and
shrink on development log-likelihood; zero switches the prior off.
"""
from __future__ import annotations

import logging

import pandas as pd

from v2.data.pa_dataset import OUTCOMES
from v2.players import inputs
from v2.players.production import aaa_hitting_rows

log = logging.getLogger(__name__)

# MLB rate over Triple-A rate per outcome, same season, hitters with 100 PA at
# both levels, 2021-2025 (485 player-seasons). Outs absorb the remainder.
TRANSLATION = {"K": 1.21, "BB": 0.68, "HBP": 0.82, "1B": 0.90, "2B": 0.82, "3B": 0.67, "HR": 0.66}
SCALE_CANDIDATES = (0.0, 0.25, 0.5, 1.0)


def pseudo_counts(season: int) -> pd.DataFrame:
    """Translated Triple-A outcome counts for one season at scale one, in the ``outcome_counts`` layout."""
    try:
        r = aaa_hitting_rows(season)
    except (FileNotFoundError, ValueError) as exc:
        log.warning(f"triple-a {season}: {exc}")
        return pd.DataFrame(columns=["player_id", "role", *OUTCOMES, "season"])
    raw = pd.DataFrame({"K": r["strikeOuts"], "BB": r["baseOnBalls"], "HBP": r["hitByPitch"], "HR": r["homeRuns"], "3B": r["triples"],
                        "2B": r["doubles"], "1B": r["hits"] - r["doubles"] - r["triples"] - r["homeRuns"]}).astype(float)
    out = pd.DataFrame({k: raw[k] * TRANSLATION[k] for k in TRANSLATION})
    out["OUT"] = (r["plateAppearances"].astype(float) - out.sum(axis=1)).clip(lower=0.0)
    out.insert(0, "player_id", r["player_id"].astype("int64"))
    out.insert(1, "role", "bat")
    out["season"] = season
    return out[["player_id", "role", *OUTCOMES, "season"]]


def prior_counts(pseudo_by_season: dict[int, pd.DataFrame], weights: dict[int, float], present: set[int], scale: float) -> pd.DataFrame | None:
    """Scaled Triple-A counts over the window, weighted like the MLB window, by player_id; None when off."""
    frames = {s: f for s, f in pseudo_by_season.items() if s in weights and not f.empty}
    if scale <= 0 or not frames:
        return None
    window = inputs.window_counts(frames, weights, present)
    window = window[window["role"].eq("bat")].set_index("player_id")[list(OUTCOMES)]
    return window * scale
