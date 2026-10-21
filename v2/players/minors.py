"""Triple-A evidence for thin MLB records: outcome pseudo-counts and Statcast inputs.

Outcomes: a hitter's Triple-A plate appearances, translated to MLB outcome
rates and season-weighted like the MLB window, shape the prior mean of his
shrunk rates (see ``inputs.prior_rates``). They never count as sample: an
established MLB record is untouched, a rookie's regresses toward his own
minor-league record instead of the position-group mean.

Inputs: the same skill measurements the MLB Statcast cache yields (whiffs,
chase, exit velocity, pitch shapes) are built from the Triple-A cache, shifted
by each input's fitted level offset, and enter the input window as extra
season rows whose sample is capped at the shrink strength's worth of plate
appearances, so they fill a thin MLB record and fade behind a full one.

The scale is chosen with the window and shrink on development log-likelihood;
zero switches both off.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from v2.data.pa_dataset import OUTCOMES
from v2.players import inputs
from v2.players.production import aaa_hitting_rows

log = logging.getLogger(__name__)

LONG_DTYPES = {"player_id": "int64", "season": "int64", "role": "object", "input": "object", "value": "float64", "sample": "float64"}
MIN_OFFSET_SAMPLE = 50.0  # sample at each level for a player-season to inform an input's level offset
PA_INPUT = {"bat": "contact.k_rate", "pit": "batmiss.k_rate"}  # inputs whose sample is plate appearances
# Usage inputs describe a role in that league (Triple-A leverage is relative to Triple-A games), not a skill.
UNTRANSLATED_TRAITS = {"workload"}

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


def empty_inputs() -> pd.DataFrame:
    """A typed empty input frame, so concatenation never widens the MLB frame's dtypes."""
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in LONG_DTYPES.items()})[inputs.LONG_COLUMNS]


def statcast_path(season: int) -> Path:
    return inputs.STATCAST_DIR / f"statcast_aaa_{season}.parquet"


def season_inputs(season: int) -> pd.DataFrame:
    """Triple-A Statcast inputs for one season in the MLB long layout, cached; empty without a Triple-A cache."""
    path = inputs.CACHE_DIR / f"aaa_inputs_{season}.parquet"
    if path.exists():
        cached = pd.read_parquet(path)
        if inputs.statcast_input_names() <= set(cached["input"]):
            return cached
    if not statcast_path(season).exists():
        return empty_inputs()
    frame = inputs.statcast_inputs(inputs.load_pitches(season, "aaa"), season)
    frame = frame.astype({"player_id": "int64", "season": "int64", "value": "float64", "sample": "float64"})
    inputs.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    log.info(f"triple-a inputs[{season}]: {len(frame):,} rows for {frame.player_id.nunique():,} players")
    return frame


def input_offsets(mlb_by_season: dict[int, pd.DataFrame], aaa_by_season: dict[int, pd.DataFrame]) -> dict[str, float]:
    """MLB level minus Triple-A level per input, from players measured at both levels in the same season.

    Each player-season with MIN_OFFSET_SAMPLE at both levels contributes his
    difference, weighted by the smaller sample.
    """
    num: dict[str, float] = {}
    den: dict[str, float] = {}
    for season, aaa in aaa_by_season.items():
        mlb = mlb_by_season.get(season)
        if mlb is None or aaa.empty:
            continue
        both = mlb.merge(aaa[aaa["input"].map(translatable)], on=["player_id", "role", "input"], suffixes=("_mlb", "_aaa"))
        both = both[(both["sample_mlb"] >= MIN_OFFSET_SAMPLE) & (both["sample_aaa"] >= MIN_OFFSET_SAMPLE)]
        w = np.minimum(both["sample_mlb"], both["sample_aaa"])
        diff = (both["value_mlb"] - both["value_aaa"]) * w
        for name, g in diff.groupby(both["input"]):
            num[name] = num.get(name, 0.0) + float(g.sum())
            den[name] = den.get(name, 0.0) + float(w[g.index].sum())
    return {name: num[name] / den[name] for name in num if den[name] > 0}


def translatable(name: str) -> bool:
    spec = inputs.INPUTS.get(name)
    return spec is not None and spec.trait not in UNTRANSLATED_TRAITS


def translated_inputs(aaa: pd.DataFrame, offsets: dict[str, float], strength: float, scale: float, players: pd.Index) -> pd.DataFrame:
    """Triple-A input rows shifted to the MLB level, samples capped at ``strength`` plate appearances' worth and scaled.

    Hitters only, like the outcome prior: Triple-A pitch shapes and whiffs did
    not improve pitcher projections. Only ``players`` (those with MLB rows in
    the window) get rows: Triple-A-only players would otherwise flood the pool
    that z-scores and input availability are measured on.
    """
    if scale <= 0 or aaa.empty or not offsets:
        return empty_inputs()
    frame = aaa[aaa["role"].eq("bat") & aaa["input"].isin(offsets) & aaa["player_id"].isin(players)].copy()
    frame["value"] = frame["value"] + frame["input"].map(offsets)
    # Each input's sample unit (swings, balls in play, pitches) relative to plate appearances.
    if frame.empty:
        return empty_inputs()
    pa = frame[frame["input"].isin(PA_INPUT.values())].groupby("role")["sample"].sum()
    per_input = frame.groupby(["role", "input"])["sample"].sum()
    ratio = per_input / per_input.index.get_level_values("role").map(pa).to_numpy()
    cap = strength * ratio.reindex(pd.MultiIndex.from_frame(frame[["role", "input"]])).to_numpy()
    frame["sample"] = np.minimum(frame["sample"], cap) * scale
    return frame[inputs.LONG_COLUMNS]


def prior_counts(pseudo_by_season: dict[int, pd.DataFrame], weights: dict[int, float], present: set[int], scale: float) -> pd.DataFrame | None:
    """Scaled Triple-A counts over the window, weighted like the MLB window, by player_id; None when off."""
    frames = {s: f for s, f in pseudo_by_season.items() if s in weights and not f.empty}
    if scale <= 0 or not frames:
        return None
    window = inputs.window_counts(frames, weights, present)
    window = window[window["role"].eq("bat")].set_index("player_id")[list(OUTCOMES)]
    return window * scale
