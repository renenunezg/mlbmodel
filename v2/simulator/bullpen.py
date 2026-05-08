"""Bullpen state + pitcher-swap logic for game simulation.

Phase 4 design: for backtests, the per-game reliever queue is the list of pitchers
who actually appeared in relief in that game from the statcast cache, in order of
first PA. The swap rules (outs/runs/PA-count thresholds) decide WHEN to swap; the
queue decides WHO. This avoids needing to reconstruct rest state from `bullpen_daily`
(which only stores per-team aggregates, not per-pitcher).

For live use (Phase 7), this should be replaced with a rest-aware roster fetch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"

# Starter pull thresholds. Match CLAUDE.md §"Bullpen rules".
PULL_OUTS_HARD = 18         # 6 IP completed → automatic pull
PULL_OUTS_RUNS = 12         # 4 IP + 4+ runs → pull
PULL_RUNS_HARD = 6          # 6+ runs → pull regardless of IP
PULL_PA_PROXY = 24          # ~95 pitches @ ~3.95 pitches/PA


def should_pull_starter(outs: int, runs_allowed: int, pa_count: int) -> bool:
    if outs >= PULL_OUTS_HARD:
        return True
    if runs_allowed >= PULL_RUNS_HARD:
        return True
    if outs >= PULL_OUTS_RUNS and runs_allowed >= 4:
        return True
    if pa_count >= PULL_PA_PROXY:
        return True
    return False


@dataclass
class BullpenQueue:
    """Per-side queue of pitcher_ids: starter first, then relievers in order."""
    starter: int
    relievers: list[int]
    pulled_idx: int = 0  # 0 = starter still in; 1 = first reliever in; ...

    def current(self) -> int:
        if self.pulled_idx == 0:
            return self.starter
        ridx = self.pulled_idx - 1
        if ridx < len(self.relievers):
            return self.relievers[ridx]
        # ran out of relievers — recycle the last reliever (rare in real games)
        return self.relievers[-1] if self.relievers else self.starter

    def advance(self) -> int:
        self.pulled_idx += 1
        return self.current()


def build_queues_from_cache(year: int) -> dict[tuple[int, str], BullpenQueue]:
    """For every (game_pk, side) in the year, read pitcher order of appearance.

    side ∈ {"home", "away"}: "home" = home team pitching = top-half PAs.
    Returns dict keyed by (game_pk, side) → BullpenQueue.
    """
    path = CACHE_DIR / f"statcast_{year}.parquet"
    df = pd.read_parquet(path, columns=[
        "game_pk", "pitcher", "inning", "inning_topbot", "at_bat_number", "pitch_number",
        "events",
    ])
    # one row per PA via the terminating pitch
    df = df[df["events"].notna()]
    df = df.sort_values(["game_pk", "inning", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    # side = team currently pitching = opposite of batting team
    df["side"] = np.where(df["inning_topbot"] == "Top", "home", "away")

    # first appearance per (game, side, pitcher) is the relevant signal for ordering
    first = (
        df.groupby(["game_pk", "side", "pitcher"], sort=False)
        .agg(first_inning=("inning", "min"), first_ab=("at_bat_number", "min"))
        .reset_index()
        .sort_values(["game_pk", "side", "first_inning", "first_ab"])
    )
    queues: dict[tuple[int, str], BullpenQueue] = {}
    for (gp, side), grp in first.groupby(["game_pk", "side"], sort=False):
        ids = grp["pitcher"].astype(np.int64).tolist()
        queues[(int(gp), side)] = BullpenQueue(starter=ids[0], relievers=ids[1:])
    return queues
