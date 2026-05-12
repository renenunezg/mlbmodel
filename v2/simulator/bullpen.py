"""Bullpen state + pitcher-swap logic for game simulation.

Backtest queue: the list of pitchers who actually appeared in relief in that game,
in order of first PA, sourced from the statcast cache. Swap rules decide WHEN to
swap; the queue decides WHO.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"

# Rest-eligibility rules. Mirror v1's reliever-availability heuristic.
ELIG_OUTS_1D = 6
ELIG_OUTS_2D = 9

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
        # ran out of relievers, recycle the last reliever (rare in real games)
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


@dataclass
class LiveQueueContext:
    """Inputs for build_queues_live per (game_pk, side)."""
    game_pk: int
    side: str            # "home" | "away"
    team: str            # canonical 3-letter team code
    starter_id: int


def _load_workload(game_date: date, teams: list[str], engine) -> pd.DataFrame:
    sql = text("""
        SELECT game_date, pitcher_id, team, outs, role
        FROM pitcher_workload
        WHERE team = ANY(:teams)
          AND game_date >= :lo AND game_date < :hi
    """)
    lo = game_date - timedelta(days=2)
    hi = game_date
    with engine.begin() as conn:
        return pd.read_sql(sql, conn, params={"teams": teams, "lo": lo, "hi": hi})


def build_queues_live(
    game_date: date,
    contexts: list[LiveQueueContext],
    engine=None,
) -> dict[tuple[int, str], BullpenQueue]:
    """Per-side rest-aware queue for live (pre-game) scoring.

    For each context, pulls the team's active 26-man pitcher roster, removes the
    probable starter, drops relievers exceeding rest thresholds in the prior
    1-2 days, and orders the remainder fewest-outs-first over the prior 2 days.

    Returns a dict keyed by (game_pk, side). Sides where the active roster
    couldn't be fetched are omitted; caller should fall back to cache or stub.
    """
    if engine is None:
        from backend.db import engine as default_engine
        engine = default_engine

    from backend.data.mlb_api import fetch_active_pitchers
    from backend.team_mappings import TEAM_ID_BY_CODE

    teams = list({c.team for c in contexts})
    wl = _load_workload(game_date, teams, engine)
    wl["game_date"] = pd.to_datetime(wl["game_date"]).dt.date
    yesterday = game_date - timedelta(days=1)

    out: dict[tuple[int, str], BullpenQueue] = {}
    for ctx in contexts:
        team_id = TEAM_ID_BY_CODE.get(ctx.team)
        if team_id is None:
            continue
        roster = fetch_active_pitchers(team_id)
        if not roster:
            continue

        team_wl = wl[wl["team"] == ctx.team]
        outs_1d = team_wl[team_wl["game_date"] == yesterday].set_index("pitcher_id")["outs"].to_dict()
        outs_2d = team_wl.groupby("pitcher_id")["outs"].sum().to_dict()

        candidates = []
        for pid in roster:
            if pid == ctx.starter_id:
                continue
            o1 = int(outs_1d.get(pid, 0))
            o2 = int(outs_2d.get(pid, 0))
            if o1 >= ELIG_OUTS_1D or o2 >= ELIG_OUTS_2D:
                continue
            candidates.append((pid, o2))

        candidates.sort(key=lambda t: t[1])
        relievers = [pid for pid, _ in candidates]
        out[(ctx.game_pk, ctx.side)] = BullpenQueue(
            starter=ctx.starter_id, relievers=relievers
        )

    return out
