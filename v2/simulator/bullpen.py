"""Pregame pitching roles, workload distributions, and explicit opener/bulk plans."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text

# Rest eligibility: outs thrown over the last one and two days.
ELIG_OUTS_1D = 6
ELIG_OUTS_2D = 9

# Starter pull thresholds.
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
    starter_role: int = 0
    workloads: dict[int, tuple[int, ...]] = field(default_factory=dict)
    roles: dict[int, int] = field(default_factory=dict)
    usage_known: bool = False
    team_outs_2d: int | None = None

    def outs_samples(self, pitcher_id: int, slot: int) -> tuple[int, ...]:
        # An unspecified plan must never give an opener six innings.
        return self.workloads.get(pitcher_id, (3,))


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
    lo = game_date - timedelta(days=60)
    hi = game_date
    with engine.begin() as conn:
        return pd.read_sql(sql, conn, params={"teams": teams, "lo": lo, "hi": hi})


def build_queues_live(
    game_date: date,
    contexts: list[LiveQueueContext],
    engine=None,
) -> dict[tuple[int, str], BullpenQueue]:
    """Build a pregame queue using active rosters and prior role/rest evidence."""
    if engine is None:
        from backend.db import engine as default_engine
        engine = default_engine

    from backend.data.mlb_api import fetch_active_pitchers
    from backend.team_mappings import TEAM_ID_BY_CODE

    teams = list({c.team for c in contexts})
    wl = _load_workload(game_date, teams, engine)
    return queues_from_workload(game_date, contexts, wl, {
        team: fetch_active_pitchers(TEAM_ID_BY_CODE[team])
        for team in teams if team in TEAM_ID_BY_CODE
    })


@dataclass(frozen=True)
class PitchingPlan:
    """An explicit game-specific pregame report, never a workload-based bulk guess."""
    game_pk: int
    side: str
    opener_id: int
    bulk_id: int
    opener_outs: int
    bulk_outs: int
    source: str
    confirmed_at: str

    def valid_for(self, context: LiveQueueContext, start_time) -> bool:
        if start_time is None or pd.isna(start_time):
            return False
        try:
            observed = pd.to_datetime(self.confirmed_at, utc=True)
            start = pd.to_datetime(start_time, utc=True)
            return bool(self.game_pk == context.game_pk and self.side == context.side
                        and self.opener_id == context.starter_id and self.bulk_id > 0
                        and self.bulk_id != self.opener_id and isinstance(self.source, str) and self.source.strip()
                        and observed < start and observed.date() >= start.date() - timedelta(days=2)
                        and 1 <= self.opener_outs <= 6 and 3 <= self.bulk_outs <= 21)
        except (TypeError, ValueError):
            return False


def queues_from_workload(
    game_date: date, contexts: list[LiveQueueContext], workload: pd.DataFrame,
    rosters: dict[str, list[int]],
) -> dict[tuple[int, str], BullpenQueue]:
    """Use prior appearances to identify regular roles, not today's bulk pitcher.

    Rotation arms and players with insufficient role history cannot enter the
    relief queue. Missing bullpen coverage ends in a neutral replacement arm.
    """
    wl = workload.copy()
    wl["game_date"] = pd.to_datetime(wl["game_date"]).dt.date
    wl = wl[(wl.game_date < game_date) & (wl.game_date >= game_date - timedelta(days=60))]
    out = {}
    for ctx in contexts:
        team = wl[wl.team == ctx.team].sort_values("game_date")
        # Sum doubleheader workload, rather than losing one appearance in a dict.
        recent = team[team.game_date >= game_date - timedelta(days=2)]
        yesterday = recent[recent.game_date == game_date - timedelta(days=1)].groupby("pitcher_id").outs.sum()
        two_days = recent.groupby("pitcher_id").outs.sum()
        histories = {int(pid): g.tail(8) for pid, g in team.groupby("pitcher_id")}
        starter_history = histories.get(ctx.starter_id, team.iloc[:0])
        starts = starter_history[starter_history.role == "SP"]
        regular_starter = len(starts) >= 2 and len(starts) >= len(starter_history) / 2
        # Known relievers opening receive short workloads; unresolved plans are
        # short as well, and suppress recommendations through usage_known.
        workloads = {ctx.starter_id: tuple(int(np.clip(x, 3, 21)) for x in starts.outs)
                     if regular_starter else (3,)}
        candidates = []
        for pid in rosters.get(ctx.team, []):
            if pid == ctx.starter_id:
                continue
            hist = histories.get(pid, team.iloc[:0])
            if len(hist) < 2 or (hist.tail(5).role != "RP").any():
                continue
            if yesterday.get(pid, 0) >= ELIG_OUTS_1D or two_days.get(pid, 0) >= ELIG_OUTS_2D:
                continue
            candidates.append((pid, len(hist), int(two_days.get(pid, 0))))
            workloads[pid] = tuple(int(np.clip(x, 1, 6)) for x in hist.outs) + (3, 3, 3)
        candidates.sort(key=lambda x: (-x[1], x[2], x[0]))
        ids = [p for p, _, _ in candidates]
        out[(ctx.game_pk, ctx.side)] = BullpenQueue(
            starter=ctx.starter_id, relievers=ids + [0],
            starter_role=0 if regular_starter else 1, workloads=workloads,
            usage_known=regular_starter and len(ids) >= 3,
            team_outs_2d=int(recent[recent.role == "RP"].outs.sum()) if len(team) else None,
        )
    return out


def apply_pitching_plan(queue: BullpenQueue, plan: PitchingPlan) -> BullpenQueue:
    """Caller must validate the report against this game's starter and first pitch."""
    relievers = [plan.bulk_id] + [p for p in queue.relievers if p not in (plan.bulk_id, plan.opener_id)]
    return BullpenQueue(
        starter=plan.opener_id, relievers=relievers, starter_role=1,
        workloads={**queue.workloads, plan.opener_id: (plan.opener_outs,), plan.bulk_id: (plan.bulk_outs,)},
        roles={plan.bulk_id: 0}, usage_known=True,
        team_outs_2d=queue.team_outs_2d,
    )
