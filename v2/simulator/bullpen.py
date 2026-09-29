"""Pregame pitching roles, workload distributions, and explicit opener/bulk plans."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text

from backend.data.game_types import POSTSEASON_GAME_TYPES

log = logging.getLogger(__name__)
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
    roster_source: str = "unknown"
    availability: dict[int, str] = field(default_factory=dict)
    availability_assumption: str = "recent_workload"

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
    game_type: str = "R"


def _load_workload(game_date: date, teams: list[str], engine) -> pd.DataFrame:
    sql = text("""
        SELECT game_date, pitcher_id, team, outs, role
        FROM pitcher_workload
        WHERE team = ANY(:teams)
          AND game_date >= :lo AND game_date < :hi
    """)
    lo = max(date(game_date.year, 1, 1), game_date - timedelta(days=60))
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

    from backend.data.mlb_api import fetch_active_pitchers, fetch_game_pitchers, fetch_probable_starters
    from backend.team_mappings import TEAM_ID_BY_CODE

    teams = list({c.team for c in contexts})
    wl = _load_workload(game_date, teams, engine)
    regular_teams = {c.team for c in contexts if c.game_type not in POSTSEASON_GAME_TYPES}
    rosters = {team: fetch_active_pitchers(TEAM_ID_BY_CODE[team], game_date)
               for team in regular_teams if team in TEAM_ID_BY_CODE}
    playoff_games = {c.game_pk for c in contexts if c.game_type in POSTSEASON_GAME_TYPES}
    game_rosters = {}
    reserved = None
    if playoff_games:
        # Probable future starters remain rotation commitments, not assumed
        # relief plans. Failure leaves their availability unconfirmed.
        try:
            upcoming = fetch_probable_starters(game_date + timedelta(days=1), days_ahead=1)
            reserved = set(upcoming.pitcher_id.dropna().astype(int)) if not upcoming.empty else set()
        except Exception as exc:
            log.warning(f"Could not verify future rotation commitments: {exc}")
        for gp in playoff_games:
            try:
                for side, ids in fetch_game_pitchers(gp).items():
                    game_rosters[(gp, side)] = ids
            except Exception as exc:
                log.warning(f"Could not verify game {gp} pitching roster: {exc}")
    return queues_from_workload(game_date, contexts, wl, rosters, game_rosters, reserved)


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
    game_rosters: dict[tuple[int, str], list[int]] | None = None,
    reserved_starters: set[int] | None = None,
) -> dict[tuple[int, str], BullpenQueue]:
    """Use prior appearances to identify regular roles, not today's bulk pitcher.

    Regular-season queues require proven relief roles and apply fatigue gates.
    Playoff bullpen arms are explicitly assumed fully rested. Relief arms come
    first; other eligible pitchers get short emergency relief, never a guessed
    bulk assignment. Announced future starters remain rotation commitments.
    """
    wl = workload.copy()
    wl["game_date"] = pd.to_datetime(wl["game_date"]).dt.date
    lo = max(date(game_date.year, 1, 1), game_date - timedelta(days=60))
    wl = wl[(wl.game_date < game_date) & (wl.game_date >= lo)]
    out = {}
    for ctx in contexts:
        postseason = ctx.game_type in POSTSEASON_GAME_TYPES
        roster = ((game_rosters or {}).get((ctx.game_pk, ctx.side), []) if postseason
                  else rosters.get(ctx.team, []))
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
        availability = {ctx.starter_id: "starting"}
        proven_relief = 0
        for pid in dict.fromkeys(roster):
            if pid == ctx.starter_id:
                continue
            hist = histories.get(pid, team.iloc[:0])
            relief = hist[hist.role == "RP"]
            regular_reliever = len(hist) >= 2 and (hist.tail(5).role == "RP").all()
            if not postseason and not regular_reliever:
                availability[pid] = "unconfirmed_relief_role"
                continue
            if postseason and pid in (reserved_starters or set()):
                availability[pid] = "scheduled_starter"
                continue
            if postseason and not regular_reliever and reserved_starters is None:
                availability[pid] = "unconfirmed_rotation_availability"
                continue
            if not postseason and (yesterday.get(pid, 0) >= ELIG_OUTS_1D or two_days.get(pid, 0) >= ELIG_OUTS_2D):
                availability[pid] = "recent_workload"
                continue
            priority = 0 if regular_reliever else 1
            availability[pid] = "relief" if regular_reliever else "emergency_relief"
            proven_relief += int(regular_reliever)
            appearances = relief if postseason else hist
            candidates.append((pid, priority, len(appearances), 0 if postseason else int(two_days.get(pid, 0))))
            workloads[pid] = (tuple(int(np.clip(x, 1, 6)) for x in appearances.outs) + (3, 3, 3)
                              if regular_reliever else (3,))
        candidates.sort(key=lambda x: (x[1], -x[2], x[3], x[0]))
        ids = [p for p, _, _, _ in candidates]
        out[(ctx.game_pk, ctx.side)] = BullpenQueue(
            starter=ctx.starter_id, relievers=ids + [0],
            starter_role=0 if regular_starter else 1, workloads=workloads,
            usage_known=regular_starter and proven_relief >= 3 and (not postseason or ctx.starter_id in roster),
            team_outs_2d=int(recent[recent.role == "RP"].outs.sum()) if len(team) else None,
            roster_source="game_boxscore" if postseason and roster else "active_roster" if not postseason and roster else "unknown",
            availability=availability,
            availability_assumption="full_rest" if postseason else "recent_workload",
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
        roster_source=queue.roster_source,
        availability={**queue.availability, plan.opener_id: "confirmed_opener", plan.bulk_id: "confirmed_bulk"},
        availability_assumption=queue.availability_assumption,
    )
