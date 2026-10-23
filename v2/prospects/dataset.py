"""Snapshot datasets: features, eligibility and (for fitting) matured targets, one row per player and season.

Season tables from ``events`` are cached per season under ``cache/prospects/tables/`` and rebuilt
when the feed cache gained games or ``TABLE_VERSION`` changed. A player is a hitter or a pitcher in a
snapshot by which sample is larger; a two-way player keeps the larger role. Rows below the sample floor are kept with
``gradeable`` False so scoring can publish them as not yet graded.
"""
from __future__ import annotations

import json
import logging
from datetime import date

import pandas as pd

from v2.prospects import events, features, identity, targets
from v2.prospects.feeds import CACHE_DIR, FIRST_SEASON, NO_SEASON, SPORT_IDS, read_kind

log = logging.getLogger(__name__)

TABLES_DIR = CACHE_DIR / "tables"
WINDOW = len(features.WINDOW_WEIGHTS)
TABLE_VERSION = 2  # bump when events changes what a season table holds; 2: call-code counts only from fully logged games
# Heuristic floors and age range, not fitted: a window below the floor is published ungraded.
FLOOR_FULL = {"bat": 150.0, "pit": 120.0}
FLOOR_COMPLEX = {"bat": 100.0, "pit": 80.0}
COMPLEX_MAX_RANK = events.LEVEL_RANK["rk+"]
MIN_AGE, MAX_AGE = 16.0, 30.0


def _feed_game_count(season: int) -> int:
    return sum(len(read_kind(s, season, "games")) for s in SPORT_IDS)


def season_tables(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hitter and pitcher season-level tables, cached until the feed cache for the season grows."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    meta = TABLES_DIR / f"meta_{season}.json"
    n_games = _feed_game_count(season)
    paths = (TABLES_DIR / f"hitters_{season}.parquet", TABLES_DIR / f"pitchers_{season}.parquet")
    if meta.exists() and all(p.exists() for p in paths) and json.loads(meta.read_text()) == {"n_games": n_games, "version": TABLE_VERSION}:
        return pd.read_parquet(paths[0]), pd.read_parquet(paths[1])
    hitters, pitchers = events.season_tables([season])
    hitters.to_parquet(paths[0], index=False)
    pitchers.to_parquet(paths[1], index=False)
    meta.write_text(json.dumps({"n_games": n_games, "version": TABLE_VERSION}))
    return hitters, pitchers


def load_tables(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tables for every season asked for; a season without feeds is an error, never a silently thinner window.

    Seasons before the first feed season and the season with no minor league play are not asked for.
    """
    hs, ps = [], []
    for season in seasons:
        if season < FIRST_SEASON or season in NO_SEASON:
            continue
        h, p = season_tables(season)
        hs.append(h)
        ps.append(p)
    return pd.concat(hs, ignore_index=True), pd.concat(ps, ignore_index=True)


def _assign_role(fh: pd.DataFrame, fp: pd.DataFrame) -> pd.DataFrame:
    """One row per player: the role with the larger sample."""
    both = set(fh["player_id"]) & set(fp["player_id"])
    if both:
        hs = fh.set_index("player_id")["sample"]
        ps = fp.set_index("player_id")["sample"]
        pit_wins = {pid for pid in both if ps[pid] > hs[pid]}
        fh = fh[~fh["player_id"].isin(pit_wins)]
        fp = fp[~fp["player_id"].isin(both - pit_wins)]
    return pd.concat([fh, fp], ignore_index=True)


def _floor(frame: pd.DataFrame) -> pd.Series:
    complex_only = frame["career_max_level_rank"].fillna(0) <= COMPLEX_MAX_RANK
    return pd.Series([ (FLOOR_COMPLEX if c else FLOOR_FULL)[r] for c, r in zip(complex_only, frame["role"])], index=frame.index)


def snapshot(season: int, tables: tuple[pd.DataFrame, pd.DataFrame], bio: pd.DataFrame, roster: pd.DataFrame,
             baselines: tuple[pd.DataFrame, pd.DataFrame], as_of: date | None = None) -> pd.DataFrame:
    """Feature rows for one snapshot season with role, eligibility and floor flags."""
    hitters, pitchers = tables
    fh = features.hitter_features(hitters, bio, season, baselines[0])
    fp = features.pitcher_features(pitchers, bio, season, baselines[1])
    frame = _assign_role(fh, fp)
    if frame.empty:
        return frame
    when = as_of or features.snapshot_date(season)
    status = identity.rookie_status(frame["player_id"].tolist(), when, roster)
    frame = frame.merge(status, on="player_id", how="left")
    frame["floor"] = _floor(frame)
    frame["gradeable"] = frame["sample"] >= frame["floor"]
    frame["in_age_range"] = frame["age"].between(MIN_AGE, MAX_AGE)
    frame["as_of"] = pd.Timestamp(when)
    return frame


def baselines(tables: tuple[pd.DataFrame, pd.DataFrame], bio: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (features.level_age_baseline(tables[0], bio, "pa", features.REGULAR_PA),
            features.level_age_baseline(tables[1], bio, "bf", features.REGULAR_BF))


def build(snapshot_seasons: list[int], last_season: int | None = None) -> pd.DataFrame:
    """Rookie-eligible snapshot rows in the age range for the seasons; with ``last_season`` the matured targets are attached.

    The floor is a flag, not a filter, so scoring can publish ungraded rows.
    """
    seasons = list(range(min(snapshot_seasons) - WINDOW + 1, max(snapshot_seasons) + 1))
    tables = load_tables(seasons)
    ids = sorted(set(tables[0]["player_id"]) | set(tables[1]["player_id"]))
    bio, roster = identity.people(ids)
    base = baselines(tables, bio)
    frames = []
    for season in snapshot_seasons:
        rows = snapshot(season, tables, bio, roster, base)
        if rows.empty:
            continue
        rows = rows[rows["rookie_eligible"] & rows["in_age_range"]]
        frames.append(rows)
        log.info(f"dataset[{season}]: {len(rows):,} rows, {int(rows['gradeable'].sum()):,} above the floor")
    out = pd.concat(frames, ignore_index=True)
    if last_season is not None:
        out = out.merge(targets.outcomes(out[["player_id", "season"]], last_season), on=["player_id", "season"], how="left")
    return out
