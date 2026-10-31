"""A season's frames cut at a date: what was knowable before it, and what came after.

``frames_before`` builds the inputs, outcome counts and positions from the pitches
before ``as_of`` (None before the first pitch), ``counts_after`` the outcome counts
from the rest of the season. Both feed the forward evaluation and the in-season
development pairs of the grade fit. A completed season's cuts are cached under
cache/players/cuts/; the current season is always recut because its pitches grow.
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache

import pandas as pd

from v2.players import bio, inputs
from v2.players.inputs import CACHE_DIR

CUT_DIR = CACHE_DIR / "cuts"
MONTHS = (4, 5, 6, 7, 8, 9)


def month_starts(season: int) -> list[date]:
    """Opening day and the month starts inside the regular season: the cuts a season is graded at."""
    start, end = bio.season_dates(season)
    return [start] + [d for month in MONTHS if start < (d := date(season, month, 1)) <= end]


@lru_cache(maxsize=1)
def load_pitches(season: int) -> pd.DataFrame:
    """The season's pitches with ``game_date`` as a date; the last season read stays in memory for its other cuts."""
    pitches = inputs.load_pitches(season)
    pitches["game_date"] = pd.to_datetime(pitches["game_date"]).dt.date
    return pitches


def _cached(season: int) -> bool:
    return season < date.today().year


def _path(season: int, as_of: date, kind: str):
    return CUT_DIR / f"{season}_{as_of.isoformat()}_{kind}.parquet"


def frames_before(season: int, as_of: date) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """(inputs, outcome counts, positions) from the season's pitches before ``as_of``; None before the first pitch."""
    kinds = ("inputs", "counts", "positions")
    paths = [_path(season, as_of, k) for k in kinds]
    if _cached(season) and all(p.exists() for p in paths):
        frames = tuple(pd.read_parquet(p) for p in paths)
        return None if frames[0].empty else frames
    pitches = load_pitches(season)
    before = pitches[pitches["game_date"] < as_of]
    if before.empty:
        frames = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    else:
        frames = (inputs.statcast_inputs(before, season), inputs.count_outcomes(before, season), bio.positions([season], before=as_of))
    if _cached(season):
        CUT_DIR.mkdir(parents=True, exist_ok=True)
        for frame, path in zip(frames, paths, strict=True):
            frame.to_parquet(path, index=False)
    return None if before.empty else frames


def counts_after(season: int, as_of: date) -> pd.DataFrame:
    """Outcome counts per (player, role) from the season's pitches on and after ``as_of``."""
    path = _path(season, as_of, "rest")
    if _cached(season) and path.exists():
        return pd.read_parquet(path)
    pitches = load_pitches(season)
    counts = inputs.count_outcomes(pitches[pitches["game_date"] >= as_of], season)
    if _cached(season):
        CUT_DIR.mkdir(parents=True, exist_ok=True)
        counts.to_parquet(path, index=False)
    return counts
