"""Idempotent per-year Statcast cache builder for prospect levels: Triple-A, the Florida State League, spring training.

Populates cache/statcast_{level}_{year}.parquet (``aaa`` by default, ``fsl`` for the Single-A
Florida State League, tracked since 2021, ``spring`` for MLB spring training games, where many
prospects pitch on non-roster invitations) with the same pitch columns as the MLB cache. Baseball
Savant's minor-league search mixes every tracked level, so each day's pitches are kept only for
the game_pks on that level's schedule.
Re-running a year fetches only the dates past the cached max date, and the
parquet is checkpointed every CHECKPOINT_DAYS so a time-boxed run keeps its
progress.

Usage:
    python -m v2.data.build_aaa_cache --years 2023 2024 2025 2026
    python -m v2.data.build_aaa_cache --years 2025 --force
    python -m v2.data.build_aaa_cache --level fsl --years 2021 2022 2023 2024 2025 2026
"""
from __future__ import annotations

import argparse
import io
import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from backend.data.game_types import regular_season_pitches
from backend.log import setup_logging
from v2.data.build_cache import DEDUPE_COLS

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"
SCHEDULE = "https://statsapi.mlb.com/api/v1/schedule"
SEARCH = "https://baseballsavant.mlb.com/statcast-search-minors/csv"
LEVELS = {"aaa": (11, None, "R"), "fsl": (14, 123, "R"), "spring": (1, None, "S")}  # level -> (sport id, league id or None, game type)
MLB_SEARCH = "https://baseballsavant.mlb.com/statcast_search/csv"
USER_AGENT = "Mozilla/5.0"
ATTEMPTS = 3
RETRY_SECONDS = 10
PAUSE_SECONDS = 1.0
CHECKPOINT_DAYS = 20


def schedule(year: int, level: str) -> dict[date, set[int]]:
    """Regular-season game_pks of the level by date."""
    sport_id, league_id, game_type = LEVELS[level]
    response = requests.get(SCHEDULE, params={"sportId": sport_id, "season": year, "gameType": game_type, "hydrate": "team"}, timeout=120)
    response.raise_for_status()
    out: dict[date, set[int]] = {}
    for day in response.json().get("dates", []):
        games = {g["gamePk"] for g in day["games"]
                 if league_id is None or g["teams"]["home"]["team"].get("league", {}).get("id") == league_id}
        if games:
            out[date.fromisoformat(day["date"])] = games
    if not out:
        raise ValueError(f"No {level} schedule for {year}")
    return out


def fetch_day(day: date, level: str = "aaa") -> pd.DataFrame:
    if level == "spring":
        url, params = MLB_SEARCH, {"all": "true", "type": "details", "player_type": "batter", "hfGT": "S|",
                                   "game_date_gt": day.isoformat(), "game_date_lt": day.isoformat()}
    else:
        url, params = SEARCH, {"all": "true", "type": "details", "minors": "true", "player_type": "batter",
                               "game_date_gt": day.isoformat(), "game_date_lt": day.isoformat()}
    for attempt in range(1, ATTEMPTS + 1):
        try:
            response = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=180)
            response.raise_for_status()
            body = response.content.decode("utf-8-sig")
            if body.lstrip().startswith("<"):
                raise RuntimeError("HTML instead of CSV")
            return pd.read_csv(io.StringIO(body), low_memory=False) if body.strip() else pd.DataFrame()
        except Exception as exc:
            if attempt == ATTEMPTS:
                raise
            log.warning(f"{day}: {exc}; retrying in {RETRY_SECONDS * attempt}s")
            time.sleep(RETRY_SECONDS * attempt)
    raise RuntimeError(day)


def season_pitches(frame: pd.DataFrame, year: int, level: str) -> pd.DataFrame:
    """The level's game type in the year: regular season, or spring training for ``spring``."""
    if LEVELS[level][2] == "R":
        return regular_season_pitches(frame, year)
    if frame.empty:
        return frame.copy()
    keep = frame["game_type"].eq("S") & pd.to_datetime(frame["game_date"]).dt.year.eq(year)
    return frame.loc[keep].copy()


def _save(df: pd.DataFrame, path: Path) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".tmp")
    df.to_parquet(pending, index=False)
    pending.replace(path)


def fetch_year(year: int, force: bool = False, level: str = "aaa") -> pd.DataFrame:
    """Populate cache/statcast_{level}_{year}.parquet, fetching only the missing dates."""
    if year > date.today().year:
        raise ValueError("Cannot build a Statcast cache for a future season")
    cache_path = CACHE_DIR / f"statcast_{level}_{year}.parquet"
    games = schedule(year, level)
    df = pd.DataFrame()
    after = date(year, 1, 1)
    if cache_path.exists() and not force:
        df = season_pitches(pd.read_parquet(cache_path), year, level)
        if not df.empty:
            df["game_date"] = pd.to_datetime(df["game_date"])
            after = df["game_date"].max().date()
            log.info(f"[{year}] cached through {after} ({len(df):,} pitches)")
    days = sorted(d for d in games if d >= after and d < date.today())
    if not days:
        log.info(f"[{year}] cache up to date")
        return df
    parts = []
    for i, day in enumerate(days, 1):
        frame = fetch_day(day, level)
        if not frame.empty:
            parts.append(frame[frame["game_pk"].isin(games[day])])
        if i % CHECKPOINT_DAYS == 0 or i == len(days):
            new = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            if not new.empty:
                new["game_date"] = pd.to_datetime(new["game_date"])
                df = pd.concat([df, season_pitches(new, year, level)], ignore_index=True).drop_duplicates(subset=DEDUPE_COLS, keep="last")
                _save(df, cache_path)
            parts = []
            log.info(f"[{year}] {i}/{len(days)} days, {len(df):,} pitches saved")
        time.sleep(PAUSE_SECONDS)
    if df.empty:
        raise ValueError(f"No {level} pitches for {year}")
    log.info(f"[{year}] {len(df):,} pitches across {df['game_pk'].nunique():,} games → {cache_path}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build per-year minor-league Statcast parquet cache.")
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--level", choices=sorted(LEVELS), default="aaa")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for year in sorted(args.years):
        fetch_year(year, force=args.force, level=args.level)


if __name__ == "__main__":
    setup_logging()
    main()
