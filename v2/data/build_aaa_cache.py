"""Idempotent per-year Triple-A Statcast cache builder.

Populates cache/statcast_aaa_{year}.parquet with the same pitch columns as the
MLB cache. Baseball Savant's minor-league search mixes every tracked level, so
each day's pitches are kept only for the game_pks on the Triple-A schedule.
Re-running a year fetches only the dates past the cached max date.

Usage:
    python -m v2.data.build_aaa_cache --years 2023 2024 2025 2026
    python -m v2.data.build_aaa_cache --years 2025 --force
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
TRIPLE_A_SPORT_ID = 11
USER_AGENT = "Mozilla/5.0"
ATTEMPTS = 3
RETRY_SECONDS = 10
PAUSE_SECONDS = 1.0


def triple_a_schedule(year: int) -> dict[date, set[int]]:
    """Regular-season Triple-A game_pks by date."""
    response = requests.get(SCHEDULE, params={"sportId": TRIPLE_A_SPORT_ID, "season": year, "gameType": "R"}, timeout=120)
    response.raise_for_status()
    out: dict[date, set[int]] = {}
    for day in response.json().get("dates", []):
        out[date.fromisoformat(day["date"])] = {g["gamePk"] for g in day["games"]}
    if not out:
        raise ValueError(f"No Triple-A schedule for {year}")
    return out


def fetch_day(day: date) -> pd.DataFrame:
    params = {"all": "true", "type": "details", "minors": "true", "player_type": "batter",
              "game_date_gt": day.isoformat(), "game_date_lt": day.isoformat()}
    for attempt in range(1, ATTEMPTS + 1):
        try:
            response = requests.get(SEARCH, params=params, headers={"User-Agent": USER_AGENT}, timeout=180)
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


def fetch_year(year: int, force: bool = False) -> pd.DataFrame:
    """Populate cache/statcast_aaa_{year}.parquet, fetching only the missing dates."""
    if year > date.today().year:
        raise ValueError("Cannot build a Statcast cache for a future season")
    cache_path = CACHE_DIR / f"statcast_aaa_{year}.parquet"
    schedule = triple_a_schedule(year)
    cached = pd.DataFrame()
    after = date(year, 1, 1)
    if cache_path.exists() and not force:
        cached = regular_season_pitches(pd.read_parquet(cache_path), year)
        if not cached.empty:
            after = pd.to_datetime(cached["game_date"]).max().date()
            log.info(f"[{year}] cached through {after} ({len(cached):,} pitches)")
    days = sorted(d for d in schedule if d >= after and d < date.today())
    parts = []
    for i, day in enumerate(days, 1):
        frame = fetch_day(day)
        if not frame.empty:
            frame = frame[frame["game_pk"].isin(schedule[day])]
            parts.append(frame)
        if i % 20 == 0:
            log.info(f"[{year}] {i}/{len(days)} days, {sum(len(p) for p in parts):,} pitches")
        time.sleep(PAUSE_SECONDS)
    new = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not new.empty:
        new["game_date"] = pd.to_datetime(new["game_date"])
        new = regular_season_pitches(new, year)
    if not cached.empty:
        cached["game_date"] = pd.to_datetime(cached["game_date"])
        df = pd.concat([cached, new], ignore_index=True).drop_duplicates(subset=DEDUPE_COLS, keep="last")
    else:
        df = new
    if df.empty:
        raise ValueError(f"No Triple-A pitches for {year}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    log.info(f"[{year}] saved {len(df):,} pitches across {df['game_pk'].nunique():,} games → {cache_path}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build per-year Triple-A Statcast parquet cache.")
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for year in sorted(args.years):
        fetch_year(year, force=args.force)


if __name__ == "__main__":
    setup_logging()
    main()
