"""Idempotent multi-year Statcast cache builder.

Populates cache/statcast_{year}.parquet (shared with v1). Re-running for a
year that already has a parquet fetches only the delta past the cached max
date.

Usage:
    python -m v2.data.build_cache --years 2026
    python -m v2.data.build_cache --years 2024 2025 2026
    python -m v2.data.build_cache --years 2025 --force   # discard existing cache
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"
DEDUPE_COLS = ["game_pk", "at_bat_number", "pitch_number"]


def fetch_year(year: int, force: bool = False) -> pd.DataFrame:
    """Populate cache/statcast_{year}.parquet, fetching only the missing tail."""
    from pybaseball import statcast

    today = date.today()
    cache_path = CACHE_DIR / f"statcast_{year}.parquet"
    season_start = date(year, 3, 25)
    season_end = date(year, 11, 30) if year < today.year else today

    cached = pd.DataFrame()
    fetch_from = season_start

    if cache_path.exists() and not force:
        cached = pd.read_parquet(cache_path)
        max_cached = pd.to_datetime(cached["game_date"]).max().date()
        fetch_from = max(season_start, max_cached)
        print(f"[{year}] cached through {max_cached} ({len(cached):,} pitches)")

    if fetch_from >= season_end and not cached.empty:
        print(f"[{year}] cache up to date through {season_end}; skipping fetch")
        return cached

    print(f"[{year}] fetching {fetch_from} → {season_end}")
    new_df = statcast(start_dt=str(fetch_from), end_dt=str(season_end))
    new_df = new_df[new_df["game_type"] == "R"]

    if not cached.empty:
        df = pd.concat([cached, new_df], ignore_index=True).drop_duplicates(
            subset=DEDUPE_COLS, keep="last"
        )
    else:
        df = new_df

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"[{year}] saved {len(df):,} pitches across {df['game_pk'].nunique():,} games → {cache_path}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build per-year Statcast parquet cache for v2.")
    ap.add_argument("--years", type=int, nargs="+", required=True, help="Seasons to fetch (e.g. 2024 2025 2026)")
    ap.add_argument("--force", action="store_true", help="Discard existing cache and refetch")
    args = ap.parse_args()

    for year in sorted(args.years):
        fetch_year(year, force=args.force)


if __name__ == "__main__":
    main()
