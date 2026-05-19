"""Backfill the weather table for every game in the statcast parquet caches.

The games table only goes back to 2026-03-25, so 2024/25 game_pks live only in
cache/statcast_{year}.parquet - which is also the exact set the weather
coefficient fit joins against. Idempotent (fetch_weather upserts).

    env/bin/python -m v2.data.backfill_weather --years 2024 2025 2026
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from backend.data.weather import fetch_weather

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2024, 2025, 2026])
    ap.add_argument("--sleep", type=float, default=0.05)
    args = ap.parse_args()

    pks: set[int] = set()
    for y in args.years:
        path = CACHE_DIR / f"statcast_{y}.parquet"
        if not path.exists():
            print(f"  skip {y}: {path} missing")
            continue
        gp = pd.read_parquet(path, columns=["game_pk"])["game_pk"].astype(int).unique()
        pks.update(gp.tolist())
        print(f"  {y}: {len(gp)} games")

    pks = sorted(pks)
    print(f"total distinct games: {len(pks)}")
    n = 0
    for i, gp in enumerate(pks, 1):
        if fetch_weather(gp) is not None:
            n += 1
        if i % 250 == 0:
            print(f"  {i}/{len(pks)} ({n} populated)")
        time.sleep(args.sleep)
    print(f"backfill_weather: {n}/{len(pks)} games populated")


if __name__ == "__main__":
    main()
