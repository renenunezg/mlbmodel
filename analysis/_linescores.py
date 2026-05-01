"""Fetch and cache MLB line scores (runs per inning) via the MLB Stats API.

Cached to analysis/cache/linescores.parquet so we only hit the API once per game.
Read-only: never writes to the production DB.
"""
from __future__ import annotations

from pathlib import Path
import time

import pandas as pd
import requests

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = CACHE_DIR / "linescores.parquet"

BASE_URL = "https://statsapi.mlb.com/api/v1"


def _fetch_one(game_pk: int) -> list[dict] | None:
    """Return list of {game_pk, inning, home_runs, away_runs} or None on failure."""
    try:
        resp = requests.get(f"{BASE_URL}/game/{game_pk}/linescore", timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  game_pk={game_pk}: {e}")
        return None

    rows = []
    for inn in data.get("innings", []):
        rows.append({
            "game_pk": int(game_pk),
            "inning": int(inn.get("num")),
            "home_runs": int((inn.get("home") or {}).get("runs") or 0),
            "away_runs": int((inn.get("away") or {}).get("runs") or 0),
        })
    return rows


def load_cache() -> pd.DataFrame:
    if CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)
    return pd.DataFrame(columns=["game_pk", "inning", "home_runs", "away_runs"])


def fetch_linescores(game_pks: list[int], sleep: float = 0.05) -> pd.DataFrame:
    """Fetch line scores for the given game_pks, caching results.

    Re-uses cached data; only fetches games not already in cache.
    """
    cache = load_cache()
    cached_ids = set(cache["game_pk"].unique())
    missing = [g for g in game_pks if g not in cached_ids]

    if not missing:
        print(f"  All {len(game_pks)} games cached.")
        return cache[cache["game_pk"].isin(game_pks)]

    print(f"  {len(missing)} games to fetch (cached: {len(cached_ids)})...")
    new_rows = []
    for i, game_pk in enumerate(missing, 1):
        rows = _fetch_one(game_pk)
        if rows:
            new_rows.extend(rows)
        if i % 50 == 0:
            print(f"    fetched {i}/{len(missing)}")
        time.sleep(sleep)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        cache = pd.concat([cache, new_df], ignore_index=True)
        cache.to_parquet(CACHE_PATH, index=False)
        print(f"  Cached {len(new_rows)} new inning rows. Total: {len(cache)}.")

    return cache[cache["game_pk"].isin(game_pks)]
