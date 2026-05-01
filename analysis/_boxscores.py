"""Fetch and cache reliever IP per team per game from MLB Stats API boxscores.

Each game's boxscore lists pitchers in order with their IP. The first pitcher
per side is the starter; the rest are relievers. We sum reliever IP per team.

Cached to analysis/cache/reliever_ip.parquet.
"""
from __future__ import annotations

from pathlib import Path
import time

import pandas as pd
import requests

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = CACHE_DIR / "reliever_ip.parquet"

BASE_URL = "https://statsapi.mlb.com/api/v1"


def _ip_to_outs(ip_str: str | None) -> int:
    """MLB IP notation: 5.1 = 5 1/3 IP = 16 outs, 5.2 = 5 2/3 IP = 17 outs."""
    if ip_str is None or ip_str == "":
        return 0
    try:
        s = str(ip_str)
        if "." in s:
            whole, frac = s.split(".")
            return int(whole) * 3 + int(frac)
        return int(s) * 3
    except Exception:
        return 0


_STARTER_MIN_OUTS = 9  # 3 IP — anything less means the "starter" was an opener


def _classify_outs(outs_list: list[int]) -> tuple[int, int, int]:
    """Return (starter_outs, reliever_outs, n_relievers).

    Opener-aware: if the first pitcher threw fewer than 3 IP (9 outs), they are
    treated as a functional reliever and rolled into reliever_outs. This avoids
    inflating "reliever workload" only because of the bulk reliever, while
    correctly capturing that the opener also came out of the bullpen.
    """
    if not outs_list:
        return 0, 0, 0
    first = outs_list[0]
    if first >= _STARTER_MIN_OUTS:
        starter_outs = first
        reliever_outs = sum(outs_list[1:])
        n_rel = len(outs_list) - 1
    else:
        starter_outs = 0
        reliever_outs = sum(outs_list)
        n_rel = len(outs_list)
    return starter_outs, reliever_outs, max(0, n_rel)


def _fetch_one(game_pk: int) -> list[dict] | None:
    """Returns list of {game_pk, team_side, starter_outs, reliever_outs, ...}."""
    try:
        resp = requests.get(f"{BASE_URL}/game/{game_pk}/boxscore", timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  game_pk={game_pk}: {e}")
        return None

    rows = []
    for side in ("home", "away"):
        team_data = data.get("teams", {}).get(side, {})
        pitcher_ids = team_data.get("pitchers", [])  # in appearance order
        players = team_data.get("players", {})
        if not pitcher_ids:
            continue

        outs_list = []
        for pid in pitcher_ids:
            p = players.get(f"ID{pid}", {})
            stats = p.get("stats", {}).get("pitching", {})
            outs_list.append(_ip_to_outs(stats.get("inningsPitched")))

        starter_outs, reliever_outs, n_rel = _classify_outs(outs_list)
        rows.append({
            "game_pk": int(game_pk),
            "team_side": side,
            "starter_outs": starter_outs,
            "reliever_outs": reliever_outs,
            "n_relievers": n_rel,
            "total_outs": sum(outs_list),
        })
    return rows


def load_cache() -> pd.DataFrame:
    if CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)
    return pd.DataFrame(columns=[
        "game_pk", "team_side", "starter_outs", "reliever_outs",
        "n_relievers", "total_outs",
    ])


def fetch_boxscores(game_pks: list[int], sleep: float = 0.05) -> pd.DataFrame:
    cache = load_cache()
    cached_ids = set(cache["game_pk"].unique())
    missing = [g for g in game_pks if g not in cached_ids]

    if not missing:
        print(f"  All {len(game_pks)} boxscores cached.")
        return cache[cache["game_pk"].isin(game_pks)]

    print(f"  {len(missing)} boxscores to fetch (cached: {len(cached_ids)})...")
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
        print(f"  Cached {len(new_rows)} new rows. Total: {len(cache)}.")

    return cache[cache["game_pk"].isin(game_pks)]
