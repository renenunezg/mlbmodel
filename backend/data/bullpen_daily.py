"""Per-game boxscore → per-team reliever/starter outs into bullpen_daily.

Backfill of historical data is one-time via scripts/backfill_bullpen_daily.py.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests
from sqlalchemy import text

from backend.db import engine

BASE_URL = "https://statsapi.mlb.com/api/v1"


def _ip_to_outs(ip_str) -> int:
    """MLB IP notation: 5.1 = 5 1/3 IP = 16 outs."""
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


_STARTER_MIN_OUTS = 9  # 3 IP - sub-3-IP "first pitcher" is treated as an opener


def _classify_outs(outs_list: list[int]) -> tuple[int, int, int]:
    """Opener-aware: sub-3-IP first pitcher rolls into reliever_outs. Returns (starter, reliever, n_rp)."""
    if not outs_list:
        return 0, 0, 0
    first = outs_list[0]
    if first >= _STARTER_MIN_OUTS:
        return first, sum(outs_list[1:]), max(0, len(outs_list) - 1)
    return 0, sum(outs_list), len(outs_list)


def _fetch_boxscore(game_pk: int) -> list[dict] | None:
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
        pitcher_ids = team_data.get("pitchers", [])
        players = team_data.get("players", {})
        if not pitcher_ids:
            continue
        outs_list = []
        for pid in pitcher_ids:
            stats = players.get(f"ID{pid}", {}).get("stats", {}).get("pitching", {})
            outs_list.append(_ip_to_outs(stats.get("inningsPitched")))
        starter_outs, reliever_outs, n_rel = _classify_outs(outs_list)
        rows.append({
            "side": side,
            "game_pk": int(game_pk),
            "starter_outs": int(starter_outs),
            "reliever_outs": int(reliever_outs),
            "n_relievers": int(n_rel),
        })
    return rows


def update_bullpen_daily(sleep: float = 0.05) -> int:
    """Upsert bullpen_daily for completed games not already present.

    Returns the number of (date, team) rows upserted.
    """
    games = pd.read_sql(
        """
        SELECT g.game_pk, g.game_date, g.home_team, g.away_team
        FROM games g
        WHERE g.home_score IS NOT NULL AND g.away_score IS NOT NULL
        """,
        engine,
    )
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.date

    # Skip games whose (date, both teams) are already in bullpen_daily
    existing = pd.read_sql(
        "SELECT game_date, team FROM bullpen_daily", engine,
    )
    existing["game_date"] = pd.to_datetime(existing["game_date"]).dt.date
    existing_pairs = set(zip(existing["game_date"], existing["team"]))

    todo = games[
        ~games.apply(
            lambda r: (r["game_date"], r["home_team"]) in existing_pairs
                      and (r["game_date"], r["away_team"]) in existing_pairs,
            axis=1,
        )
    ]

    if todo.empty:
        print(f"  bullpen_daily already current ({len(games)} games covered)")
        return 0

    print(f"  Fetching boxscores for {len(todo)} games...")
    new_rows = []
    for i, g in enumerate(todo.itertuples(index=False), 1):
        rows = _fetch_boxscore(int(g.game_pk))
        if rows:
            for r in rows:
                team = g.home_team if r["side"] == "home" else g.away_team
                new_rows.append({
                    "game_date": g.game_date,
                    "team": team,
                    "reliever_outs": r["reliever_outs"],
                    "starter_outs": r["starter_outs"],
                    "n_relievers": r["n_relievers"],
                })
        if i % 25 == 0:
            print(f"    {i}/{len(todo)}")
        time.sleep(sleep)

    if not new_rows:
        return 0

    df = pd.DataFrame(new_rows)
    daily = df.groupby(["game_date", "team"]).agg(
        reliever_outs=("reliever_outs", "sum"),
        starter_outs=("starter_outs", "sum"),
        n_relievers=("n_relievers", "sum"),
    ).reset_index()

    upsert_sql = text("""
        INSERT INTO bullpen_daily (game_date, team, reliever_outs, starter_outs, n_relievers, updated_at)
        VALUES (:game_date, :team, :reliever_outs, :starter_outs, :n_relievers, NOW())
        ON CONFLICT (game_date, team) DO UPDATE SET
          reliever_outs = EXCLUDED.reliever_outs,
          starter_outs  = EXCLUDED.starter_outs,
          n_relievers   = EXCLUDED.n_relievers,
          updated_at    = NOW()
    """)

    with engine.begin() as conn:
        for _, r in daily.iterrows():
            conn.execute(upsert_sql, {
                "game_date": r["game_date"],
                "team": r["team"],
                "reliever_outs": int(r["reliever_outs"]),
                "starter_outs": int(r["starter_outs"]),
                "n_relievers": int(r["n_relievers"]),
            })

    print(f"  Upserted {len(daily)} (date, team) bullpen rows")
    return len(daily)
