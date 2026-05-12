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


def _fetch_boxscore(game_pk: int) -> dict | None:
    """Returns {"teams": [...], "pitchers": [...]} or None on fetch failure.

    Per-team rows match the bullpen_daily schema. Per-pitcher rows tag each
    appearance as SP (first pitcher with >= 3 IP) or RP.
    """
    try:
        resp = requests.get(f"{BASE_URL}/game/{game_pk}/boxscore", timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  game_pk={game_pk}: {e}")
        return None

    team_rows = []
    pitcher_rows = []
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
        team_rows.append({
            "side": side,
            "game_pk": int(game_pk),
            "starter_outs": int(starter_outs),
            "reliever_outs": int(reliever_outs),
            "n_relievers": int(n_rel),
        })
        is_real_starter = outs_list[0] >= _STARTER_MIN_OUTS
        for idx, (pid, outs) in enumerate(zip(pitcher_ids, outs_list)):
            role = "SP" if (idx == 0 and is_real_starter) else "RP"
            pitcher_rows.append({
                "side": side,
                "pitcher_id": int(pid),
                "outs": int(outs),
                "role": role,
            })
    return {"teams": team_rows, "pitchers": pitcher_rows}


def update_bullpen_daily(sleep: float = 0.05) -> int:
    """Upsert bullpen_daily AND pitcher_workload for completed games not already present.

    Single boxscore fetch per game feeds both tables. Returns the number of
    (date, team) rows upserted into bullpen_daily.
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
    team_new = []
    pitcher_new = []
    for i, g in enumerate(todo.itertuples(index=False), 1):
        bx = _fetch_boxscore(int(g.game_pk))
        if bx:
            for r in bx["teams"]:
                team = g.home_team if r["side"] == "home" else g.away_team
                team_new.append({
                    "game_date": g.game_date,
                    "team": team,
                    "reliever_outs": r["reliever_outs"],
                    "starter_outs": r["starter_outs"],
                    "n_relievers": r["n_relievers"],
                })
            for p in bx["pitchers"]:
                team = g.home_team if p["side"] == "home" else g.away_team
                pitcher_new.append({
                    "game_date": g.game_date,
                    "pitcher_id": p["pitcher_id"],
                    "team": team,
                    "outs": p["outs"],
                    "role": p["role"],
                })
        if i % 25 == 0:
            print(f"    {i}/{len(todo)}")
        time.sleep(sleep)

    if not team_new:
        return 0

    df = pd.DataFrame(team_new)
    daily = df.groupby(["game_date", "team"]).agg(
        reliever_outs=("reliever_outs", "sum"),
        starter_outs=("starter_outs", "sum"),
        n_relievers=("n_relievers", "sum"),
    ).reset_index()

    upsert_team_sql = text("""
        INSERT INTO bullpen_daily (game_date, team, reliever_outs, starter_outs, n_relievers, updated_at)
        VALUES (:game_date, :team, :reliever_outs, :starter_outs, :n_relievers, NOW())
        ON CONFLICT (game_date, team) DO UPDATE SET
          reliever_outs = EXCLUDED.reliever_outs,
          starter_outs  = EXCLUDED.starter_outs,
          n_relievers   = EXCLUDED.n_relievers,
          updated_at    = NOW()
    """)

    upsert_pitcher_sql = text("""
        INSERT INTO pitcher_workload (game_date, pitcher_id, team, outs, role, updated_at)
        VALUES (:game_date, :pitcher_id, :team, :outs, :role, NOW())
        ON CONFLICT (game_date, pitcher_id) DO UPDATE SET
          team       = EXCLUDED.team,
          outs       = EXCLUDED.outs,
          role       = EXCLUDED.role,
          updated_at = NOW()
    """)

    with engine.begin() as conn:
        for _, r in daily.iterrows():
            conn.execute(upsert_team_sql, {
                "game_date": r["game_date"],
                "team": r["team"],
                "reliever_outs": int(r["reliever_outs"]),
                "starter_outs": int(r["starter_outs"]),
                "n_relievers": int(r["n_relievers"]),
            })
        # Doubleheaders: a pitcher can appear twice on the same date for the
        # same team. Sum their outs and take whichever role appeared first.
        if pitcher_new:
            pdf = pd.DataFrame(pitcher_new)
            pdaily = pdf.groupby(["game_date", "pitcher_id"]).agg(
                team=("team", "first"),
                outs=("outs", "sum"),
                role=("role", "first"),
            ).reset_index()
            for _, r in pdaily.iterrows():
                conn.execute(upsert_pitcher_sql, {
                    "game_date": r["game_date"],
                    "pitcher_id": int(r["pitcher_id"]),
                    "team": r["team"],
                    "outs": int(r["outs"]),
                    "role": r["role"],
                })

    print(f"  Upserted {len(daily)} (date, team) bullpen rows + {len(pitcher_new)} pitcher rows")
    return len(daily)
