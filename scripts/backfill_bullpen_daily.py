"""One-time backfill of bullpen_daily from cached analysis boxscores.

Reads analysis/cache/reliever_ip.parquet (created by 03_bullpen_fatigue.py)
and joins with `games` to upsert one row per (date, team) into bullpen_daily.

Idempotent: re-running just refreshes already-present rows.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
from sqlalchemy import text

from backend.db import engine

CACHE_PATH = Path(__file__).parent.parent / "analysis" / "cache" / "reliever_ip.parquet"


def main() -> None:
    if not CACHE_PATH.exists():
        sys.exit(f"Cache not found at {CACHE_PATH}. Run analysis/03_bullpen_fatigue.py first.")

    bx = pd.read_parquet(CACHE_PATH)
    print(f"Loaded {len(bx)} boxscore-team rows from cache")

    games = pd.read_sql(
        "SELECT game_pk, game_date, home_team, away_team FROM games", engine
    )
    df = bx.merge(games, on="game_pk", how="inner")
    df["team"] = np.where(df["team_side"] == "home", df["home_team"], df["away_team"])

    # If a team plays a doubleheader, sum both games' reliever outs that day
    daily = df.groupby(["game_date", "team"]).agg(
        reliever_outs=("reliever_outs", "sum"),
        starter_outs=("starter_outs", "sum"),
        n_relievers=("n_relievers", "sum"),
    ).reset_index()
    print(f"Aggregated to {len(daily)} (date, team) rows")

    # Upsert
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

    print(f"Upserted {len(daily)} rows into bullpen_daily")


if __name__ == "__main__":
    main()
