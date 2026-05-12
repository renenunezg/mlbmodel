"""One-shot backfill of pitcher_workload from statcast parquet cache.

Reads cache/statcast_{year}.parquet, derives per-(date, pitcher) outs and role,
and upserts into Supabase. Idempotent.

Usage:
    python -m v2.data.backfill_pitcher_workload --years 2024 2025 2026
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from backend.db import engine

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"
STARTER_MIN_OUTS = 9


def _load_year(year: int) -> pd.DataFrame:
    path = CACHE_DIR / f"statcast_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    cols = [
        "game_date", "game_pk", "pitcher", "inning_topbot",
        "home_team", "away_team", "events", "outs_when_up",
        "inning", "at_bat_number",
    ]
    df = pd.read_parquet(path, columns=cols)
    df = df[df["events"].notna()].copy()
    df["pitcher"] = df["pitcher"].astype(np.int64)
    df["outs"] = df["outs_when_up"].fillna(0).astype(np.int64)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    return df


def _compute_outs_added(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["game_pk", "at_bat_number"]).reset_index(drop=True)
    grp = df.groupby("game_pk", sort=False)
    df["next_outs"] = grp["outs"].shift(-1)
    df["next_inning"] = grp["inning"].shift(-1)
    inning_ends = df["next_inning"].isna() | (df["next_inning"] != df["inning"])
    df["outs_added"] = np.where(
        inning_ends, 3 - df["outs"], df["next_outs"] - df["outs"]
    ).astype(np.int64)
    df.loc[df["events"] == "home_run", "outs_added"] = 0
    df = df[df["outs_added"].between(0, 3)].copy()
    return df


def _team_for_pitcher(df: pd.DataFrame) -> pd.Series:
    return np.where(df["inning_topbot"] == "Top", df["home_team"], df["away_team"])


def _classify_roles(per_game: pd.DataFrame) -> pd.DataFrame:
    """First pitcher per (game_pk, team) is SP if outs >= 9, else RP. Rest are RP."""
    per_game = per_game.sort_values(["game_pk", "team", "first_ab"]).reset_index(drop=True)
    per_game["order"] = per_game.groupby(["game_pk", "team"]).cumcount()
    is_real_starter = (per_game["order"] == 0) & (per_game["outs"] >= STARTER_MIN_OUTS)
    per_game["role"] = np.where(is_real_starter, "SP", "RP")
    return per_game


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, required=True)
    args = ap.parse_args()

    frames = []
    for y in args.years:
        print(f"Loading {y}...", flush=True)
        d = _load_year(y)
        print(f"  loaded {len(d)} PAs", flush=True)
        d = _compute_outs_added(d)
        d["team"] = _team_for_pitcher(d)
        per_pg = d.groupby(["game_date", "game_pk", "team", "pitcher"]).agg(
            outs=("outs_added", "sum"),
            first_ab=("at_bat_number", "min"),
        ).reset_index()
        per_pg = _classify_roles(per_pg)
        frames.append(per_pg)
        print(f"  {len(per_pg)} pitcher-game rows", flush=True)

    all_rows = pd.concat(frames, ignore_index=True)

    # Doubleheaders: sum outs across game_pks per (date, pitcher); keep first-seen role/team.
    daily = all_rows.sort_values(["game_date", "pitcher", "first_ab"]).groupby(
        ["game_date", "pitcher"]
    ).agg(
        team=("team", "first"),
        outs=("outs", "sum"),
        role=("role", "first"),
    ).reset_index()
    print(f"Aggregated to {len(daily)} (date, pitcher) rows")

    upsert_sql = text("""
        INSERT INTO pitcher_workload (game_date, pitcher_id, team, outs, role, updated_at)
        VALUES (:game_date, :pitcher_id, :team, :outs, :role, NOW())
        ON CONFLICT (game_date, pitcher_id) DO UPDATE SET
          team       = EXCLUDED.team,
          outs       = EXCLUDED.outs,
          role       = EXCLUDED.role,
          updated_at = NOW()
    """)

    payload = [
        {
            "game_date": r.game_date,
            "pitcher_id": int(r.pitcher),
            "team": r.team,
            "outs": int(r.outs),
            "role": r.role,
        }
        for r in daily.itertuples(index=False)
    ]

    BATCH = 500
    for i in range(0, len(payload), BATCH):
        with engine.begin() as conn:
            conn.execute(upsert_sql, payload[i:i + BATCH])
        print(f"  {min(i + BATCH, len(payload))}/{len(payload)}", flush=True)

    print(f"Upserted {len(daily)} rows into pitcher_workload", flush=True)


if __name__ == "__main__":
    main()
