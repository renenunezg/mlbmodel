"""Hourly lineup refresh: re-scores games where posted lineups differ from what was used.

Reads model_outputs_v2.lineup_hash to detect changes. Only rewrites today's
model_outputs_v2 rows — never touches model_outputs_season_v2.

Usage:
    python -m v2.pipeline.refresh_lineups [--date YYYY-MM-DD] [--n-sims N]
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import date

import pandas as pd
from sqlalchemy import text

from backend.data.mlb_api import fetch_lineup
from backend.db import engine
from v2.pipeline.score_games import score


def _lineup_hash(lineup: dict[str, list[int]]) -> str:
    combined = sorted(lineup.get("home", [])) + sorted(lineup.get("away", []))
    return hashlib.sha1(str(combined).encode()).hexdigest()[:16]


def _fetch_scheduled_games(date_str: str) -> pd.DataFrame:
    q = text("""
        SELECT g.game_pk, COALESCE(o.lineup_hash, '') AS stored_hash
        FROM games g
        LEFT JOIN model_outputs_v2 o
          ON o.game_pk = g.game_pk AND o.team = g.home_team
        WHERE g.game_date = :d
          AND g.home_score IS NULL
    """)
    with engine.begin() as conn:
        return pd.read_sql(q, conn, params={"d": date_str})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=str(date.today()))
    ap.add_argument("--n-sims", type=int, default=10000)
    args = ap.parse_args()

    games = _fetch_scheduled_games(args.date)
    if games.empty:
        print(f"[refresh_lineups] no scheduled games on {args.date}")
        return

    changed = []
    for row in games.itertuples(index=False):
        lineup = fetch_lineup(row.game_pk)
        h = _lineup_hash(lineup)
        if any(lineup.get("home", [])) and h != row.stored_hash:
            changed.append(row.game_pk)

    if not changed:
        print(f"[refresh_lineups] no lineup changes on {args.date}")
        return

    print(f"[refresh_lineups] {len(changed)} games with new lineups: {changed}")
    score(args.date, n_sims=args.n_sims, write=True)


if __name__ == "__main__":
    main()
