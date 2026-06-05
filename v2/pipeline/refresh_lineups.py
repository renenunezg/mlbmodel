"""Hourly refresh: re-scores games where the probable starter or posted lineup changed.

Re-fetches probable starters (a starter announced after the morning run otherwise
never lands in the DB until tomorrow) and reads model_outputs.lineup_hash to detect
lineup changes. A game is re-scored when either input changed. Rewrites both
model_outputs and model_outputs_season so the historical record matches the last
pre-game score. The unstarted-game filter in _fetch_scheduled_games freezes rows at
first pitch, so evaluation reflects the closest-to-first-pitch prediction.

Usage:
    python -m v2.pipeline.refresh_lineups [--date YYYY-MM-DD] [--n-sims N]
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import date

import pandas as pd
from sqlalchemy import text

from backend.data.mlb_api import fetch_lineup, fetch_probable_starters
from backend.db import engine
from pipeline import upsert_probable_starters
from v2.pipeline.score_games import score


def _lineup_hash(lineup: dict[str, list[int]]) -> str:
    combined = sorted(lineup.get("home", [])) + sorted(lineup.get("away", []))
    return hashlib.sha1(str(combined).encode()).hexdigest()[:16]


def _fetch_scheduled_games(date_str: str) -> pd.DataFrame:
    """Games on `date_str` that haven't started yet.

    The lock is on first pitch: `home_score IS NULL` rules out finals,
    `start_time > NOW()` rules out anything in progress. Once a game is
    underway its prediction is frozen — refresh can never rewrite it.
    """
    q = text("""
        SELECT g.game_pk, COALESCE(o.lineup_hash, '') AS stored_hash
        FROM games g
        LEFT JOIN model_outputs o
          ON o.game_pk = g.game_pk AND o.team = g.home_team
        WHERE g.game_date = :d
          AND g.home_score IS NULL
          AND g.start_time IS NOT NULL
          AND g.start_time > NOW()
    """)
    with engine.begin() as conn:
        return pd.read_sql(q, conn, params={"d": date_str})


def _starter_map(date_str: str) -> dict[int, tuple]:
    """game_pk -> (home_pitcher_id, away_pitcher_id); None on a side with no known starter."""
    q = text("""
        SELECT ps.game_pk, ps.is_home, ps.pitcher_id
        FROM probable_starters ps
        JOIN games g USING (game_pk)
        WHERE g.game_date = :d
    """)
    with engine.begin() as conn:
        df = pd.read_sql(q, conn, params={"d": date_str})
    out: dict[int, tuple] = {}
    for gp, grp in df.groupby("game_pk"):
        def _side(is_home):
            s = grp[grp.is_home == is_home]["pitcher_id"]
            return int(s.iloc[0]) if len(s) and pd.notna(s.iloc[0]) else None
        out[int(gp)] = (_side(True), _side(False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=str(date.today()))
    ap.add_argument("--n-sims", type=int, default=10000)
    args = ap.parse_args()

    games = _fetch_scheduled_games(args.date)
    if games.empty:
        print(f"[refresh_lineups] no unstarted games on {args.date}")
        return

    before = _starter_map(args.date)
    upsert_probable_starters(fetch_probable_starters(game_date=date.fromisoformat(args.date), days_ahead=0))
    after = _starter_map(args.date)

    changed = set()
    for row in games.itertuples(index=False):
        gp = int(row.game_pk)
        if before.get(gp) != after.get(gp):
            changed.add(gp)
        lineup = fetch_lineup(gp)
        h = _lineup_hash(lineup)
        if any(lineup.get("home", [])) and h != row.stored_hash:
            changed.add(gp)

    if not changed:
        print(f"[refresh_lineups] no starter or lineup changes on {args.date}")
        return

    changed = sorted(changed)
    print(f"[refresh_lineups] {len(changed)} games changed (starter/lineup): {changed}")
    # Weather updates as first pitch approaches; refresh it for the re-scored games.
    from backend.data.weather import fetch_weather
    for gp in changed:
        fetch_weather(gp)
    # Re-score changed games and mirror to season so /history and the eval
    # ledger see the post-lineup prediction. The unstarted-game filter above
    # means once a game starts, its season row is frozen.
    score(
        args.date,
        n_sims=args.n_sims,
        write=True,
        game_pks=changed,
        update_season=True,
    )


if __name__ == "__main__":
    main()
