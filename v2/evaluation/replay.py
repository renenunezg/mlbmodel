"""Populate model_outputs_season for a date range by replaying score_games.

Idempotent: append_season() in v2/markets/writer.py upserts on (game_pk, team).
Re-running over a date that's already populated is safe but wastes wall time;
use --resume to skip dates that already have v2 rows in the season table.

Usage:
    python -m v2.evaluation.replay --start 2025-03-27 --end 2025-09-28 --n-sims 2000
    python -m v2.evaluation.replay --start 2025-03-27 --end 2025-09-28 --resume
"""
from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from backend.db import engine
from v2.pipeline.score_games import score


LOG_PATH = Path(__file__).resolve().parent / "replay_progress.log"


def _daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _dates_already_done(start: date, end: date) -> set[date]:
    """Distinct dates in model_outputs_season within [start, end]."""
    q = text(
        "SELECT DISTINCT date::date AS d FROM model_outputs_season "
        "WHERE date::date BETWEEN :s AND :e"
    )
    with engine.begin() as conn:
        rows = conn.execute(q, {"s": start, "e": end}).fetchall()
    return {r[0] for r in rows}


def _final_games_on(d: date) -> int:
    q = text("SELECT COUNT(*) FROM games WHERE game_date = :d AND status = 'Final'")
    with engine.begin() as conn:
        return int(conn.execute(q, {"d": d}).scalar() or 0)


def _log(date_str: str, games: int, wall_s: float, note: str = "") -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(f"{pd.Timestamp.utcnow().isoformat()}\t{date_str}\t{games}\t{wall_s:.1f}\t{note}\n")


def replay_range(start: date, end: date, n_sims: int, resume: bool, seed: int) -> None:
    done = _dates_already_done(start, end) if resume else set()
    if resume:
        print(f"[replay] resume: {len(done)} dates already in model_outputs_season")

    dates = list(_daterange(start, end))
    print(f"[replay] range {start} -> {end} ({len(dates)} dates), n_sims={n_sims}")

    for d in dates:
        if d in done:
            print(f"[replay] {d}: skip (already in season table)")
            continue
        n_final = _final_games_on(d)
        if n_final == 0:
            print(f"[replay] {d}: skip (no Final games)")
            _log(str(d), 0, 0.0, "no_final")
            continue

        t0 = time.time()
        try:
            df = score(str(d), n_sims=n_sims, write=True, seed=seed, freeze_started=False)
        except Exception as e:
            wall = time.time() - t0
            print(f"[replay] {d}: ERROR after {wall:.1f}s: {type(e).__name__}: {e}")
            _log(str(d), 0, wall, f"error:{type(e).__name__}")
            continue
        wall = time.time() - t0
        n_rows = len(df) if df is not None else 0
        sources = df["lineup_source"].value_counts().to_dict() if n_rows else {}
        sources_str = ",".join(f"{k}={v}" for k, v in sources.items())
        print(f"[replay] {d}: {n_rows} rows in {wall:.1f}s [{sources_str}]")
        _log(str(d), n_rows, wall, sources_str)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--n-sims", type=int, default=2000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    start = pd.Timestamp(args.start).date()
    end = pd.Timestamp(args.end).date()
    replay_range(start, end, args.n_sims, args.resume, args.seed)


if __name__ == "__main__":
    main()
