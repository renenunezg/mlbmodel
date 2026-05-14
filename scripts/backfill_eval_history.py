"""Replay daily evaluation across the full season using the unified v1+v2 view.

Each existing model_evaluation row was originally written with whatever data
sat in model_outputs_season at that time:
  - pre-cutover days: v1-only (the table was v1's then)
  - cutover day onward: v2-only (after the rename)
This produces a discontinuous history chart on the Performance tab.

This script loops over a date range, calling backend.evaluate_model.main(as_of=D)
for each date so every model_evaluation row is recomputed against the same
continuous source (model_outputs_season_unified). After it runs, the daily
eval chart reflects v1's real performance for pre-cutover dates and v2's
real performance for post-cutover dates.

Usage:
    env/bin/python -m scripts.backfill_eval_history --start 2026-03-27 --end 2026-05-13
    env/bin/python -m scripts.backfill_eval_history --start 2026-05-12   # cutover-only
"""
from __future__ import annotations

import argparse
import datetime
import sys
import traceback

from sqlalchemy import text

from backend import evaluate_model, strategy
from backend.db import engine
from backend.evaluate_model import main as eval_main


def _wipe_row(eval_date: datetime.date) -> None:
    """Delete the existing eval_date rows so the replay is a clean insert.

    The upsert in _write_evaluation_row skips None values to protect a
    populated row from a partial overwrite. That's the right policy for
    live eval, but it means a backfill that produces fewer fields than the
    stale row would leave stale numbers behind. Wipe the old row first.
    """
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM model_evaluation WHERE date = :d"),
            {"d": eval_date},
        )
        conn.execute(
            text("DELETE FROM model_calibration WHERE date = :d"),
            {"d": eval_date},
        )
        conn.execute(
            text("DELETE FROM model_edge_buckets WHERE date = :d"),
            {"d": eval_date},
        )


def daterange(start: datetime.date, end: datetime.date):
    d = start
    while d <= end:
        yield d
        d += datetime.timedelta(days=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD; first as_of day to replay")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD; last as_of day to replay (inclusive)")
    args = ap.parse_args()

    start = datetime.date.fromisoformat(args.start)
    end = datetime.date.fromisoformat(args.end)
    if end < start:
        print(f"refuse: end ({end}) < start ({start})")
        return 2

    # Temporarily disable the V1_CUTOVER_DATE freeze guard so eval rows can
    # be written for pre-cutover dates. Restore on exit so production runs
    # remain protected.
    original_cutover = strategy.V1_CUTOVER_DATE
    strategy.V1_CUTOVER_DATE = datetime.date(1900, 1, 1)
    evaluate_model.V1_CUTOVER_DATE = datetime.date(1900, 1, 1)

    failures: list[tuple[datetime.date, str]] = []
    try:
        for as_of in daterange(start, end):
            eval_date = as_of - datetime.timedelta(days=1)
            print(f"\n=== as_of {as_of}  (eval_date {eval_date}) ===")
            try:
                _wipe_row(eval_date)
                eval_main(as_of=as_of)
            except Exception as e:
                print(f"  FAILED: {e}")
                traceback.print_exc()
                failures.append((as_of, str(e)))
    finally:
        strategy.V1_CUTOVER_DATE = original_cutover
        evaluate_model.V1_CUTOVER_DATE = original_cutover

    print("\n" + "=" * 60)
    if failures:
        print(f"backfill done with {len(failures)} failures:")
        for d, msg in failures:
            print(f"  {d}: {msg}")
        return 1
    print(f"backfill complete: replayed {(end - start).days + 1} days")
    return 0


if __name__ == "__main__":
    sys.exit(main())
