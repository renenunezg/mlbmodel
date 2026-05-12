"""Morning orchestrator for v2 daily scoring pipeline.

Sequence:
  1. Refresh schedule + scores
  2. Refresh bullpen_daily + pitcher_workload
  3. Fetch odds
  4. Score today's games (v2 Bayesian sim)
  5. Run sanity checks

Usage:
    python -m v2.pipeline.daily_run [--date YYYY-MM-DD] [--n-sims N]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date

from backend.data.bullpen_daily import update_bullpen_daily
from backend.data.odds_api import fetch_odds
from pipeline import fetch_and_load_odds, update_scores_and_schedule
from v2.pipeline.score_games import score
from v2.pipeline.verify import run_checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=str(date.today()))
    ap.add_argument("--n-sims", type=int, default=10000)
    args = ap.parse_args()

    print(f"[daily_run] date={args.date}")

    print("[daily_run] step 1: schedule + scores")
    update_scores_and_schedule()

    print("[daily_run] step 2: bullpen_daily + pitcher_workload")
    update_bullpen_daily()

    print("[daily_run] step 3: odds")
    fetch_and_load_odds()

    print(f"[daily_run] step 4: scoring {args.date}")
    rows = score(args.date, n_sims=args.n_sims, write=True)
    if rows.empty:
        print(f"[daily_run] no games on {args.date}, exiting")
        sys.exit(0)

    print("[daily_run] step 5: verify")
    ok = run_checks(args.date)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
