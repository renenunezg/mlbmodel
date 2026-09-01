"""Daily orchestrator for the v2 scoring pipeline.

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
import logging
import sys
from datetime import date

from backend.data.bullpen_daily import update_bullpen_daily
from backend.data.weather import update_weather_for_date
from backend.log import setup_logging
from pipeline import fetch_and_load_odds, update_scores_and_schedule
from v2.pipeline.score_games import score
from v2.pipeline.verify import run_checks

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=str(date.today()))
    ap.add_argument("--n-sims", type=int, default=10000)
    ap.add_argument("--optional-odds-refresh", action="store_true")
    args = ap.parse_args()
    target_date = date.fromisoformat(args.date)
    log.info(f"date={args.date}")

    log.info("step 1: schedule + scores")
    update_scores_and_schedule()

    log.info("step 2: bullpen_daily + pitcher_workload")
    update_bullpen_daily()

    log.info("step 3: odds")
    fetch_and_load_odds(target_date, optional=args.optional_odds_refresh)

    log.info("step 3b: weather")
    update_weather_for_date(target_date)

    log.info(f"step 4: scoring {args.date}")
    rows = score(args.date, n_sims=args.n_sims, write=True)
    if rows.empty:
        log.info(f"no games on {args.date}, exiting")
        sys.exit(0)

    log.info("step 5: verify")
    ok = run_checks(args.date)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    setup_logging()
    main()
