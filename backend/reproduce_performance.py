"""Recompute every MLB performance-page number from the exported ledger CSVs alone.

Reads forecasts.csv and bets.csv written by backend/export_ledger.py; no database or
network access. It runs backend.evaluation.evaluate, the same function the nightly job
uses, so its output should match mlb.model_evaluation for the same date to the last digit.

Usage:
    python -m backend.reproduce_performance DIR [--eval-date YYYY-MM-DD]
"""
import argparse
import datetime
import json
from pathlib import Path

import pandas as pd

from backend.evaluation import clean_row, evaluate, normalize_ledger


def load(directory: Path):
    flags = {"true_values": ["True"], "false_values": ["False"]}
    rows = pd.read_csv(directory / "forecasts.csv", keep_default_na=True, **flags)
    rows["game_date"] = pd.to_datetime(rows["game_date"])
    forecasts = rows[rows["row_type"] == "forecast"].copy()
    forecasts["runs_hist"] = forecasts["runs_hist"].map(lambda v: json.loads(v) if isinstance(v, str) else None)
    forecasts["prediction_updated_at"] = pd.to_datetime(forecasts["forecast_at"], utc=True)
    forecasts["game_start_time"] = pd.to_datetime(forecasts["start_time"], utc=True)
    forecasts["actual_win"] = forecasts["actual_win"].astype(int)
    excluded = rows[rows["row_type"] == "excluded"][["game_pk", "game_date", "exclusion_reason"]].rename(
        columns={"exclusion_reason": "reason"})
    rewritten_days = sorted(set(forecasts.loc[forecasts["rewritten"].astype(bool), "game_date"].dt.date))
    ledger = normalize_ledger(pd.read_csv(directory / "bets.csv", **flags))
    return forecasts, ledger, excluded, rewritten_days


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path, help="Directory holding forecasts.csv and bets.csv")
    parser.add_argument("--eval-date", type=datetime.date.fromisoformat,
                        help="Last game date graded (default: the latest game date in the export)")
    args = parser.parse_args()
    forecasts, ledger, excluded, rewritten_days = load(args.directory)
    eval_date = args.eval_date or forecasts["game_date"].max().date()
    results, season_calibration = evaluate(forecasts, ledger, excluded, rewritten_days, eval_date)
    for window, (base_row, metrics, buckets) in results.items():
        print(json.dumps({"window": window, **clean_row({**base_row, **metrics}), "edge_buckets": buckets},
                         default=str, sort_keys=True))
    print(json.dumps({"window": "season", "model_calibration": season_calibration}, sort_keys=True))


if __name__ == "__main__":
    main()
