"""Compare variance candidates using frozen pregame forecasts on the same games.

The former in-sample 2025 simulator sweep was not a forecasting acceptance gate.
Generate independent pregame exports for each candidate and compare them here.
No variance setting is promoted from hindsight lineups or actual relief usage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from v2.market_model.acceptance import compare_forecasts, load_export


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path, nargs="+")
    args = parser.parse_args()
    baseline = load_export(args.baseline)
    reports = {str(path): compare_forecasts(load_export(path), baseline) for path in args.candidates}
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
