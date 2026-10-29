"""Export the graded MLB record as two CSVs anyone can re-score.

forecasts.csv: one row per graded team forecast (row_type=forecast) plus one row per
completed game with no graded forecast (row_type=excluded, with exclusion_reason).
bets.csv: one row per graded bet from mlb.bet_ledger_v, with the market's grading rule.

backend/reproduce_performance.py reads only these files and prints every number on the
MLB performance page. Read-only against the database.

Usage:
    python -m backend.export_ledger --out DIR [--as-of YYYY-MM-DD]
"""
import argparse
import datetime
import json
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from backend.db import engine
from backend.evaluate_model import load_inputs
from backend.log import setup_logging

log = logging.getLogger(__name__)

FORECAST_COLUMNS = [
    "row_type", "game_pk", "team", "home_team", "date", "game_date", "model_version", "forecast_at", "start_time",
    "win_prob", "expected_runs", "actual_runs", "actual_win", "moneyline", "spread", "spread_odds", "total",
    "total_over_odds", "total_under_odds", "p_cover", "p_over", "p_under", "ev_flag", "run_line_ev_flag",
    "total_play", "runs_hist", "market_offers", "rewritten", "written_after_start", "exclusion_reason",
]
BET_COLUMNS = [
    "date", "game_pk", "team", "bet_type", "model_version", "forecast_at", "start_time", "stake", "american_odds",
    "decimal_odds", "totals_side", "edge", "min_edge", "won", "push", "payout", "rewritten", "written_after_start",
    "stake_column", "stake_rule", "push_rule",
]


def _json_or_blank(value):
    return "" if value is None or (isinstance(value, float) and pd.isna(value)) else json.dumps(value)


def export(out_dir: Path, today: datetime.date) -> tuple[int, int, int]:
    eval_df, _, exclusions, rewritten_days = load_inputs(today)
    eval_date = today - datetime.timedelta(days=1)
    offers = pd.read_sql(text(
        "SELECT game_pk, team, prediction_context->'market_offers' AS market_offers "
        "FROM model_outputs_season WHERE prediction_context ? 'market_offers'"), con=engine)
    bets = pd.read_sql(text(
        "SELECT l.*, r.min_edge, r.stake_column, r.stake_rule, r.push_rule "
        "FROM bet_ledger_v l JOIN grading_rules r USING (bet_type) WHERE l.date::date <= :d "
        "ORDER BY l.date, l.game_pk, l.bet_type, l.team"), con=engine, params={"d": eval_date})

    forecasts = eval_df[eval_df["game_date"] <= pd.Timestamp(eval_date)].merge(
        offers, on=["game_pk", "team"], how="left")
    forecasts = forecasts.assign(
        row_type="forecast",
        forecast_at=forecasts["prediction_updated_at"],
        start_time=forecasts["game_start_time"],
        runs_hist=forecasts["runs_hist"].map(_json_or_blank),
        market_offers=forecasts["market_offers"].map(_json_or_blank),
        rewritten=forecasts["game_date"].dt.date.isin(set(rewritten_days)),
        written_after_start=(pd.to_datetime(forecasts["prediction_updated_at"], utc=True)
                             >= pd.to_datetime(forecasts["game_start_time"], utc=True))
        .where(forecasts["prediction_updated_at"].notna()),
        exclusion_reason="",
    )
    excluded = exclusions[exclusions["game_date"] <= pd.Timestamp(eval_date)].rename(
        columns={"reason": "exclusion_reason"}).assign(row_type="excluded")
    rows = pd.concat([forecasts, excluded], ignore_index=True)[FORECAST_COLUMNS]
    rows = rows.sort_values(["game_date", "game_pk", "row_type", "team"], na_position="first")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_dir / "forecasts.csv", index=False)
    bets[BET_COLUMNS].to_csv(out_dir / "bets.csv", index=False)
    return int((rows["row_type"] == "forecast").sum()), int((rows["row_type"] == "excluded").sum()), len(bets)


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="Directory for forecasts.csv and bets.csv")
    parser.add_argument("--as-of", type=datetime.date.fromisoformat, default=datetime.date.today(),
                        help="Export the record graded through the day before this date")
    args = parser.parse_args()
    n_forecasts, n_excluded, n_bets = export(args.out, args.as_of)
    log.info(f"{n_forecasts} forecasts, {n_excluded} excluded games, {n_bets} bets -> {args.out}")
