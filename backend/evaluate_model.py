"""
Evaluate model predictions against actual game results.

Computes accuracy metrics for moneyline, run line, and totals picks,
plus regression, probabilistic, financial, market-baseline and interval
metrics. Writes results to model_evaluation (one row per eval_window),
model_calibration (season and season_v2), and model_edge_buckets.

The computation lives in backend/evaluation.py; this module reads its inputs
from the database and writes the rows.

Usage:
    python -m backend.evaluate_model [--as-of YYYY-MM-DD] [--backfill-from YYYY-MM-DD]
                                     [--windows season_v2 ...] [--dry-run]
"""

import argparse
import datetime
import json
import logging

import pandas as pd
from sqlalchemy import MetaData, text
from sqlalchemy.dialects.postgresql import insert

from backend.db import engine
from backend.evaluation import (
    CLOSING_LINE_COLUMNS,
    LEDGER_COLUMNS,
    WINDOWS,
    classify_exclusions,
    clean_row,
    evaluate,
    final_games,
    normalize_ledger,
    prepare_eval_frame,
)
from backend.log import setup_logging
from backend.strategy import EV_THRESHOLDS, V1_CUTOVER_DATE

log = logging.getLogger(__name__)


def _build_bet_ledger():
    """Bet ledger from the canonical SQL view `bet_ledger_v`.

    The site queries the same view, so the two sides cannot drift on
    filter or grading logic.
    """
    ledger = pd.read_sql(text(f"SELECT {', '.join(LEDGER_COLUMNS)} FROM bet_ledger_v"), con=engine)
    return normalize_ledger(ledger)


def check_grading_rules():
    """Refuse to grade when the ledger's thresholds differ from the flagging thresholds."""
    with engine.connect() as conn:
        rules = dict(conn.execute(text("SELECT bet_type, min_edge FROM grading_rules")).all())
    expected = {"ml": EV_THRESHOLDS["ml"], "rl": EV_THRESHOLDS["rl"], "total": EV_THRESHOLDS["totals"]}
    if {k: float(v) for k, v in rules.items()} != expected:
        raise RuntimeError(f"mlb.grading_rules {rules} differs from backend.strategy.EV_THRESHOLDS {expected}")


def _evaluation_table():
    """model_evaluation as reflected now; reflect once per run, not per row."""
    metadata = MetaData()
    metadata.reflect(bind=engine, only=["model_evaluation"])
    return metadata.tables["model_evaluation"]


def _write_evaluation_row(eval_date, eval_window, base_row, metric_dict, table=None):
    """Upsert one row into model_evaluation."""
    if eval_date < V1_CUTOVER_DATE:
        log.info(f"skip model_evaluation write for {eval_date} ({eval_window}); pre-cutover")
        return
    table = table if table is not None else _evaluation_table()

    row = {**base_row, **metric_dict, "date": eval_date, "eval_window": eval_window,
           "evaluation_state": "canonical", "evaluation_started_at": datetime.datetime.now(datetime.UTC)}
    row = clean_row(row)

    # Skip the write entirely if the headline metrics are all NULL. This
    # happens when nightly-eval fires before any of the target date's games
    # have graded - inserting a placeholder row would leak NULLs into the
    # site's "latest" lookup.
    if all(row.get(k) is None for k in ("brier_score", "mae", "log_loss")):
        log.info(f"no graded data for {eval_date} ({eval_window}); skipping write")
        return

    update_cols = _evaluation_update_values(row)

    with engine.begin() as conn:
        conn.execute(
            insert(table)
            .values(**row)
            .on_conflict_do_update(
                index_elements=["date", "eval_window"],
                set_=update_cols,
            )
        )


def _evaluation_update_values(row):
    """Return the cells explicitly computed by the current evaluation run.

    An omitted key means a partial rerun did not compute that metric and must
    preserve the stored value. An explicit ``None`` means the metric was
    recomputed and is now unavailable, such as accuracy when the bet count is
    zero, so the old value must be cleared.
    """
    return {
        key: value
        for key, value in row.items()
        if key not in ("date", "eval_window")
    }


def _write_calibration(eval_date, eval_window, cal_bins):
    """Replace one window's calibration curve bins for a date."""
    if eval_date < V1_CUTOVER_DATE:
        log.info(f"skip model_calibration write for {eval_date} ({eval_window}); pre-cutover")
        return
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM model_calibration WHERE date = :date AND eval_window = :window"),
                     {"date": eval_date, "window": eval_window})
        for b in cal_bins:
            conn.execute(text("""
                INSERT INTO model_calibration (date, eval_window, bin_mid, predicted_mean, observed_rate, count)
                VALUES (:date, :eval_window, :bin_mid, :predicted_mean, :observed_rate, :count)
                ON CONFLICT (date, eval_window, bin_mid) DO UPDATE SET
                    predicted_mean = EXCLUDED.predicted_mean,
                    observed_rate = EXCLUDED.observed_rate,
                    count = EXCLUDED.count
            """), {"date": eval_date, "eval_window": eval_window, **b})


def _write_edge_buckets(eval_date, eval_window, buckets):
    """Upsert edge bucket stats."""
    if eval_date < V1_CUTOVER_DATE:
        log.info(f"skip model_edge_buckets write for {eval_date} ({eval_window}); pre-cutover")
        return
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM model_edge_buckets WHERE date = :date AND eval_window = :window"),
                     {"date": eval_date, "window": eval_window})
        for b in buckets:
            conn.execute(text("""
                INSERT INTO model_edge_buckets (date, eval_window, bucket_label, n_bets, hit_rate, roi)
                VALUES (:date, :eval_window, :bucket_label, :n_bets, :hit_rate, :roi)
                ON CONFLICT (date, eval_window, bucket_label) DO UPDATE SET
                    n_bets = EXCLUDED.n_bets,
                    hit_rate = EXCLUDED.hit_rate,
                    roi = EXCLUDED.roi
            """), {"date": eval_date, "eval_window": eval_window, **b})


def load_inputs(today):
    """Forecasts joined to results, exclusions, rewritten days and closing lines, read from the database.

    The bet ledger is read separately (_build_bet_ledger) so callers with their own ledger query skip it.
    """
    # Continuous v1+v2 track record. The unified view routes pre-cutover
    # dates to the frozen v1 archive (real v1 live picks) and post-cutover
    # to the live v2 table. v2's hindsight backfill rows for pre-cutover
    # dates are filtered out by the view, so they never enter the eval.
    model_df = pd.read_sql(text("SELECT u.*, s.runs_hist FROM model_outputs_season_unified u "
        "LEFT JOIN model_outputs_season s ON s.game_pk = u.game_pk AND s.team = u.team "
        "AND s.date::date = u.date::date AND u.model_version = 'v2'"), con=engine)
    games_df = pd.read_sql_table("games", con=engine)
    forecast_dates = pd.read_sql(text(
        "SELECT DISTINCT game_pk, date::date AS date, 'v2' AS source FROM model_outputs_season "
        "UNION SELECT DISTINCT game_pk, date::date, 'v1' FROM model_outputs_season_v1_archive"), con=engine)
    with engine.connect() as conn:
        rewritten_days = conn.execute(text("SELECT date FROM rewritten_forecast_days")).scalars().all()
    closing_lines = pd.read_sql(text(f"SELECT {', '.join(CLOSING_LINE_COLUMNS)} FROM closing_lines"), con=engine)
    games = final_games(games_df, today)
    eval_df = prepare_eval_frame(model_df, games)
    exclusions = classify_exclusions(games, eval_df, forecast_dates)
    return eval_df, exclusions, rewritten_days, closing_lines


def main(as_of: datetime.date | None = None, dry_run: bool = False,
         backfill_from: datetime.date | None = None, windows: tuple[str, ...] = tuple(WINDOWS)):
    """Run full evaluation and write results to DB.

    Args:
        as_of: pretend "today" is this date. eval_date = as_of - 1. Used to
            replay eval rows for past days.
        dry_run: compute every row and print it instead of writing.
        backfill_from: evaluate every eval_date from this date through as_of - 1,
            reading the inputs once. Each date sees only games and bets on or
            before it, as a nightly run on the following morning would.
        windows: the WINDOWS keys to compute and write; others are left as stored.
    """
    today = as_of if as_of is not None else datetime.date.today()
    check_grading_rules()
    eval_df, exclusions, rewritten_days, closing_lines = load_inputs(today)
    ledger = _build_bet_ledger()

    if eval_df.empty:
        log.info("No predictions matched to completed games yet.")
        return

    # eval_date is always yesterday so the same date can never be re-targeted
    # from a different angle (e.g. a manual rerun during the day shifting max).
    last = today - datetime.timedelta(days=1)
    first = backfill_from if backfill_from is not None else last
    if first > last:
        raise ValueError(f"--backfill-from {first} is after the last eval date {last}")
    table = None if dry_run else _evaluation_table()
    eval_date = first
    while eval_date <= last:
        results, calibration = evaluate(eval_df, ledger, exclusions, rewritten_days, eval_date, closing_lines,
                                        windows)
        if dry_run:
            for window_name, (base_row, metrics, buckets) in results.items():
                print(json.dumps({"date": eval_date, "window": window_name, **clean_row({**base_row, **metrics}),
                                  "edge_buckets": buckets, "calibration": calibration.get(window_name)},
                                 default=str, sort_keys=True))
        else:
            for window_name, (base_row, metrics, buckets) in results.items():
                _write_evaluation_row(eval_date, window_name, base_row, metrics, table)
                _write_edge_buckets(eval_date, window_name, buckets)
            for window_name, cal_bins in calibration.items():
                _write_calibration(eval_date, window_name, cal_bins)
            log.info(f"Evaluation written for {eval_date} ({', '.join(results)} + calibration)")
        eval_date += datetime.timedelta(days=1)


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--as-of", type=datetime.date.fromisoformat, help="Evaluate as if today were this date")
    parser.add_argument("--backfill-from", type=datetime.date.fromisoformat,
                        help="Evaluate every date from this one through the day before --as-of")
    parser.add_argument("--windows", nargs="+", choices=list(WINDOWS), default=list(WINDOWS),
                        help="Windows to compute and write (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Print the rows instead of writing them")
    args = parser.parse_args()
    main(as_of=args.as_of, dry_run=args.dry_run, backfill_from=args.backfill_from, windows=tuple(args.windows))
