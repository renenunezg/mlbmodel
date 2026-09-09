"""
Evaluate model predictions against actual game results.

Computes accuracy metrics for moneyline, run line, and totals picks,
plus regression, probabilistic, and financial metrics.
Writes results to model_evaluation (with eval_window), model_calibration,
and model_edge_buckets.

Usage:
    from backend.evaluate_model import main
    main()
"""

import datetime
import logging

import numpy as np
import pandas as pd
from sqlalchemy import MetaData, text
from sqlalchemy.dialects.postgresql import insert

from backend.db import engine
from backend.log import setup_logging
from backend.metrics import (
    calibration_curve,
    equity_curve_from_ledger,
    financial_summary,
    hit_rate_by_edge_bucket,
    probabilistic_summary,
    regression_summary,
    segment_summary,
)
from backend.strategy import V1_CUTOVER_DATE

log = logging.getLogger(__name__)


def _build_bet_ledger(eval_df=None):
    """Bet ledger from the canonical SQL view `bet_ledger_v`.

    The site queries the same view, so the two sides cannot drift on
    filter or grading logic.

    `eval_df` is accepted for backwards compatibility with callers but
    ignored - the view computes its own join against `model_outputs_season`
    and `games`.
    """
    del eval_df
    ledger = pd.read_sql(
        text(
            "SELECT date, bet_type, team, game_pk, stake, decimal_odds, "
            "won, payout, edge, american_odds, totals_side, push "
            "FROM bet_ledger_v"
        ),
        con=engine,
    )
    if ledger.empty:
        return pd.DataFrame(
            columns=["date", "bet_type", "team", "game_pk", "stake",
                     "decimal_odds", "won", "payout", "edge",
                     "american_odds", "totals_side", "push"]
        )
    ledger["date"] = pd.to_datetime(ledger["date"])
    ledger["won"] = ledger["won"].astype(bool)
    ledger["push"] = ledger["push"].astype(bool)
    return ledger


def _write_evaluation_row(eval_date, eval_window, base_row, metric_dict):
    """Upsert one row into model_evaluation."""
    if eval_date < V1_CUTOVER_DATE:
        log.info(f"skip model_evaluation write for {eval_date} ({eval_window}); pre-cutover")
        return
    metadata = MetaData()
    metadata.reflect(bind=engine)
    table = metadata.tables["model_evaluation"]

    row = {**base_row, **metric_dict, "date": eval_date, "eval_window": eval_window}
    # Clean NaN/inf for Postgres
    for k, v in row.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            row[k] = None

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


def _write_calibration(eval_date, cal_bins):
    """Upsert calibration curve bins for a date."""
    if eval_date < V1_CUTOVER_DATE:
        log.info(f"skip model_calibration write for {eval_date}; pre-cutover")
        return
    if not cal_bins:
        return
    with engine.begin() as conn:
        for b in cal_bins:
            conn.execute(text("""
                INSERT INTO model_calibration (date, bin_mid, predicted_mean, observed_rate, count)
                VALUES (:date, :bin_mid, :predicted_mean, :observed_rate, :count)
                ON CONFLICT (date, bin_mid) DO UPDATE SET
                    predicted_mean = EXCLUDED.predicted_mean,
                    observed_rate = EXCLUDED.observed_rate,
                    count = EXCLUDED.count
            """), {"date": eval_date, **b})



def _write_edge_buckets(eval_date, eval_window, buckets):
    """Upsert edge bucket stats."""
    if eval_date < V1_CUTOVER_DATE:
        log.info(f"skip model_edge_buckets write for {eval_date} ({eval_window}); pre-cutover")
        return
    if not buckets:
        return
    with engine.begin() as conn:
        for b in buckets:
            conn.execute(text("""
                INSERT INTO model_edge_buckets (date, eval_window, bucket_label, n_bets, hit_rate, roi)
                VALUES (:date, :eval_window, :bucket_label, :n_bets, :hit_rate, :roi)
                ON CONFLICT (date, eval_window, bucket_label) DO UPDATE SET
                    n_bets = EXCLUDED.n_bets,
                    hit_rate = EXCLUDED.hit_rate,
                    roi = EXCLUDED.roi
            """), {"date": eval_date, "eval_window": eval_window, **b})



def _bet_record(bets):
    """(n_bets, wins, accuracy) for a ledger slice; pushes count as bets but
    are excluded from the accuracy denominator."""
    n = len(bets)
    if n == 0:
        return 0, 0, np.nan
    wins = int(bets["won"].sum())
    pushes = int(bets["push"].sum()) if "push" in bets.columns else 0
    decided = n - pushes
    return n, wins, (wins / decided if decided > 0 else np.nan)


def _compute_base_row(window_df, window_ledger):
    """Accuracy counts for a window. Bet-level counts come from the ledger
    so they always agree with the ROI / segment metrics."""
    # Pick accuracy is per-game: take the team the model favored (higher
    # win_prob) and check whether it won. Counting both team rows would
    # double-count, and a row-level `pred_win == actual_win` mislabels the
    # loser row as correct on degenerate v1 days where both teams sit at exactly
    # 0.500 (no favorite). Excluding those non-picks keeps it order-independent
    # and consistent with the site. MAE / mean win_prob stay per-team-row.
    picks = window_df.sort_values("win_prob").drop_duplicates("game_pk", keep="last")
    picks = picks[picks["win_prob"] > 0.5]
    total_correct = int((picks["actual_win"] == 1).sum())
    total_predictions = len(picks)
    runs_mae = abs(window_df["expected_runs"] - window_df["actual_runs"]).mean()

    if window_ledger is not None and not window_ledger.empty:
        ml_bets = window_ledger[window_ledger["bet_type"] == "ml"]
        rl_bets = window_ledger[window_ledger["bet_type"] == "rl"]
        totals_bets = window_ledger[window_ledger["bet_type"] == "total"]
    else:
        empty = pd.DataFrame(columns=["won", "push"])
        ml_bets = rl_bets = totals_bets = empty

    # A push is a bet (it counts toward predictions) but neither a win nor a
    # loss, so accuracy is wins over decided bets.
    ml_total, ml_correct, ml_accuracy = _bet_record(ml_bets)
    rl_total, rl_correct, rl_accuracy = _bet_record(rl_bets)
    totals_total, totals_correct, totals_accuracy = _bet_record(totals_bets)

    return {
        "total_correct": total_correct,
        "total_predictions": total_predictions,
        "total_accuracy": round(total_correct / total_predictions, 4) if total_predictions > 0 else None,
        "ml_correct": ml_correct,
        "ml_predictions": ml_total,
        "ml_accuracy": round(float(ml_accuracy), 4) if pd.notna(ml_accuracy) else None,
        "run_line_correct": rl_correct,
        "run_line_predictions": rl_total,
        "run_line_accuracy": round(float(rl_accuracy), 4) if pd.notna(rl_accuracy) else None,
        "totals_correct": totals_correct,
        "totals_predictions": totals_total,
        "totals_accuracy": round(float(totals_accuracy), 4) if pd.notna(totals_accuracy) else None,
        "average_total_diff": round(float(runs_mae), 4),
        "average_win_prob": round(float(window_df["win_prob"].mean()), 4),
    }


def _merge_predictions_with_results(model_df, all_results):
    predictions = model_df.copy()
    results = all_results.copy()
    predictions["_prediction_date"] = pd.to_datetime(predictions["date"]).dt.date
    results["_result_date"] = pd.to_datetime(results["game_date"]).dt.date

    merged = pd.merge(
        predictions,
        results[[
            "game_pk", "game_date", "team", "actual_runs", "winning_team",
            "actual_margin", "_result_date",
        ]],
        on=["game_pk", "team"],
        how="inner",
    )
    merged = merged[merged["_prediction_date"] == merged["_result_date"]]
    return (
        merged
        .drop(columns=["_prediction_date", "_result_date"])
        .dropna(subset=["actual_runs"])
    )


def main(as_of: datetime.date | None = None):
    """Run full evaluation and write results to DB.

    Args:
        as_of: pretend "today" is this date. eval_date = as_of - 1. Used by
            the historical backfill script to replay eval rows for past days.
    """
    today = as_of if as_of is not None else datetime.date.today()

    # Continuous v1+v2 track record. The unified view routes pre-cutover
    # dates to the frozen v1 archive (real v1 live picks) and post-cutover
    # to the live v2 table. v2's hindsight backfill rows for pre-cutover
    # dates are filtered out by the view, so they never enter the eval.
    model_df = pd.read_sql(text("SELECT * FROM model_outputs_season_unified"), con=engine)
    games_df = pd.read_sql_table("games", con=engine)

    games_df = games_df[games_df["status"] == "Final"].dropna(subset=["home_score", "away_score"])
    # Cap evaluation to games completed by `today`. Live runs are a no-op
    # (no future games are Final yet); historical replays via as_of need this
    # so future games don't leak into a past eval row's "season" window.
    games_df["game_date"] = pd.to_datetime(games_df["game_date"])
    games_df = games_df[games_df["game_date"] < pd.Timestamp(today)]
    if games_df.empty:
        log.info("No completed games to evaluate against.")
        return

    games_df["winning_team"] = np.where(
        games_df["home_score"] > games_df["away_score"],
        games_df["home_team"],
        games_df["away_team"],
    )
    games_df["game_total"] = games_df["home_score"] + games_df["away_score"]

    # Reshape games: one row per team
    home_df = games_df.rename(columns={"home_team": "team", "home_score": "actual_runs"})[
        ["game_date", "game_pk", "team", "actual_runs", "winning_team"]
    ]
    away_df = games_df.rename(columns={"away_team": "team", "away_score": "actual_runs"})[
        ["game_date", "game_pk", "team", "actual_runs", "winning_team"]
    ]
    home_df["actual_margin"] = games_df["home_score"].values - games_df["away_score"].values
    away_df["actual_margin"] = games_df["away_score"].values - games_df["home_score"].values
    all_results = pd.concat([home_df, away_df], ignore_index=True)

    # Merge predictions with results
    eval_df = _merge_predictions_with_results(model_df, all_results)

    # Also merge game_total for totals evaluation
    eval_df = eval_df.merge(
        games_df[["game_pk", "game_total"]].drop_duplicates(subset=["game_pk"]),
        on="game_pk",
        how="left",
    )

    if eval_df.empty:
        log.info("No predictions matched to completed games yet.")
        return

    # Win prediction accuracy
    eval_df["actual_win"] = (eval_df["team"] == eval_df["winning_team"]).astype(int)
    eval_df["pred_win"] = (eval_df["win_prob"] > 0.5).astype(int)

    accuracy = (eval_df["pred_win"] == eval_df["actual_win"]).mean()
    runs_mae = abs(eval_df["expected_runs"] - eval_df["actual_runs"]).mean()
    log.info(f"Win accuracy: {accuracy:.2%} | Runs MAE: {runs_mae:.3f} | {len(eval_df)} predictions evaluated")

    # --- Build bet ledger ---
    ledger = _build_bet_ledger(eval_df)

    # eval_date is always yesterday so the same date can never be re-targeted
    # from a different angle (e.g. a manual rerun during the day shifting max).
    eval_date = today - datetime.timedelta(days=1)

    eval_df["game_date"] = pd.to_datetime(eval_df["game_date"])
    latest_date = pd.Timestamp(eval_date)

    windows = {
        "day": eval_df[eval_df["game_date"] == latest_date],
        "7d": eval_df[eval_df["game_date"] >= latest_date - pd.Timedelta(days=7)],
        "30d": eval_df[eval_df["game_date"] >= latest_date - pd.Timedelta(days=30)],
        "season": eval_df,
    }

    if not ledger.empty:
        ledger["date"] = pd.to_datetime(ledger["date"])

    for window_name, window_df in windows.items():
        if window_df.empty:
            continue

        # Build window ledger first; base_row counts are derived from it.
        if not ledger.empty:
            window_dates = set(window_df["game_date"].dt.date)
            window_ledger = ledger[ledger["date"].dt.date.isin(window_dates)]
        else:
            window_ledger = pd.DataFrame(columns=ledger.columns if not ledger.empty else [])

        base_row = _compute_base_row(window_df, window_ledger)

        y_true = window_df["actual_runs"].values.astype(float)
        y_pred = window_df["expected_runs"].values.astype(float)
        probs = window_df["win_prob"].values.astype(float)
        outcomes = window_df["actual_win"].values.astype(float)

        reg = regression_summary(y_true, y_pred)
        prob = probabilistic_summary(probs, outcomes, y_pred, y_true)

        # Financial metrics for this window (window_ledger built above)
        fin = financial_summary(window_ledger) if not window_ledger.empty else {}
        seg = segment_summary(window_ledger)

        # Equity end
        eq = equity_curve_from_ledger(window_ledger)
        equity_end = float(eq["equity"].iloc[-1]) if not eq.empty else 1.0

        metrics = {
            **reg,
            **prob,
            **fin,
            **seg,
            "equity_end_units": round(equity_end, 4),
        }
        # Remove residual_ sub-keys from the DB row (they don't have columns)
        metrics = {k: v for k, v in metrics.items() if not k.startswith("residual_")}

        _write_evaluation_row(eval_date, window_name, base_row, metrics)

        # Edge buckets per window
        if not window_ledger.empty:
            buckets = hit_rate_by_edge_bucket(window_ledger)
            _write_edge_buckets(eval_date, window_name, buckets)

    # --- Calibration curve (season-wide, latest date) ---
    season_df = windows["season"]
    cal_bins = calibration_curve(
        season_df["win_prob"].values,
        season_df["actual_win"].values,
    )
    _write_calibration(eval_date, cal_bins)

    log.info(f"Evaluation written for {eval_date} (4 windows + calibration)")


if __name__ == "__main__":
    setup_logging()
    main()
