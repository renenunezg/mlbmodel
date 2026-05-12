"""
Evaluate model predictions against actual game results.

Computes accuracy metrics for moneyline, run line, and totals picks,
plus comprehensive regression, probabilistic, and financial metrics.
Writes results to model_evaluation (with eval_window), model_calibration,
model_feature_importance, and model_edge_buckets tables.

Usage:
    from backend.evaluate_model import main
    main()
"""

import datetime
import json
import subprocess
import pandas as pd
import numpy as np
from sqlalchemy import MetaData, text
from sqlalchemy.dialects.postgresql import insert

from backend.db import engine
from backend.kelly import american_to_decimal
from backend.metrics import (
    regression_summary,
    probabilistic_summary,
    calibration_curve,
    financial_summary,
    segment_summary,
    equity_curve_from_ledger,
    hit_rate_by_edge_bucket,
)
from backend.model import FEATURE_COLS
from backend.strategy import EV_THRESHOLDS


def _calc_run_line_pick(row):
    try:
        spread_value = float(row["spread"])
    except Exception:
        return np.nan
    margin = row.get("actual_margin", np.nan)
    if pd.isna(margin):
        return np.nan
    if spread_value < 0:
        return 1 if margin >= abs(spread_value) else 0
    elif spread_value > 0:
        return 1 if (row["actual_win"] == 1 or margin >= -spread_value) else 0
    return np.nan


def _calc_total_pick(row):
    try:
        line = float(row["total"])
    except Exception:
        return np.nan
    actual = row["game_total"]
    if pd.isna(actual):
        return np.nan
    direction = row["total_play"].strip().lower()
    if direction == "over":
        return 1 if actual > line else (np.nan if actual == line else 0)
    elif direction == "under":
        return 1 if actual < line else (np.nan if actual == line else 0)
    return np.nan


def _build_bet_ledger(eval_df):
    """Bet ledger from evaluated predictions. Re-checks EV threshold; totals deduped per game_pk."""
    rows = []
    seen_total_pks = set()

    for _, r in eval_df.iterrows():
        date = r.get("game_date", r.get("date"))

        # Moneyline
        if r.get("ev_flag") == r.get("team") and pd.notna(r.get("moneyline")):
            dec_odds = american_to_decimal(r["moneyline"])
            stake = r.get("kelly_quarter_ml")
            wp = r.get("win_prob")
            if (
                dec_odds
                and pd.notna(stake) and stake > 0
                and pd.notna(wp)
            ):
                edge = float(wp) - 1.0 / dec_odds
                if edge >= EV_THRESHOLDS["ml"]:
                    won = bool(r.get("actual_win") == 1)
                    rows.append({
                        "date": date, "bet_type": "ml", "team": r["team"],
                        "game_pk": r["game_pk"], "stake": float(stake),
                        "decimal_odds": float(dec_odds), "won": won,
                        "payout": float(stake * dec_odds) if won else 0.0,
                        "edge": edge,
                        "american_odds": float(r["moneyline"]),
                        "totals_side": None,
                    })

        # Run line
        if r.get("run_line_ev_flag") == r.get("team") and pd.notna(r.get("spread_odds")):
            dec_odds = american_to_decimal(r["spread_odds"])
            stake = r.get("kelly_quarter_rl")
            p_cover = r.get("p_cover")
            if (
                dec_odds
                and pd.notna(stake) and stake > 0
                and pd.notna(p_cover)
            ):
                edge = float(p_cover) - 1.0 / dec_odds
                if edge >= EV_THRESHOLDS["rl"]:
                    rl_correct = _calc_run_line_pick(r)
                    won = bool(rl_correct == 1) if pd.notna(rl_correct) else False
                    rows.append({
                        "date": date, "bet_type": "rl", "team": r["team"],
                        "game_pk": r["game_pk"], "stake": float(stake),
                        "decimal_odds": float(dec_odds), "won": won,
                        "payout": float(stake * dec_odds) if won else 0.0,
                        "edge": edge,
                        "american_odds": None,
                        "totals_side": None,
                    })

        # Totals (deduped per game_pk)
        if (
            r.get("total_play") in ("Over", "Under")
            and pd.notna(r.get("game_total"))
            and r["game_pk"] not in seen_total_pks
        ):
            direction = r["total_play"].strip().lower()
            odds_col = "total_over_odds" if direction == "over" else "total_under_odds"
            book_odds = r.get(odds_col)
            stake = r.get("kelly_quarter_total")
            model_p = r.get("p_over") if direction == "over" else r.get("p_under")
            if (
                pd.notna(book_odds)
                and pd.notna(stake) and stake > 0
                and pd.notna(model_p)
            ):
                dec_odds = american_to_decimal(book_odds)
                if dec_odds:
                    edge = float(model_p) - 1.0 / dec_odds
                    if edge >= EV_THRESHOLDS["totals"]:
                        seen_total_pks.add(r["game_pk"])
                        total_correct = _calc_total_pick(r)
                        won = bool(total_correct == 1) if pd.notna(total_correct) else False
                        rows.append({
                            "date": date, "bet_type": "total", "team": r["team"],
                            "game_pk": r["game_pk"], "stake": float(stake),
                            "decimal_odds": float(dec_odds), "won": won,
                            "payout": float(stake * dec_odds) if won else 0.0,
                            "edge": edge,
                            "american_odds": None,
                            "totals_side": direction,
                        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["date", "bet_type", "team", "game_pk", "stake",
                 "decimal_odds", "won", "payout", "edge",
                 "american_odds", "totals_side"]
    )


def _write_evaluation_row(eval_date, eval_window, base_row, metric_dict):
    """Upsert one row into model_evaluation."""
    metadata = MetaData()
    metadata.reflect(bind=engine)
    table = metadata.tables["model_evaluation"]

    row = {**base_row, **metric_dict, "date": eval_date, "eval_window": eval_window}
    # Clean NaN/inf for Postgres
    for k, v in row.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            row[k] = None

    update_cols = {k: v for k, v in row.items() if k not in ("date", "eval_window")}

    with engine.begin() as conn:
        conn.execute(
            insert(table)
            .values(**row)
            .on_conflict_do_update(
                index_elements=["date", "eval_window"],
                set_=update_cols,
            )
        )


def _write_calibration(eval_date, cal_bins):
    """Upsert calibration curve bins for a date."""
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


def _write_feature_importance(eval_date, importance_dict):
    """Upsert feature importance for a date."""
    if not importance_dict:
        return
    with engine.begin() as conn:
        for feature, imp in importance_dict.items():
            conn.execute(text("""
                INSERT INTO model_feature_importance (date, feature, importance)
                VALUES (:date, :feature, :importance)
                ON CONFLICT (date, feature) DO UPDATE SET importance = EXCLUDED.importance
            """), {"date": eval_date, "feature": feature, "importance": float(imp)})


def _write_edge_buckets(eval_date, eval_window, buckets):
    """Upsert edge bucket stats."""
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


def _write_experiment_run(cv_metrics, best_params):
    """Log the training run to experiment_runs."""
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_sha = None

    best_cv_mae = np.mean([m["mae"] for m in cv_metrics]) if cv_metrics else None

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO experiment_runs (git_sha, hyperparameters, best_cv_mae, feature_list)
            VALUES (:git_sha, :params, :mae, :features)
        """), {
            "git_sha": git_sha,
            "params": json.dumps(best_params) if best_params else None,
            "mae": float(best_cv_mae) if best_cv_mae else None,
            "features": FEATURE_COLS,
        })


def _compute_base_row(window_df, window_ledger):
    """Accuracy counts for a window. Bet-level counts come from the ledger
    so they always agree with the ROI / segment metrics."""
    total_correct = int((window_df["pred_win"] == window_df["actual_win"]).sum())
    total_predictions = len(window_df)
    runs_mae = abs(window_df["expected_runs"] - window_df["actual_runs"]).mean()

    if window_ledger is not None and not window_ledger.empty:
        ml_bets = window_ledger[window_ledger["bet_type"] == "ml"]
        rl_bets = window_ledger[window_ledger["bet_type"] == "rl"]
        totals_bets = window_ledger[window_ledger["bet_type"] == "total"]
    else:
        empty = pd.DataFrame(columns=["won"])
        ml_bets = rl_bets = totals_bets = empty

    ml_total = len(ml_bets); ml_correct = int(ml_bets["won"].sum()) if ml_total else 0
    rl_total = len(rl_bets); rl_correct = int(rl_bets["won"].sum()) if rl_total else 0
    totals_total = len(totals_bets); totals_correct = int(totals_bets["won"].sum()) if totals_total else 0

    ml_accuracy = ml_correct / ml_total if ml_total > 0 else np.nan
    rl_accuracy = rl_correct / rl_total if rl_total > 0 else np.nan
    totals_accuracy = totals_correct / totals_total if totals_total > 0 else np.nan

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


def main(model=None, cv_metrics=None, best_params=None):
    """Run full evaluation and write results to DB.

    Args:
        model: trained XGBoost model (for feature importance). None = skip.
        cv_metrics: list of per-fold metric dicts from train_model_cv. None = skip.
        best_params: dict of best hyperparameters. None = skip.
    """
    today = datetime.date.today()

    # --- Feature importance (depends only on trained model, not completed games) ---
    if model is not None:
        importance = dict(zip(FEATURE_COLS, model.feature_importances_))
        _write_feature_importance(today, importance)

    # --- Experiment tracking (depends only on CV metrics, not completed games) ---
    if cv_metrics:
        _write_experiment_run(cv_metrics, best_params)

    model_df = pd.read_sql(text("SELECT * FROM model_outputs_season"), con=engine)
    games_df = pd.read_sql_table("games", con=engine)

    games_df = games_df[games_df["status"] == "Final"].dropna(subset=["home_score", "away_score"])
    if games_df.empty:
        print("  No completed games to evaluate against.")
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
    eval_df = pd.merge(
        model_df,
        all_results[["game_pk", "game_date", "team", "actual_runs", "winning_team", "actual_margin"]],
        on=["game_pk", "team"],
        how="inner",
    ).dropna(subset=["actual_runs"])

    # Also merge game_total for totals evaluation
    eval_df = eval_df.merge(
        games_df[["game_pk", "game_total"]].drop_duplicates(subset=["game_pk"]),
        on="game_pk",
        how="left",
    )

    if eval_df.empty:
        print("  No predictions matched to completed games yet.")
        return

    # Win prediction accuracy
    eval_df["actual_win"] = (eval_df["team"] == eval_df["winning_team"]).astype(int)
    eval_df["pred_win"] = (eval_df["win_prob"] > 0.5).astype(int)

    accuracy = (eval_df["pred_win"] == eval_df["actual_win"]).mean()
    runs_mae = abs(eval_df["expected_runs"] - eval_df["actual_runs"]).mean()
    print(f"  Win accuracy: {accuracy:.2%} | Runs MAE: {runs_mae:.3f} | {len(eval_df)} predictions evaluated")

    # --- Build bet ledger ---
    ledger = _build_bet_ledger(eval_df)

    eval_date = pd.to_datetime(eval_df["game_date"].max()).date()

    # --- Compute metrics for multiple windows ---
    eval_df["game_date"] = pd.to_datetime(eval_df["game_date"])
    latest_date = eval_df["game_date"].max()

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

    print(f"  Evaluation written for {eval_date} (4 windows + calibration)")


if __name__ == "__main__":
    main()
