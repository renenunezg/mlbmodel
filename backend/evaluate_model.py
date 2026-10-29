"""
Evaluate model predictions against actual game results.

Computes accuracy metrics for moneyline, run line, and totals picks,
plus regression, probabilistic, financial, market-baseline and interval
metrics. Writes results to model_evaluation (one row per eval_window),
model_calibration, and model_edge_buckets.

`evaluate()` is pure: the nightly job feeds it from the database and
backend/reproduce_performance.py feeds it from the exported ledger CSVs, so
both print the same numbers.

Usage:
    python -m backend.evaluate_model [--as-of YYYY-MM-DD] [--dry-run]
"""

import argparse
import datetime
import json
import logging

import numpy as np
import pandas as pd
from sqlalchemy import MetaData, text
from sqlalchemy.dialects.postgresql import insert

from backend.db import engine
from backend.log import setup_logging
from backend.metrics import (
    bootstrap_mean_ci,
    bootstrap_roi_ci,
    calibration_curve,
    equity_curve_from_ledger,
    financial_summary,
    hit_rate_by_edge_bucket,
    log_loss_terms,
    probabilistic_summary,
    regression_summary,
    segment_summary,
)
from backend.simulation import american_to_prob
from backend.strategy import EV_THRESHOLDS, V1_CUTOVER_DATE

log = logging.getLogger(__name__)

LEDGER_COLUMNS = ["date", "bet_type", "team", "game_pk", "stake", "decimal_odds", "won", "payout", "edge",
                  "american_odds", "totals_side", "push", "model_version", "written_after_start", "rewritten"]

# Window name -> (days back from eval_date, inclusive; None = whole record) and model population.
WINDOWS = {
    "day": (0, None),
    "7d": (6, None),
    "30d": (29, None),
    "season": (None, None),
    "season_v2": (None, "v2"),
}

# ROI segments that carry a bootstrap interval, keyed by the model_evaluation column stem.
ROI_SEGMENTS = {
    "roi": lambda b: b,
    "roi_favorites": lambda b: b[(b["bet_type"] == "ml") & (b["american_odds"] < 0)],
    "roi_underdogs": lambda b: b[(b["bet_type"] == "ml") & (b["american_odds"] > 0)],
    "roi_run_line": lambda b: b[b["bet_type"] == "rl"],
    "overs_roi": lambda b: b[b["totals_side"] == "over"],
    "unders_roi": lambda b: b[b["totals_side"] == "under"],
}

EXCLUSION_REASONS = ("no_forecast", "v1_outage_hindsight_only", "forecast_dated_other_day")


def _build_bet_ledger(eval_df=None):
    """Bet ledger from the canonical SQL view `bet_ledger_v`.

    The site queries the same view, so the two sides cannot drift on
    filter or grading logic.

    `eval_df` is accepted for backwards compatibility with callers but
    ignored - the view computes its own join against `model_outputs_season`
    and `games`.
    """
    del eval_df
    ledger = pd.read_sql(text(f"SELECT {', '.join(LEDGER_COLUMNS)} FROM bet_ledger_v"), con=engine)
    return normalize_ledger(ledger)


def normalize_ledger(ledger):
    """Types shared by the database and CSV loaders."""
    if ledger.empty:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    ledger = ledger.copy()
    ledger["date"] = pd.to_datetime(ledger["date"])
    for column in ("won", "push", "rewritten"):
        ledger[column] = ledger[column].astype(bool)
    # v1 stored no publish time, so lateness is unknown there; it is not counted as late.
    ledger["written_after_start"] = ledger["written_after_start"].astype("boolean").fillna(False).astype(bool)
    for column in ("stake", "decimal_odds", "payout", "edge", "american_odds"):
        ledger[column] = pd.to_numeric(ledger[column], errors="raise")
    return ledger


def check_grading_rules():
    """Refuse to grade when the ledger's thresholds differ from the flagging thresholds."""
    with engine.connect() as conn:
        rules = dict(conn.execute(text("SELECT bet_type, min_edge FROM grading_rules")).all())
    expected = {"ml": EV_THRESHOLDS["ml"], "rl": EV_THRESHOLDS["rl"], "total": EV_THRESHOLDS["totals"]}
    if {k: float(v) for k, v in rules.items()} != expected:
        raise RuntimeError(f"mlb.grading_rules {rules} differs from backend.strategy.EV_THRESHOLDS {expected}")


def _write_evaluation_row(eval_date, eval_window, base_row, metric_dict):
    """Upsert one row into model_evaluation."""
    if eval_date < V1_CUTOVER_DATE:
        log.info(f"skip model_evaluation write for {eval_date} ({eval_window}); pre-cutover")
        return
    metadata = MetaData()
    metadata.reflect(bind=engine)
    table = metadata.tables["model_evaluation"]

    row = {**base_row, **metric_dict, "date": eval_date, "eval_window": eval_window,
           "evaluation_state": "canonical", "evaluation_started_at": datetime.datetime.now(datetime.UTC)}
    row = _clean_row(row)

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


def _clean_row(row):
    """NaN and inf become NULL for Postgres."""
    return {k: (None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v) for k, v in row.items()}


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
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM model_calibration WHERE date = :date"), {"date": eval_date})
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


def _favorite_picks(window_df):
    """One row per game for the side the model favored; games at exactly 0.5 have no pick."""
    picks = window_df.sort_values("win_prob").drop_duplicates("game_pk", keep="last")
    return picks[picks["win_prob"] > 0.5]


def _compute_base_row(window_df, window_ledger):
    """Accuracy counts for a window. Bet-level counts come from the ledger
    so they always agree with the ROI / segment metrics."""
    # Pick accuracy is per-game: take the team the model favored (higher
    # win_prob) and check whether it won. Counting both team rows would
    # double-count, and a row-level `pred_win == actual_win` mislabels the
    # loser row as correct on degenerate v1 days where both teams sit at exactly
    # 0.500 (no favorite). Excluding those non-picks keeps it order-independent
    # and consistent with the site. MAE / mean win_prob stay per-team-row.
    picks = _favorite_picks(window_df)
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


def final_games(games_df, today):
    """Completed games with scores, capped before `today` so a replay never sees later results."""
    games = games_df[games_df["status"] == "Final"].dropna(subset=["home_score", "away_score"]).copy()
    games["game_date"] = pd.to_datetime(games["game_date"])
    return games[games["game_date"] < pd.Timestamp(today)]


def prepare_eval_frame(model_df, games):
    """Join forecasts to final results: one row per graded team forecast."""
    games = games.copy()
    games["winning_team"] = np.where(games["home_score"] > games["away_score"], games["home_team"], games["away_team"])
    games["game_total"] = games["home_score"] + games["away_score"]

    # Reshape games: one row per team
    home_df = games.rename(columns={"home_team": "team", "home_score": "actual_runs"})[
        ["game_date", "game_pk", "team", "actual_runs", "winning_team"]
    ]
    away_df = games.rename(columns={"away_team": "team", "away_score": "actual_runs"})[
        ["game_date", "game_pk", "team", "actual_runs", "winning_team"]
    ]
    home_df["actual_margin"] = games["home_score"].values - games["away_score"].values
    away_df["actual_margin"] = games["away_score"].values - games["home_score"].values
    all_results = pd.concat([home_df, away_df], ignore_index=True)

    eval_df = _merge_predictions_with_results(model_df, all_results)
    eval_df = eval_df.merge(
        games[["game_pk", "game_total", "start_time"]].drop_duplicates(subset=["game_pk"])
        .rename(columns={"start_time": "game_start_time"}),
        on="game_pk", how="left",
    )
    eval_df["game_date"] = pd.to_datetime(eval_df["game_date"])
    eval_df["actual_win"] = (eval_df["team"] == eval_df["winning_team"]).astype(int)
    eval_df["pred_win"] = (eval_df["win_prob"] > 0.5).astype(int)
    return eval_df


def classify_exclusions(games, eval_df, forecast_dates):
    """Final games with no graded forecast, with the reason.

    forecast_dates holds every stored forecast (game_pk, date, source) from both
    archives, including v2 hindsight rows the unified view filters out.
    """
    missing = games[~games["game_pk"].isin(eval_df["game_pk"])][["game_pk", "game_date"]].copy()
    if missing.empty:
        return missing.assign(reason=pd.Series(dtype=str))
    stored = forecast_dates.copy()
    stored["date"] = pd.to_datetime(stored["date"])
    stored = stored.merge(missing, on="game_pk")

    def reason(game):
        rows = stored[stored["game_pk"] == game.game_pk]
        if rows.empty:
            return "no_forecast"
        on_day = rows[rows["date"] == game.game_date]
        if on_day.empty:
            return "forecast_dated_other_day"
        if game.game_date.date() < V1_CUTOVER_DATE and set(on_day["source"]) == {"v2"}:
            return "v1_outage_hindsight_only"
        raise ValueError(f"Game {game.game_pk} has a same-day forecast but no graded row")

    missing["reason"] = [reason(g) for g in missing.itertuples()]
    return missing


def _market_comparison(window_df):
    """Model and de-vigged market scores on games with both sides' stored prices."""
    rows = window_df.assign(home=window_df["team"] == window_df["home_team"])
    home = rows[rows["home"]].set_index("game_pk")
    away = rows[~rows["home"]].set_index("game_pk")
    games = home[["win_prob", "actual_win", "moneyline"]].join(
        away[["win_prob", "moneyline"]], rsuffix="_away", how="inner")
    priced = games["moneyline"].abs().ge(100) & games["moneyline_away"].abs().ge(100)
    games = games[priced]
    if games.empty:
        return {"paired_games": 0}, None
    implied_home = games["moneyline"].map(american_to_prob)
    implied_away = games["moneyline_away"].map(american_to_prob)
    market = implied_home / (implied_home + implied_away)
    y = games["actual_win"].astype(float)
    # Model error averages both team rows, matching brier_score on the full window.
    model_sq = ((games["win_prob"] - y) ** 2 + (games["win_prob_away"] - (1 - y)) ** 2) / 2
    model_ll = (log_loss_terms(games["win_prob"], y) + log_loss_terms(games["win_prob_away"], 1 - y)) / 2
    market_sq = (market - y) ** 2
    diff = (model_sq - market_sq).to_numpy()
    low, high = bootstrap_mean_ci(diff)
    return {
        "paired_games": int(len(games)),
        "paired_brier": float(model_sq.mean()),
        "paired_log_loss": float(model_ll.mean()),
        "market_brier": float(market_sq.mean()),
        "market_log_loss": float(log_loss_terms(market, y).mean()),
        "brier_vs_market_ci95_low": low,
        "brier_vs_market_ci95_high": high,
    }, diff


def _interval_columns(window_df, window_ledger):
    out = {}
    per_game_sq = (window_df["win_prob"] - window_df["actual_win"]) ** 2
    out["brier_ci95_low"], out["brier_ci95_high"] = bootstrap_mean_ci(
        per_game_sq.groupby(window_df["game_pk"]).mean().to_numpy())
    picks = _favorite_picks(window_df)
    out["total_accuracy_ci95_low"], out["total_accuracy_ci95_high"] = bootstrap_mean_ci(
        (picks["actual_win"] == 1).astype(float).to_numpy())
    for stem, select in ROI_SEGMENTS.items():
        segment = select(window_ledger) if not window_ledger.empty else window_ledger
        out[f"{stem}_ci95_low"], out[f"{stem}_ci95_high"] = bootstrap_roi_ci(segment)
    return out


def _window_slices(eval_df, ledger, exclusions, eval_date, days_back, model):
    latest = pd.Timestamp(eval_date)
    in_frame = eval_df["game_date"] <= latest
    in_ledger = ledger["date"] <= latest
    in_excluded = exclusions["game_date"] <= latest
    in_population = ledger["date"] <= latest
    if days_back is not None:
        start = latest - pd.Timedelta(days=days_back)
        in_frame &= eval_df["game_date"] >= start
        in_ledger &= ledger["date"] >= start
        in_excluded &= exclusions["game_date"] >= start
    if model is not None:
        in_frame &= eval_df["model_version"] == model
        in_ledger &= ledger["model_version"] == model
        in_population &= ledger["model_version"] == model
        in_excluded &= exclusions["game_date"] >= pd.Timestamp(V1_CUTOVER_DATE)
    return eval_df[in_frame], ledger[in_ledger], exclusions[in_excluded], ledger[in_population]


def evaluate(eval_df, ledger, exclusions, rewritten_days, eval_date):
    """Every model_evaluation row, edge bucket set and the season calibration for one eval_date.

    Returns ({window: (base_row, metrics, buckets)}, season_calibration_bins).
    """
    results = {}
    for window_name, (days_back, model) in WINDOWS.items():
        window_df, window_ledger, window_excluded, population_ledger = _window_slices(
            eval_df, ledger, exclusions, eval_date, days_back, model)
        if window_df.empty:
            continue

        base_row = _compute_base_row(window_df, window_ledger)

        y_true = window_df["actual_runs"].values.astype(float)
        y_pred = window_df["expected_runs"].values.astype(float)
        probs = window_df["win_prob"].values.astype(float)
        outcomes = window_df["actual_win"].values.astype(float)

        reg = regression_summary(y_true, y_pred)
        prob = probabilistic_summary(probs, outcomes, histograms=window_df["runs_hist"].tolist(), actual_runs=y_true)
        fin = financial_summary(window_ledger)
        seg = segment_summary(window_ledger)
        market, _ = _market_comparison(window_df)

        # Equity is cumulative from the first graded bet of this window's model population.
        eq = equity_curve_from_ledger(population_ledger)
        equity_end = float(eq["equity"].iloc[-1]) if not eq.empty else 1.0

        window_dates = set(window_df["game_date"].dt.date)
        late_games = window_df.loc[
            window_df["prediction_updated_at"].notna()
            & (pd.to_datetime(window_df["prediction_updated_at"], utc=True)
               >= pd.to_datetime(window_df["game_start_time"], utc=True)),
            "game_pk"].nunique()
        excluded = window_excluded["reason"].value_counts()

        metrics = {
            **reg,
            **prob,
            **fin,
            **seg,
            **market,
            **_interval_columns(window_df, window_ledger),
            "equity_end_units": round(equity_end, 4),
            "scored_games": int(window_df["game_pk"].nunique()),
            "n_bets": int(len(window_ledger)),
            "calibration_bins": calibration_curve(probs, outcomes),
            "predictions_rewritten": bool(window_dates & set(rewritten_days)),
            "rewritten_bets": int(window_ledger["rewritten"].sum()) if not window_ledger.empty else 0,
            "late_forecast_games": int(late_games),
            "late_forecast_bets": int(window_ledger["written_after_start"].sum()) if not window_ledger.empty else 0,
            "excluded_games": {reason: int(excluded.get(reason, 0)) for reason in EXCLUSION_REASONS},
        }
        # Remove residual_ sub-keys from the DB row (they don't have columns)
        metrics = {k: v for k, v in metrics.items() if not k.startswith("residual_")}

        buckets = hit_rate_by_edge_bucket(window_ledger) if not window_ledger.empty else []
        results[window_name] = (base_row, metrics, buckets)

    season_rows = _window_slices(eval_df, ledger, exclusions, eval_date, None, None)[0]
    cal_bins = calibration_curve(season_rows["win_prob"].values, season_rows["actual_win"].values)
    return results, cal_bins


def load_inputs(today):
    """Everything evaluate() needs, read from the database."""
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
    games = final_games(games_df, today)
    eval_df = prepare_eval_frame(model_df, games)
    exclusions = classify_exclusions(games, eval_df, forecast_dates)
    return eval_df, _build_bet_ledger(), exclusions, rewritten_days


def main(as_of: datetime.date | None = None, dry_run: bool = False):
    """Run full evaluation and write results to DB.

    Args:
        as_of: pretend "today" is this date. eval_date = as_of - 1. Used by
            the historical backfill script to replay eval rows for past days.
        dry_run: compute every row and print it instead of writing.
    """
    today = as_of if as_of is not None else datetime.date.today()
    check_grading_rules()
    eval_df, ledger, exclusions, rewritten_days = load_inputs(today)

    if eval_df.empty:
        log.info("No predictions matched to completed games yet.")
        return

    # eval_date is always yesterday so the same date can never be re-targeted
    # from a different angle (e.g. a manual rerun during the day shifting max).
    eval_date = today - datetime.timedelta(days=1)
    results, cal_bins = evaluate(eval_df, ledger, exclusions, rewritten_days, eval_date)

    if dry_run:
        for window_name, (base_row, metrics, buckets) in results.items():
            print(json.dumps({"window": window_name, **_clean_row({**base_row, **metrics}), "edge_buckets": buckets},
                             default=str, sort_keys=True))
        return results

    for window_name, (base_row, metrics, buckets) in results.items():
        _write_evaluation_row(eval_date, window_name, base_row, metrics)
        _write_edge_buckets(eval_date, window_name, buckets)
    _write_calibration(eval_date, cal_bins)

    log.info(f"Evaluation written for {eval_date} ({len(results)} windows + calibration)")


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--as-of", type=datetime.date.fromisoformat, help="Evaluate as if today were this date")
    parser.add_argument("--dry-run", action="store_true", help="Print the rows instead of writing them")
    args = parser.parse_args()
    main(as_of=args.as_of, dry_run=args.dry_run)
