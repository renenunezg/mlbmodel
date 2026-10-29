"""Pure model evaluation: graded populations, windows and every published metric.

No database access. backend/evaluate_model.py feeds it from Supabase each night and
backend/reproduce_performance.py feeds it from the exported ledger CSVs, so both
compute the same numbers from the same rows.
"""
import numpy as np
import pandas as pd

from backend.metrics import (
    SEGMENT_ROI_COLUMNS,
    bootstrap_mean_ci,
    bootstrap_roi_ci,
    calibration_curve,
    equity_curve_from_ledger,
    financial_summary,
    hit_rate_by_edge_bucket,
    ledger_segments,
    log_loss_terms,
    probabilistic_summary,
    regression_summary,
    segment_summary,
)
from backend.simulation import american_to_prob
from backend.strategy import V1_CUTOVER_DATE

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


EXCLUSION_REASONS = ("no_forecast", "v1_outage_hindsight_only", "forecast_dated_other_day")


def normalize_ledger(ledger):
    """Types shared by the database and CSV loaders."""
    if ledger.empty:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    ledger = ledger.copy()
    ledger["date"] = pd.to_datetime(ledger["date"])
    for column in ("won", "push", "rewritten"):
        ledger[column] = ledger[column].astype(bool)
    # bet_ledger_v owns the flag; only v1, which stored no publish time, may leave it unknown.
    ledger["written_after_start"] = _known_for_v2(ledger["written_after_start"].astype("boolean"),
                                                  ledger["model_version"], "bet_ledger_v.written_after_start")
    for column in ("stake", "decimal_odds", "payout", "edge", "american_odds"):
        ledger[column] = pd.to_numeric(ledger[column], errors="raise")
    return ledger


def _known_for_v2(flag, model_version, what):
    """Unknown lateness counts as on time only for v1; for any later model it is a data error."""
    unknown = flag.isna() & model_version.ne("v1")
    if unknown.any():
        raise ValueError(f"{what} is unknown for {int(unknown.sum())} non-v1 rows")
    return flag.fillna(False).astype(bool)


def written_after_start(rows):
    """Per forecast row: published at or after first pitch (prediction_updated_at vs the game's start_time).

    bet_ledger_v computes the same flag for bets; this covers every graded forecast, bet or not.
    """
    published = pd.to_datetime(rows["prediction_updated_at"], utc=True)
    start = pd.to_datetime(rows["game_start_time"], utc=True)
    flag = (published >= start).astype("boolean").mask(published.isna() | start.isna())
    return _known_for_v2(flag, rows["model_version"], "forecast publish or start time")


def clean_row(row):
    """NaN and inf become NULL for Postgres."""
    return {k: (None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v) for k, v in row.items()}


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
        away[["win_prob", "moneyline"]], rsuffix="_away", how="inner").sort_index()
    priced = games["moneyline"].abs().ge(100) & games["moneyline_away"].abs().ge(100)
    games = games[priced]
    if games.empty:
        return {"paired_games": 0}
    implied_home = games["moneyline"].map(american_to_prob)
    implied_away = games["moneyline_away"].map(american_to_prob)
    market = implied_home / (implied_home + implied_away)
    y = games["actual_win"].astype(float)
    # Model error averages both team rows, matching brier_score on the full window.
    model_sq = ((games["win_prob"] - y) ** 2 + (games["win_prob_away"] - (1 - y)) ** 2) / 2
    model_ll = (log_loss_terms(games["win_prob"], y) + log_loss_terms(games["win_prob_away"], 1 - y)) / 2
    market_sq = (market - y) ** 2
    low, high = bootstrap_mean_ci((model_sq - market_sq).to_numpy())
    return {
        "paired_games": int(len(games)),
        "paired_brier": float(model_sq.mean()),
        "paired_log_loss": float(model_ll.mean()),
        "market_brier": float(market_sq.mean()),
        "market_log_loss": float(log_loss_terms(market, y).mean()),
        "brier_vs_market_ci95_low": low,
        "brier_vs_market_ci95_high": high,
    }


def _interval_columns(window_df, window_ledger, segments):
    out = {}
    per_game_sq = (window_df["win_prob"] - window_df["actual_win"]) ** 2
    out["brier_ci95_low"], out["brier_ci95_high"] = bootstrap_mean_ci(
        per_game_sq.groupby(window_df["game_pk"]).mean().to_numpy())
    picks = _favorite_picks(window_df).sort_values("game_pk")
    out["total_accuracy_ci95_low"], out["total_accuracy_ci95_high"] = bootstrap_mean_ci(
        (picks["actual_win"] == 1).astype(float).to_numpy())
    out["roi_ci95_low"], out["roi_ci95_high"] = bootstrap_roi_ci(window_ledger)
    for name, bets in segments.items():
        column = SEGMENT_ROI_COLUMNS[name]
        out[f"{column}_ci95_low"], out[f"{column}_ci95_high"] = bootstrap_roi_ci(bets)
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
    cal_bins = []
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
        segments = ledger_segments(window_ledger)
        seg = segment_summary(window_ledger, segments)
        market = _market_comparison(window_df)

        # Equity is cumulative from the first graded bet of this window's model population.
        eq = equity_curve_from_ledger(population_ledger)
        equity_end = float(eq["equity"].iloc[-1]) if not eq.empty else 1.0

        window_dates = set(window_df["game_date"].dt.date)
        late_games = window_df.loc[written_after_start(window_df), "game_pk"].nunique()
        excluded = window_excluded["reason"].value_counts()

        metrics = {
            **reg,
            **prob,
            **fin,
            **seg,
            **market,
            **_interval_columns(window_df, window_ledger, segments),
            "equity_end_units": round(equity_end, 4),
            "scored_games": int(window_df["game_pk"].nunique()),
            "n_bets": int(len(window_ledger)),
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
        if window_name == "season":
            cal_bins = calibration_curve(probs, outcomes)
    return results, cal_bins
