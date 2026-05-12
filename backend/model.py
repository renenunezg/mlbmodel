"""XGBoost training, calibration, and prediction for expected runs.

Just the model. Data loading is in data.py, NB simulation is in simulation.py,
EV/Kelly are in strategy.py. The main() orchestrator at the bottom ties them
together for the daily pipeline.
"""
import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
from sqlalchemy import text
import xgboost as xgb
import joblib

# Persisted model artifacts live in models/ at the repo root.
_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_MODELS_DIR.mkdir(parents=True, exist_ok=True)
XGB_MODEL_PATH = _MODELS_DIR / "xgb_model.json"
CALIBRATOR_PATH = _MODELS_DIR / "isotonic_calibrator.pkl"
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.isotonic import IsotonicRegression
import warnings

from backend.db import engine
from backend.features import (
    LEAGUE_AVG,
    _safe_read_table,
    load_training_data,
    preprocess_data,
)
from backend.simulation import (
    win_prob,
    convert_to_odds,
    american_to_prob,
    apply_market_probs,
)
from backend.strategy import (
    flag_ev,
    flag_runline_ev,
    flag_total_play,
    apply_kelly_sizing,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


# 14 features for the XGBoost model
FEATURE_COLS = [
    'xfip',               # starter xFIP (computed from Statcast)
    'xfip_bullpen',       # bullpen xFIP
    'starter_whip',       # starter WHIP
    'bullpen_k_9',        # bullpen K/9
    'batting_ops',        # team OPS blended across starter + bullpen handedness
    'batting_iso',        # team ISO blended across starter + bullpen handedness
    'batting_k_pct',      # team K% blended across starter + bullpen handedness
    'avg_last5',          # rolling 5-game run average
    'avg_last10',         # rolling 10-game run average
    'std_last5',          # rolling 5-game std dev (volatility signal)
    'park_factor',        # park factor (affects offense)
    'is_home',            # home field advantage (0/1)
    'own_bp_outs_2d',     # own bullpen reliever outs in prior 2 days (rest signal)
    'opp_bp_outs_2d',     # opposing bullpen reliever outs in prior 2 days (fatigue signal)
]


def _train_simple(X, y):
    """Fallback training with default params when data is too sparse for CV."""
    print("\nTraining with default params (too few samples for CV)...")
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    model.fit(X, y)
    # No OOF predictions possible without CV - calibrator will be skipped downstream.
    return model, [], np.full(len(X), np.nan), {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1}


def train_model_cv(X, y, n_splits=5):
    """Train XGBoost with TimeSeriesSplit cross-validation and hyperparameter tuning.

    Falls back to simple training if there are too few samples for CV.
    Returns the best model and CV metrics.
    """
    min_samples_for_cv = (n_splits + 1) * 10  # need at least 10 samples per fold
    if len(X) < min_samples_for_cv:
        print(f"\nOnly {len(X)} samples - need {min_samples_for_cv} for {n_splits}-fold CV.")
        if len(X) < 20:
            return _train_simple(X, y)
        # Reduce folds for small datasets
        n_splits = max(2, len(X) // 10 - 1)
        print(f"  Reducing to {n_splits} folds.")

    print(f"\nCross-validating with TimeSeriesSplit (n_splits={n_splits})...")

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1],
        "min_child_weight": [3, 5],
    }

    grid_search = GridSearchCV(
        xgb.XGBRegressor(objective="reg:squarederror", random_state=42),
        param_grid,
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="neg_mean_absolute_error",
        verbose=0,
        n_jobs=-1,
    )
    grid_search.fit(X, y)

    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV MAE: {-grid_search.best_score_:.3f}")

    # Report per-fold metrics with best params - also collect out-of-fold predictions
    # aligned to X's row positions. These OOF predictions are used downstream to fit
    # the isotonic calibrator without in-sample leakage.
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    oof_preds = np.full(len(X), np.nan)
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(
            objective="reg:squarederror", random_state=42, **grid_search.best_params_
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        oof_preds[val_idx] = preds

        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        fold_metrics.append({"fold": fold, "mae": mae, "rmse": rmse})
        print(f"  Fold {fold}: MAE={mae:.3f}, RMSE={rmse:.3f}")

    avg_mae = np.mean([m["mae"] for m in fold_metrics])
    avg_rmse = np.mean([m["rmse"] for m in fold_metrics])
    print(f"  CV Average: MAE={avg_mae:.3f}, RMSE={avg_rmse:.3f}")

    n_oof = int(np.sum(~np.isnan(oof_preds)))
    print(f"  Collected {n_oof}/{len(X)} OOF predictions for calibration")

    return grid_search.best_estimator_, fold_metrics, oof_preds, grid_search.best_params_


def compute_predictions(df, model, calibrator=None, use_existing_xR=False):
    """Predict expected runs and compute negative-binomial win probabilities.

    If calibrator is provided, applies isotonic regression to calibrate
    win probabilities after the NB step, then renormalizes per game.

    If use_existing_xR is True, the caller has already set `xR` on df (e.g.
    out-of-fold predictions from CV) and model.predict is skipped. This lets
    us compute leak-free win probabilities for calibrator fitting.
    """
    df = df.copy()
    df["is_home"] = df["is_home"].astype(int)

    if use_existing_xR:
        df["xR"] = pd.to_numeric(df["xR"], errors="coerce").round(2)
    else:
        X = df[FEATURE_COLS].copy()
        df["xR"] = model.predict(X).round(2)

    # Compute negative-binomial win probs per game
    df["win_prob"] = np.nan
    df["our_odds"] = np.nan

    for game_pk, group in df.groupby("game_pk"):
        if len(group) != 2:
            continue
        rows = group.sort_values("is_home", ascending=True)
        lambda_away = max(rows.iloc[0]["xR"], 0.5)
        lambda_home = max(rows.iloc[1]["xR"], 0.5)

        p_home = win_prob(lambda_home, lambda_away)
        p_away = 1.0 - p_home

        df.loc[group.index[group["is_home"] == 1], "win_prob"] = round(p_home, 3)
        df.loc[group.index[group["is_home"] == 0], "win_prob"] = round(p_away, 3)

    df["win_prob"] = df["win_prob"].clip(0.05, 0.95)

    if calibrator is not None:
        df["win_prob"] = calibrator.predict(df["win_prob"].values)
        # Renormalize per game so complementary probs sum to 1
        for game_pk, group in df.groupby("game_pk"):
            if len(group) == 2:
                total = df.loc[group.index, "win_prob"].sum()
                if total > 0:
                    df.loc[group.index, "win_prob"] = df.loc[group.index, "win_prob"] / total
        df["win_prob"] = df["win_prob"].clip(0.05, 0.95)

    df["our_odds"] = df["win_prob"].apply(convert_to_odds)
    return df


def fit_calibrator(df):
    """Isotonic regression: NB win probs → observed outcomes. None if <400 outcomes."""
    outcomes = []
    for game_pk, group in df.groupby("game_pk"):
        if len(group) != 2:
            continue
        rows = group.sort_values("is_home")
        away_runs = rows.iloc[0]["actual_runs"]
        home_runs = rows.iloc[1]["actual_runs"]
        if pd.isna(away_runs) or pd.isna(home_runs):
            continue
        if home_runs > away_runs:
            outcomes.append((rows.index[0], 0.0))  # away lost
            outcomes.append((rows.index[1], 1.0))  # home won
        elif away_runs > home_runs:
            outcomes.append((rows.index[0], 1.0))  # away won
            outcomes.append((rows.index[1], 0.0))  # home lost
        else:
            outcomes.append((rows.index[0], 0.5))  # tie
            outcomes.append((rows.index[1], 0.5))

    if len(outcomes) < 400:
        print(f"  Only {len(outcomes)} outcomes - skipping calibration (need >= 400)")
        return None

    idx, y_true = zip(*outcomes)
    y_pred = df.loc[list(idx), "win_prob"].values
    y_true = np.array(y_true)

    calibrator = IsotonicRegression(y_min=0.05, y_max=0.95, out_of_bounds="clip")
    calibrator.fit(y_pred, y_true)
    print(f"  Isotonic calibrator fit on {len(y_true)} outcomes ({len(y_true) // 2} games)")
    return calibrator


def _get_table_columns(table_name):
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = :t"
            ),
            {"t": table_name},
        )
        return {row[0] for row in result}


def _upsert_season_outputs(df):
    """Upsert into model_outputs_season. Filters df to columns the table actually has."""
    existing_cols = _get_table_columns("model_outputs_season")
    cols = [c for c in df.columns if c in existing_cols]
    if not cols:
        print("  WARNING: no matching columns for model_outputs_season upsert")
        return
    placeholders = ", ".join([f":{c}" for c in cols])
    col_names = ", ".join(cols)
    update_set = ", ".join([f"{c} = EXCLUDED.{c}" for c in cols if c not in ("game_pk", "team")])

    # Once a game leaves Preview (goes Live or Final), its prediction row is
    # read-only. INSERTs for new game_pks are unaffected. Without this guard,
    # a mid-day hand-run rewrites bets that were already in flight.
    sql = f"""
        INSERT INTO model_outputs_season ({col_names})
        VALUES ({placeholders})
        ON CONFLICT (game_pk, team) DO UPDATE SET {update_set}
        WHERE EXISTS (
            SELECT 1 FROM games g
            WHERE g.game_pk = model_outputs_season.game_pk
              AND g.status = 'Preview'
        )
    """

    def _coerce(val):
        if pd.isna(val):
            return None
        if isinstance(val, np.integer):
            return int(val)
        if isinstance(val, np.floating):
            return float(val)
        if isinstance(val, np.bool_):
            return bool(val)
        return val

    batch = [{c: _coerce(row[c]) for c in cols} for _, row in df.iterrows()]
    if not batch:
        return

    with engine.begin() as conn:
        conn.execute(text(sql), batch)


def main():
    df = load_training_data()
    if df.empty:
        print("No training data available. Exiting.")
        return

    df = preprocess_data(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Split: rows with actual results (for training) vs upcoming games
    train_df = df.dropna(subset=["actual_runs"]).copy()

    # Predict only on today's games - past games are training data only,
    # re-predicting them would be cheating and skew output
    # Use Pacific timezone (handles PST/PDT automatically)
    today_str = datetime.datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()

    # game_date is NaT for upcoming games (not Final yet), so look up today's
    # game_pks directly from the games table rather than filtering by date on df
    with engine.connect() as conn:
        today_pks_df = pd.read_sql(
            text("SELECT game_pk FROM games WHERE game_date = :today"),
            conn,
            params={"today": today_str},
        )
    today_pks = set(today_pks_df["game_pk"].tolist())
    today_df = df[df["game_pk"].isin(today_pks)].copy()
    today_df["game_date"] = today_str  # upcoming games have NaT; set explicitly

    if today_df.empty:
        print(f"  No games found for today ({today_str}). Nothing to predict.")
        return

    if train_df.empty:
        print("No completed games to train on yet. Using league-average model.")
        # For day 1-2 of season: still produce predictions using a dummy model

        df["is_home"] = df["is_home"].astype(int)
        # Use league avg xR (~4.5) as baseline prediction
        df["xR"] = LEAGUE_AVG["avg_last5"]
        train_df = df
        model = None
        cv_metrics = None
        best_params = None
    else:
        # Sort by date for proper temporal ordering in CV
        train_df = train_df.sort_values("game_date").reset_index(drop=True)


        train_df["is_home"] = train_df["is_home"].astype(int)

        X = train_df[FEATURE_COLS].copy()
        y = pd.to_numeric(train_df["actual_runs"], errors="coerce")

        # Drop rows where target is NaN
        valid = y.dropna().index
        X = X.loc[valid]
        y = y.loc[valid]

        if len(X) < 4:
            print(f"Only {len(X)} training samples - too few for reliable model. Using league avg.")
            model = None
            oof_preds = None
            cv_metrics = None
            best_params = None
            train_df["xR"] = LEAGUE_AVG["avg_last5"]
        else:
            model, cv_metrics, oof_preds, best_params = train_model_cv(X, y)

    # Save model and fit calibrator if we trained one
    calibrator = None
    if model is not None:
        model.save_model(str(XGB_MODEL_PATH))
        print(f"Model trained and saved to {XGB_MODEL_PATH}")

        # Fit isotonic calibrator on OUT-OF-FOLD predictions (leak-free).
        # Rows that were never in a validation fold have NaN xR and get dropped.
        print("\nFitting isotonic calibrator on OOF predictions...")
        if oof_preds is not None and np.any(~np.isnan(oof_preds)):
            cal_df = train_df.loc[X.index].copy()
            cal_df["xR"] = oof_preds
            cal_df = cal_df.dropna(subset=["xR"])
            # Keep only games where BOTH teams have OOF predictions
            pair_counts = cal_df.groupby("game_pk").size()
            valid_games = pair_counts[pair_counts == 2].index
            cal_df = cal_df[cal_df["game_pk"].isin(valid_games)]
            if len(cal_df) >= 800:
                train_preds = compute_predictions(cal_df, model, calibrator=None, use_existing_xR=True)
                calibrator = fit_calibrator(train_preds)
                if calibrator is not None:
                    joblib.dump(calibrator, CALIBRATOR_PATH)
                    print(f"  Calibrator saved to {CALIBRATOR_PATH}")
            else:
                print(f"  Only {len(cal_df)} paired OOF rows - skipping calibration (need >= 800)")
        else:
            print("  No OOF predictions available - skipping calibration")

    # Predict only on today's games
    predict_df = today_df.copy()
    predict_df = preprocess_data(predict_df)
    predict_df = predict_df.replace([np.inf, -np.inf], np.nan)

    # Fill NaN features with league averages for prediction
    for col, fallback in LEAGUE_AVG.items():
        if col in predict_df.columns:
            predict_df[col] = predict_df[col].fillna(fallback)

    if model is not None:
        predictions = compute_predictions(predict_df, model, calibrator=calibrator)
    else:
        # No trained model - use league-average xR and home-field win prob
        predict_df["is_home"] = predict_df["is_home"].astype(int)
        predict_df["xR"] = LEAGUE_AVG["avg_last5"]
        predict_df["win_prob"] = np.where(predict_df["is_home"] == 1, 0.54, 0.46)
        predict_df["our_odds"] = predict_df["win_prob"].apply(convert_to_odds)
        predictions = predict_df

    final_output = predictions.sort_values(
        ["game_pk", "is_home"], ascending=[True, True]
    ).reset_index(drop=True)[["game_pk", "game_date", "team", "starter", "xR", "win_prob", "our_odds", "is_home"]]

    # Merge odds data - include total_over_odds/total_under_odds so the totals EV
    # flag can compare model over/under probability to the book's implied price.
    odds_df = _safe_read_table("odds")
    odds_cols = ["game_pk", "team", "moneyline", "total", "spread", "spread_odds",
                 "total_over_odds", "total_under_odds"]
    if not odds_df.empty:
        odds_df = odds_df.drop_duplicates(subset=["game_pk", "team"], keep="first")
        available = [c for c in odds_cols if c in odds_df.columns]
        final_output = pd.merge(
            final_output,
            odds_df[available],
            on=["game_pk", "team"],
            how="left",
        )
        for col in odds_cols:
            if col not in final_output.columns:
                final_output[col] = np.nan
    else:
        for col in odds_cols:
            if col not in ("game_pk", "team"):
                final_output[col] = np.nan

    # Compute additional columns
    our_totals = final_output.groupby("game_pk")["xR"].sum().round(2).rename("our_total")
    final_output = final_output.merge(our_totals, on="game_pk", how="left")
    final_output["total"] = pd.to_numeric(final_output["total"], errors="coerce")
    final_output["total_diff"] = (final_output["our_total"] - final_output["total"]).round(2)

    # Joint-distribution market probabilities (p_cover, p_over, p_under) from
    # the book's actual total line and spread.
    final_output = apply_market_probs(final_output)

    final_output["total_play"] = final_output.apply(flag_total_play, axis=1)
    final_output["ev_flag"] = final_output.apply(flag_ev, axis=1)
    final_output["run_line_ev_flag"] = final_output.apply(flag_runline_ev, axis=1)

    # Kelly sizing for ML / RL / totals
    final_output = apply_kelly_sizing(final_output)

    # Displayed edge = implied-probability diff used by flag_ev / flag_runline_ev.
    # Must match the threshold logic so cards and +EV badges agree.
    final_output["ml_confidence"] = (
        final_output["our_odds"].apply(american_to_prob)
        - final_output["moneyline"].apply(american_to_prob)
    )
    final_output["run_line_confidence"] = (
        final_output["p_cover"] - final_output["spread_odds"].apply(american_to_prob)
    )

    final_output["high_variance_flag"] = predictions.get("std_last5", pd.Series(dtype=float)).apply(
        lambda x: "Yes" if pd.notna(x) and x > 4.0 else "No"
    )

    print("\nFinal output with flags:")
    print(final_output)

    # Refresh model_outputs (today's snapshot). TRUNCATE + append preserves the
    # table's RLS, policies, and indexes; to_sql(if_exists="replace") would drop them.
    final_output = final_output.rename(columns={"xR": "expected_runs", "game_date": "date"})
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE model_outputs"))
        final_output.to_sql("model_outputs", con=conn, if_exists="append", index=False)

    # Upsert into model_outputs_season (historical record, deduplicated by game_pk+team)
    float_cols = ["expected_runs", "win_prob", "our_total"]
    for col in float_cols:
        if col in final_output.columns:
            final_output[col] = final_output[col].map(lambda x: round(x, 2) if pd.notnull(x) else x)

    final_output = final_output.round(2)
    _upsert_season_outputs(final_output)

    print("\nmodel_outputs tables updated.")

    # Return model artifacts so the pipeline can pass them to evaluate_model
    return {
        "model": model,
        "cv_metrics": cv_metrics if model is not None else None,
        "best_params": best_params if model is not None else None,
    }


if __name__ == "__main__":
    main()
