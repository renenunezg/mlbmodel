import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import xgboost as xgb
from scipy.stats import poisson
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/mlbmodel")
engine = create_engine(DATABASE_URL)

# 11 features for the XGBoost model
FEATURE_COLS = [
    'xfip',               # starter xFIP (computed from Statcast)
    'xfip_bullpen',       # bullpen xFIP
    'starter_whip',       # starter WHIP
    'bullpen_k_9',        # bullpen K/9
    'batting_ops',        # team OPS vs opponent handedness
    'batting_iso',        # team ISO vs opponent handedness (power)
    'batting_k_pct',      # team K% vs opponent handedness (contact)
    'avg_last5',          # rolling 5-game run average
    'avg_last10',         # rolling 10-game run average
    'park_factor',        # park factor (affects offense)
    'is_home',            # home field advantage (0/1)
]


def _safe_read_table(table_name):
    """Read a SQL table, returning empty DataFrame if table has no rows."""
    try:
        df = pd.read_sql_table(table_name, con=engine)
        if df.empty:
            print(f"  Warning: {table_name} is empty")
        return df
    except Exception as e:
        print(f"  Warning: could not read {table_name}: {e}")
        return pd.DataFrame()


# League-average fallbacks for early season when stats tables are empty
LEAGUE_AVG = {
    "xfip": 4.20, "whip": 1.30, "xfip_bullpen": 4.10, "bullpen_k_9": 9.0,
    "batting_ops": 0.720, "batting_iso": 0.160, "batting_k_pct": 22.0,
    "park_factor": 100, "avg_last5": 4.5, "avg_last10": 4.5, "std_last5": 2.5,
}


def load_training_data():
    """Load and merge all data sources into a single training DataFrame.

    Handles early-season scenarios where tables may be empty or have
    sparse data by falling back to league averages.
    """
    print("Loading training data...")
    starters = _safe_read_table("probable_starters")
    if starters.empty:
        print("  ERROR: probable_starters is empty — cannot proceed.")
        return pd.DataFrame()

    # Merge in pitcher stats (xfip, whip) via pitcher_id (MLB player ID)
    sp_stats = _safe_read_table("pitcher_stats")
    if not sp_stats.empty and "pitcher_id" in sp_stats.columns and "pitcher_id" in starters.columns:
        starters = pd.merge(
            starters,
            sp_stats[["pitcher_id", "xfip", "whip"]],
            on="pitcher_id",
            how="left",
        )
    else:
        starters["xfip"] = np.nan
        starters["whip"] = np.nan

    # Merge in bullpen stats (xfip, k_9)
    bp_stats = _safe_read_table("bullpen_stats")
    if not bp_stats.empty:
        starters = pd.merge(
            starters,
            bp_stats[["team", "xfip", "k_9"]].rename(columns={"xfip": "xfip_bullpen", "k_9": "bullpen_k_9"}),
            on="team",
            how="left",
        )
    else:
        starters["xfip_bullpen"] = np.nan
        starters["bullpen_k_9"] = np.nan

    # Load batting splits from unified team_batting table
    batting = _safe_read_table("team_batting")
    bat_cols = ["team", "split", "ops", "iso", "k_pct"]
    if not batting.empty:
        batting = batting[[c for c in bat_cols if c in batting.columns]]
        vs_r = batting[batting["split"] == "vs_rhp"].drop(columns=["split"]).rename(
            columns={"ops": "ops_vs_r", "iso": "iso_vs_r", "k_pct": "k_pct_vs_r"}
        )
        vs_l = batting[batting["split"] == "vs_lhp"].drop(columns=["split"]).rename(
            columns={"ops": "ops_vs_l", "iso": "iso_vs_l", "k_pct": "k_pct_vs_l"}
        )
        starters = pd.merge(starters, vs_r, on="team", how="left")
        starters = pd.merge(starters, vs_l, on="team", how="left")
    else:
        for suffix in ["_vs_r", "_vs_l"]:
            for col in ["ops", "iso", "k_pct"]:
                starters[f"{col}{suffix}"] = np.nan

    # Self-merge to get opponent pitcher handedness
    opp = starters[["game_pk", "team", "handedness"]].rename(
        columns={"team": "opp_team", "handedness": "opp_handedness"}
    )
    starters = pd.merge(starters, opp, on="game_pk")
    starters = starters[starters["team"] != starters["opp_team"]]

    # Select batting stats based on opponent pitcher handedness
    for stat, vs_r_col, vs_l_col in [
        ("batting_ops", "ops_vs_r", "ops_vs_l"),
        ("batting_iso", "iso_vs_r", "iso_vs_l"),
        ("batting_k_pct", "k_pct_vs_r", "k_pct_vs_l"),
    ]:
        starters[stat] = np.where(
            starters["opp_handedness"] == "R",
            starters.get(vs_r_col, np.nan),
            starters.get(vs_l_col, np.nan),
        )

    # Load park factors and merge based on home team
    parks = _safe_read_table("park_factors")
    if not parks.empty:
        home_teams = starters[starters["is_home"] == True][["game_pk", "team"]].rename(columns={"team": "home_team"})
        starters = pd.merge(starters, home_teams, on="game_pk", how="left")
        home_parks = pd.merge(home_teams, parks, left_on="home_team", right_on="team", how="left")[["game_pk", "park_factor"]]
        starters = pd.merge(starters, home_parks, on="game_pk", how="left")
    else:
        starters["park_factor"] = 100

    # Load completed games
    games = _safe_read_table("games")
    if not games.empty:
        games = games[games["status"] == "Final"]
        games = games.dropna(subset=["away_score", "home_score"])

    if games.empty:
        print("  Warning: no completed games — rolling averages will use league average")
        starters["game_date"] = pd.NaT
        starters["actual_runs"] = np.nan
        starters["avg_last5"] = np.nan
        starters["avg_last10"] = np.nan
        starters["std_last5"] = np.nan
    else:
        starters = pd.merge(
            starters,
            games[["game_pk", "game_date", "away_team", "away_score", "home_team", "home_score"]].rename(
                columns={"home_team": "game_home_team", "away_team": "game_away_team"}
            ),
            on="game_pk",
            how="left",
        )

        starters["actual_runs"] = np.where(
            starters["is_home"] == True,
            starters["home_score"],
            starters["away_score"],
        )

        # Compute rolling averages
        games["game_date"] = pd.to_datetime(games["game_date"])
        starters["game_date"] = pd.to_datetime(starters["game_date"])

        home_runs = games[["game_date", "home_team", "home_score"]].rename(columns={"home_team": "team", "home_score": "runs"})
        away_runs = games[["game_date", "away_team", "away_score"]].rename(columns={"away_team": "team", "away_score": "runs"})
        team_games = pd.concat([home_runs, away_runs]).sort_values(["team", "game_date"]).reset_index(drop=True)

        def get_rolling_averages(team, current_date):
            if pd.isna(current_date):
                return pd.Series({"avg_last5": np.nan, "avg_last10": np.nan, "std_last5": np.nan})
            past = team_games[(team_games["team"] == team) & (team_games["game_date"] < current_date)]
            last5 = past.tail(5)["runs"]
            last10 = past.tail(10)["runs"]
            return pd.Series({
                "avg_last5": last5.mean() if not last5.empty else np.nan,
                "avg_last10": last10.mean() if not last10.empty else np.nan,
                "std_last5": last5.std() if len(last5) >= 2 else np.nan,
            })

        starters[["avg_last5", "avg_last10", "std_last5"]] = starters.apply(
            lambda row: get_rolling_averages(row["team"], row["game_date"]), axis=1
        )

    # Filter for games with exactly 2 teams
    starters = starters.groupby("game_pk").filter(lambda x: len(x) == 2)
    starters = starters.sort_values(["game_pk", "is_home"]).reset_index(drop=True)

    # Fill NaN features with league averages
    for col, fallback in LEAGUE_AVG.items():
        if col in starters.columns:
            n_missing = starters[col].isna().sum()
            if n_missing > 0:
                print(f"  Filling {n_missing} NaN values in '{col}' with league avg ({fallback})")
                starters[col] = starters[col].fillna(fallback)

    starters["starter_whip"] = starters.get("whip", starters.get("starter_whip", LEAGUE_AVG["whip"]))

    out_cols = [
        "game_pk", "game_date", "team", "pitcher_name", "is_home",
        "xfip", "starter_whip", "xfip_bullpen", "bullpen_k_9",
        "batting_ops", "batting_iso", "batting_k_pct",
        "park_factor", "actual_runs", "avg_last5", "avg_last10", "std_last5",
    ]
    result = starters[[c for c in out_cols if c in starters.columns]].copy()
    result = result.rename(columns={"pitcher_name": "starter"})

    print(f"  Loaded {len(result)} rows ({result['game_pk'].nunique()} games)")
    return result


def preprocess_data(df):
    """Convert columns to numeric types."""
    numeric_cols = [
        "xfip", "xfip_bullpen", "starter_whip", "bullpen_k_9",
        "batting_ops", "batting_iso", "batting_k_pct",
        "avg_last5", "avg_last10", "std_last5", "park_factor",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


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
    return model, []


def train_model_cv(X, y, n_splits=5):
    """Train XGBoost with TimeSeriesSplit cross-validation and hyperparameter tuning.

    Falls back to simple training if there are too few samples for CV.
    Returns the best model and CV metrics.
    """
    min_samples_for_cv = (n_splits + 1) * 10  # need at least 10 samples per fold
    if len(X) < min_samples_for_cv:
        print(f"\nOnly {len(X)} samples — need {min_samples_for_cv} for {n_splits}-fold CV.")
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

    # Report per-fold metrics with best params
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(
            objective="reg:squarederror", random_state=42, **grid_search.best_params_
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        fold_metrics.append({"fold": fold, "mae": mae, "rmse": rmse})
        print(f"  Fold {fold}: MAE={mae:.3f}, RMSE={rmse:.3f}")

    avg_mae = np.mean([m["mae"] for m in fold_metrics])
    avg_rmse = np.mean([m["rmse"] for m in fold_metrics])
    print(f"  CV Average: MAE={avg_mae:.3f}, RMSE={avg_rmse:.3f}")

    return grid_search.best_estimator_, fold_metrics


def poisson_win_prob(lambda_a, lambda_b, max_runs=15):
    """Compute P(team A wins) using independent Poisson distributions.

    Ties are allocated proportionally (extra innings approximation).
    """
    a_probs = poisson.pmf(np.arange(max_runs + 1), lambda_a)
    b_probs = poisson.pmf(np.arange(max_runs + 1), lambda_b)

    # Joint probability matrix: rows = team A scores, cols = team B scores
    joint = np.outer(a_probs, b_probs)

    p_win = np.tril(joint, k=-1).sum()   # P(A > B)
    p_tie = np.trace(joint)               # P(A == B)

    # Allocate ties proportionally to expected runs
    total_lambda = lambda_a + lambda_b
    tie_share = lambda_a / total_lambda if total_lambda > 0 else 0.5
    p_win += p_tie * tie_share

    return round(float(p_win), 4)


def convert_to_odds(p):
    """Convert win probability to American odds."""
    if p < 0.5:
        return round(((1 - p) / p) * 100)
    elif p > 0.5:
        return round(-(p / (1 - p)) * 100) if p < 1 else -1000
    else:
        return 100


def american_to_prob(odds):
    """Convert American odds to implied probability."""
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def flag_ev(row, threshold=0.03):
    try:
        our_prob = american_to_prob(row["our_odds"])
        book_prob = american_to_prob(row["moneyline"])
        if pd.isna(book_prob):
            return "No Play"
        edge = our_prob - book_prob
        return row["team"] if edge >= threshold else "No Play"
    except Exception:
        return "No Play"


def flag_runline_ev(row, threshold=0.03):
    try:
        book_prob = american_to_prob(row["spread_odds"])
        if pd.isna(book_prob):
            return "No Play"
        model_prob = max(row["win_prob"] - 0.10, 0)
        edge = model_prob - book_prob
        return row["team"] if edge >= threshold else "No Play"
    except Exception:
        return "No Play"


def compute_predictions(df, model):
    """Predict expected runs and compute Poisson-based win probabilities."""

    df = df.copy()
    df["is_home"] = df["is_home"].astype(int)

    X = df[FEATURE_COLS].copy()
    df["xR"] = model.predict(X).round(2)

    # Compute Poisson win probs per game
    df["win_prob"] = np.nan
    df["our_odds"] = np.nan

    for game_pk, group in df.groupby("game_pk"):
        if len(group) != 2:
            continue
        rows = group.sort_values("is_home", ascending=True)
        lambda_away = max(rows.iloc[0]["xR"], 0.5)
        lambda_home = max(rows.iloc[1]["xR"], 0.5)

        p_home = poisson_win_prob(lambda_home, lambda_away)
        p_away = 1.0 - p_home

        df.loc[group.index[group["is_home"] == 1], "win_prob"] = round(p_home, 3)
        df.loc[group.index[group["is_home"] == 0], "win_prob"] = round(p_away, 3)

    df["win_prob"] = df["win_prob"].clip(0.001, 0.999)
    df["our_odds"] = df["win_prob"].apply(convert_to_odds)
    return df


def _upsert_season_outputs(df):
    """Upsert into model_outputs_season using ON CONFLICT (game_pk, team)."""
    cols = df.columns.tolist()
    placeholders = ", ".join([f":{c}" for c in cols])
    col_names = ", ".join(cols)
    update_set = ", ".join([f"{c} = EXCLUDED.{c}" for c in cols if c not in ("game_pk", "team")])

    sql = f"""
        INSERT INTO model_outputs_season ({col_names})
        VALUES ({placeholders})
        ON CONFLICT (game_pk, team) DO UPDATE SET {update_set}
    """

    with engine.begin() as conn:
        for _, row in df.iterrows():
            params = {}
            for c in cols:
                val = row[c]
                if pd.isna(val):
                    params[c] = None
                elif isinstance(val, (np.integer,)):
                    params[c] = int(val)
                elif isinstance(val, (np.floating,)):
                    params[c] = float(val)
                elif isinstance(val, (np.bool_,)):
                    params[c] = bool(val)
                else:
                    params[c] = val
            conn.execute(text(sql), params)


def main():
    df = load_training_data()
    if df.empty:
        print("No training data available. Exiting.")
        return

    df = preprocess_data(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Split: rows with actual results (for training) vs upcoming games (for prediction only)
    train_df = df.dropna(subset=["actual_runs"]).copy()
    upcoming_df = df[df["actual_runs"].isna()].copy()

    if train_df.empty:
        print("No completed games to train on yet. Using league-average model.")
        # For day 1-2 of season: still produce predictions using a dummy model

        df["is_home"] = df["is_home"].astype(int)
        # Use league avg xR (~4.5) as baseline prediction
        df["xR"] = LEAGUE_AVG["avg_last5"]
        train_df = df
        model = None
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
            print(f"Only {len(X)} training samples — too few for reliable model. Using league avg.")
            model = None
            train_df["xR"] = LEAGUE_AVG["avg_last5"]
        else:
            model, cv_metrics = train_model_cv(X, y)

    # Save model if we trained one
    if model is not None:
        model.save_model("xgb_model.json")
        print("Model trained and saved to xgb_model.json")

    # Produce predictions on all data (trained games + upcoming)
    predict_df = df.copy()
    predict_df = preprocess_data(predict_df)
    predict_df = predict_df.replace([np.inf, -np.inf], np.nan)

    # Fill NaN features with league averages for prediction
    for col, fallback in LEAGUE_AVG.items():
        if col in predict_df.columns:
            predict_df[col] = predict_df[col].fillna(fallback)

    if model is not None:
        predictions = compute_predictions(predict_df, model)
    else:
        # No trained model — use league-average xR and Poisson for win probs
        predict_df = predict_df.copy()
        predict_df["is_home"] = predict_df["is_home"].astype(int)
        predict_df["xR"] = LEAGUE_AVG["avg_last5"]
        predict_df["win_prob"] = np.where(predict_df["is_home"] == 1, 0.54, 0.46)
        predict_df["our_odds"] = predict_df["win_prob"].apply(convert_to_odds)
        predictions = predict_df

    final_output = predictions.sort_values(
        ["game_pk", "is_home"], ascending=[True, True]
    ).reset_index(drop=True)[["game_pk", "game_date", "team", "starter", "xR", "win_prob", "our_odds"]]

    # Merge odds data
    odds_df = _safe_read_table("odds")
    if not odds_df.empty:
        odds_df = odds_df.drop_duplicates(subset=["game_pk", "team"], keep="first")
        final_output = pd.merge(
            final_output,
            odds_df[["game_pk", "team", "moneyline", "total", "spread", "spread_odds"]],
            on=["game_pk", "team"],
            how="left",
        )
    else:
        for col in ["moneyline", "total", "spread", "spread_odds"]:
            final_output[col] = np.nan

    # Compute additional columns
    our_totals = final_output.groupby("game_pk")["xR"].sum().round(2).rename("our_total")
    final_output = final_output.merge(our_totals, on="game_pk", how="left")
    final_output["total"] = pd.to_numeric(final_output["total"], errors="coerce")
    final_output["total_diff"] = (final_output["our_total"] - final_output["total"]).round(2)

    def flag_total_play(row):
        if pd.isna(row["total_diff"]):
            return "No Play"
        if row["total_diff"] >= 1:
            return "Over"
        elif row["total_diff"] <= -1:
            return "Under"
        return "No Play"

    final_output["total_play"] = final_output.apply(flag_total_play, axis=1)
    final_output["ev_flag"] = final_output.apply(flag_ev, axis=1)
    final_output["run_line_ev_flag"] = final_output.apply(flag_runline_ev, axis=1)

    final_output["ml_confidence"] = final_output.apply(
        lambda row: round(row["win_prob"] - american_to_prob(row["moneyline"]), 3)
        if pd.notna(row.get("moneyline")) else np.nan,
        axis=1,
    )

    final_output["run_line_confidence"] = final_output.apply(
        lambda row: round(max(row["win_prob"] - 0.10, 0) - american_to_prob(row["spread_odds"]), 3)
        if pd.notna(row.get("spread_odds")) else np.nan,
        axis=1,
    )

    final_output["high_variance_flag"] = predictions.get("std_last5", pd.Series(dtype=float)).apply(
        lambda x: "Yes" if pd.notna(x) and x > 4.0 else "No"
    )

    print("\nFinal output with flags:")
    print(final_output)

    # Insert into model_outputs table (today's snapshot, replaced each run)
    final_output = final_output.rename(columns={"xR": "expected_runs", "game_date": "date"})
    final_output.to_sql("model_outputs", con=engine, if_exists="replace", index=False)

    # Upsert into model_outputs_season (historical record, deduplicated by game_pk+team)
    float_cols = ["expected_runs", "win_prob", "our_total"]
    for col in float_cols:
        if col in final_output.columns:
            final_output[col] = final_output[col].map(lambda x: round(x, 2) if pd.notnull(x) else x)

    final_output = final_output.round(2)
    _upsert_season_outputs(final_output)

    print("\nmodel_outputs tables updated.")


if __name__ == "__main__":
    main()
