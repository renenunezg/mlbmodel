import os
import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import xgboost as xgb
import joblib
from scipy.stats import nbinom
from backend.kelly import american_to_decimal, kelly_fraction, compute_kelly_row
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/mlbmodel")
engine = create_engine(DATABASE_URL)

# 12 features for the XGBoost model
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

    # Blend batting splits: ~60% of innings face the starter (known handedness),
    # ~40% the bullpen (~60% RHP league-wide regardless of starter).
    STARTER_INNING_SHARE = 0.6
    BULLPEN_RHP_SHARE = 0.6
    for stat, vs_r_col, vs_l_col in [
        ("batting_ops", "ops_vs_r", "ops_vs_l"),
        ("batting_iso", "iso_vs_r", "iso_vs_l"),
        ("batting_k_pct", "k_pct_vs_r", "k_pct_vs_l"),
    ]:
        vs_r = starters.get(vs_r_col, pd.Series(np.nan, index=starters.index))
        vs_l = starters.get(vs_l_col, pd.Series(np.nan, index=starters.index))
        split_vs_starter = np.where(starters["opp_handedness"] == "R", vs_r, vs_l)
        split_vs_bullpen = BULLPEN_RHP_SHARE * vs_r + (1 - BULLPEN_RHP_SHARE) * vs_l
        starters[stat] = (
            STARTER_INNING_SHARE * split_vs_starter
            + (1 - STARTER_INNING_SHARE) * split_vs_bullpen
        )

    # Load park factors and merge based on home team
    parks = _safe_read_table("park_factors")
    if not parks.empty:
        home_teams = starters[starters["is_home"] == True][["game_pk", "team"]].rename(columns={"team": "home_team"})
        starters = pd.merge(starters, home_teams, on="game_pk", how="left")
        home_parks = pd.merge(home_teams, parks, left_on="home_team", right_on="team", how="left")[["game_pk", "home_team", "park_factor"]]
        # Audit: surface silent merge failures (unmapped home team names) as loud warnings.
        unmapped = home_parks[home_parks["park_factor"].isna()]["home_team"].unique().tolist()
        if unmapped:
            print(f"  WARNING: park_factors missing for home teams {unmapped} — check team_mappings.py")
        starters = pd.merge(starters, home_parks.drop(columns=["home_team"]), on="game_pk", how="left")
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

        # Rolling team run averages — strictly prior dates only. Games sharing a
        # (team, game_date) inherit the window computed as of the first game on that date,
        # so doubleheader game 2 never leaks its own result into its features.
        games["game_date"] = pd.to_datetime(games["game_date"])
        starters["game_date"] = pd.to_datetime(starters["game_date"])

        home_runs = games[["game_pk", "game_date", "home_team", "home_score"]].rename(
            columns={"home_team": "team", "home_score": "runs"}
        )
        away_runs = games[["game_pk", "game_date", "away_team", "away_score"]].rename(
            columns={"away_team": "team", "away_score": "runs"}
        )
        team_games = pd.concat([home_runs, away_runs], ignore_index=True)
        team_games = team_games.sort_values(["team", "game_date", "game_pk"]).reset_index(drop=True)

        # Shift by 1 within each team so the window excludes the current row's runs
        team_games["runs_shifted"] = team_games.groupby("team")["runs"].shift(1)

        grp = team_games.groupby("team")["runs_shifted"]
        team_games["avg_last5"] = grp.rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
        team_games["avg_last10"] = grp.rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
        team_games["std_last5"] = grp.rolling(5, min_periods=2).std().reset_index(level=0, drop=True)

        # Doubleheader handling: all games on the same date share the FIRST game's rolling,
        # so neither game sees the other's runs (matches old `game_date < current_date`).
        for col in ["avg_last5", "avg_last10", "std_last5"]:
            team_games[col] = team_games.groupby(["team", "game_date"])[col].transform("first")

        starters = pd.merge(
            starters,
            team_games[["game_pk", "team", "avg_last5", "avg_last10", "std_last5"]],
            on=["game_pk", "team"],
            how="left",
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
    # No OOF predictions possible without CV — calibrator will be skipped downstream.
    return model, [], np.full(len(X), np.nan), {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1}


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

    # Report per-fold metrics with best params — also collect out-of-fold predictions
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


# Negative binomial dispersion parameter.  Variance = lambda + lambda^2 / r.
# r=6 is calibrated from MLB historical run distributions; it adds realistic
# overdispersion compared to plain Poisson, producing less extreme win probs.
NBINOM_R = 6.0


def compute_game_probs(lambda_home, lambda_away, total_line=None, spread_home=None,
                       max_runs=25, r=NBINOM_R):
    """Full game-level probability dict from the joint negative-binomial run distribution.

    Replaces the old magic `win_prob - 0.10` heuristic for run line and the
    `±1 run` heuristic for totals — both are now derived directly from the joint
    distribution, which naturally tightens in high-scoring environments and
    widens in low-scoring ones.

    Args:
        lambda_home: expected runs for home team
        lambda_away: expected runs for away team
        total_line: book total runs line (for over/under). None → skip.
        spread_home: signed spread from home team's perspective
            (e.g. -1.5 = home favored by 1.5, +1.5 = home underdog). None → skip.
        max_runs: truncation point for the run distribution (25 covers 99.9%+ of games)
        r: negative binomial dispersion parameter (higher r → less overdispersion)

    Returns dict with keys: p_home_win, p_away_win, p_home_cover, p_away_cover,
    p_over, p_under. Keys whose input is missing are set to None.
    Ties on the moneyline are allocated proportionally to expected runs
    (extra-innings approximation). Pushes on run line / totals are split 50/50.
    """
    runs = np.arange(max_runs + 1)
    h_probs = nbinom.pmf(runs, r, r / (r + lambda_home))
    a_probs = nbinom.pmf(runs, r, r / (r + lambda_away))
    # joint[h, a] = P(home scores h, away scores a)
    joint = np.outer(h_probs, a_probs)

    # Moneyline with proportional tie allocation
    p_home_outright = float(np.tril(joint, k=-1).sum())  # h > a
    p_away_outright = float(np.triu(joint, k=1).sum())   # a > h
    p_tie = float(np.trace(joint))
    total_lambda = lambda_home + lambda_away
    home_tie_share = lambda_home / total_lambda if total_lambda > 0 else 0.5
    p_home_win = p_home_outright + p_tie * home_tie_share
    p_away_win = p_away_outright + p_tie * (1 - home_tie_share)

    result = {
        "p_home_win": round(p_home_win, 4),
        "p_away_win": round(p_away_win, 4),
        "p_home_cover": None,
        "p_away_cover": None,
        "p_over": None,
        "p_under": None,
    }

    h_mat = runs[:, None]
    a_mat = runs[None, :]

    # Run line: home "covers" if home_runs + spread_home > away_runs. Pushes split 50/50.
    if spread_home is not None and not pd.isna(spread_home):
        margin = h_mat - a_mat
        p_home_strict = float(joint[margin > -spread_home].sum())
        p_push = float(joint[margin == -spread_home].sum())
        p_away_strict = float(joint[margin < -spread_home].sum())
        result["p_home_cover"] = round(p_home_strict + 0.5 * p_push, 4)
        result["p_away_cover"] = round(p_away_strict + 0.5 * p_push, 4)

    # Totals: over/under the book's total_line. Pushes split 50/50.
    if total_line is not None and not pd.isna(total_line):
        total_runs = h_mat + a_mat
        p_over_strict = float(joint[total_runs > total_line].sum())
        p_under_strict = float(joint[total_runs < total_line].sum())
        p_push_total = float(joint[total_runs == total_line].sum())
        result["p_over"] = round(p_over_strict + 0.5 * p_push_total, 4)
        result["p_under"] = round(p_under_strict + 0.5 * p_push_total, 4)

    return result


def win_prob(lambda_a, lambda_b, max_runs=15, r=NBINOM_R):
    """Backward-compatible: returns P(team A beats team B) as a float.

    Thin wrapper around compute_game_probs so existing callers (backtest.py,
    compute_predictions) continue to work. See compute_game_probs for the full
    joint-distribution output.
    """
    probs = compute_game_probs(lambda_a, lambda_b, max_runs=max_runs, r=r)
    return probs["p_home_win"]


# Backward-compatible alias used by backtest.py
poisson_win_prob = win_prob


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


# Dedupe warnings so a systematic issue doesn't spam stdout once per row.
_flag_warnings_seen: set[tuple[str, str]] = set()


def _warn_flag_error(fn_name: str, exc: Exception) -> None:
    key = (fn_name, f"{type(exc).__name__}: {exc}")
    if key not in _flag_warnings_seen:
        _flag_warnings_seen.add(key)
        print(f"  WARNING: {fn_name} raised {type(exc).__name__}: {exc} — returning 'No Play'")


def flag_ev(row, threshold=0.03):
    try:
        our_prob = american_to_prob(row["our_odds"])
        book_prob = american_to_prob(row["moneyline"])
        if pd.isna(book_prob):
            return "No Play"
        edge = our_prob - book_prob
        return row["team"] if edge >= threshold else "No Play"
    except Exception as e:
        _warn_flag_error("flag_ev", e)
        return "No Play"


def flag_runline_ev(row, threshold=0.03):
    """Flag a run line play if our model's cover probability (from the joint
    negative-binomial distribution, accounting for the book's actual spread)
    beats the book's implied cover probability by at least `threshold`.
    """
    try:
        book_prob = american_to_prob(row["spread_odds"])
        model_prob = row.get("p_cover")
        if pd.isna(book_prob) or pd.isna(model_prob):
            return "No Play"
        edge = model_prob - book_prob
        return row["team"] if edge >= threshold else "No Play"
    except Exception as e:
        _warn_flag_error("flag_runline_ev", e)
        return "No Play"


def flag_total_play(row, threshold=0.03):
    """Flag Over/Under based on joint-distribution probabilities vs book odds."""
    try:
        over_prob_book = american_to_prob(row.get("total_over_odds"))
        under_prob_book = american_to_prob(row.get("total_under_odds"))
        p_over = row.get("p_over")
        p_under = row.get("p_under")
        if pd.notna(p_over) and pd.notna(over_prob_book) and (p_over - over_prob_book) >= threshold:
            return "Over"
        if pd.notna(p_under) and pd.notna(under_prob_book) and (p_under - under_prob_book) >= threshold:
            return "Under"
        # Fallback when book over/under odds missing: use diff heuristic so the UI
        # still surfaces directional model disagreement with the line.
        if pd.isna(over_prob_book) and pd.isna(under_prob_book) and pd.notna(row.get("total_diff")):
            if row["total_diff"] >= 1:
                return "Over"
            if row["total_diff"] <= -1:
                return "Under"
        return "No Play"
    except Exception as e:
        _warn_flag_error("flag_total_play", e)
        return "No Play"


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
    """Fit isotonic regression on negative-binomial win probs vs actual outcomes.

    Args:
        df: DataFrame with win_prob and actual_runs columns (completed games only).

    Returns:
        Fitted IsotonicRegression, or None if fewer than 400 outcomes available.
    """
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
        print(f"  Only {len(outcomes)} outcomes — skipping calibration (need >= 400)")
        return None

    idx, y_true = zip(*outcomes)
    y_pred = df.loc[list(idx), "win_prob"].values
    y_true = np.array(y_true)

    calibrator = IsotonicRegression(y_min=0.05, y_max=0.95, out_of_bounds="clip")
    calibrator.fit(y_pred, y_true)
    print(f"  Isotonic calibrator fit on {len(y_true)} outcomes ({len(y_true) // 2} games)")
    return calibrator


def _get_table_columns(table_name):
    """Return the set of column names currently present in a table."""
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = :t"
            ),
            {"t": table_name},
        )
        return {row[0] for row in result}


def _apply_market_probs(df):
    """Compute p_cover / p_over / p_under per row from the joint NB distribution,
    using the book's actual total_line and spread (pulled from the row's odds columns).

    Expected columns on df: game_pk, team, is_home, xR, total, spread.
    Adds columns: p_cover, p_over, p_under.
    """
    df = df.copy()
    df["p_cover"] = np.nan
    df["p_over"] = np.nan
    df["p_under"] = np.nan

    for game_pk, group in df.groupby("game_pk"):
        if len(group) != 2:
            continue
        rows = group.sort_values("is_home", ascending=True)
        away_row = rows.iloc[0]
        home_row = rows.iloc[1]
        lambda_away = max(float(away_row["xR"]), 0.5)
        lambda_home = max(float(home_row["xR"]), 0.5)

        total_line = home_row.get("total")
        if pd.isna(total_line):
            total_line = None
        # spread is stored per-row, signed from that row's team perspective.
        # We want the spread from the HOME team's perspective.
        spread_home = home_row.get("spread")
        if pd.isna(spread_home):
            spread_home = None

        probs = compute_game_probs(
            lambda_home, lambda_away,
            total_line=total_line,
            spread_home=spread_home,
        )

        home_idx = rows.index[rows["is_home"] == 1]
        away_idx = rows.index[rows["is_home"] == 0]

        if probs["p_home_cover"] is not None:
            df.loc[home_idx, "p_cover"] = probs["p_home_cover"]
            df.loc[away_idx, "p_cover"] = probs["p_away_cover"]
        if probs["p_over"] is not None:
            df.loc[group.index, "p_over"] = probs["p_over"]
            df.loc[group.index, "p_under"] = probs["p_under"]

    return df


def _upsert_season_outputs(df):
    """Upsert into model_outputs_season using ON CONFLICT (game_pk, team).

    Filters df to only columns that currently exist in the target table so that
    new in-memory columns (added before a migration lands) don't break the insert.
    """
    existing_cols = _get_table_columns("model_outputs_season")
    cols = [c for c in df.columns if c in existing_cols]
    if not cols:
        print("  WARNING: no matching columns for model_outputs_season upsert")
        return
    placeholders = ", ".join([f":{c}" for c in cols])
    col_names = ", ".join(cols)
    update_set = ", ".join([f"{c} = EXCLUDED.{c}" for c in cols if c not in ("game_pk", "team")])

    sql = f"""
        INSERT INTO model_outputs_season ({col_names})
        VALUES ({placeholders})
        ON CONFLICT (game_pk, team) DO UPDATE SET {update_set}
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

    # Predict only on today's games — past games are training data only,
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
            print(f"Only {len(X)} training samples — too few for reliable model. Using league avg.")
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
        model.save_model("xgb_model.json")
        print("Model trained and saved to xgb_model.json")

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
                    joblib.dump(calibrator, "isotonic_calibrator.pkl")
                    print("  Calibrator saved to isotonic_calibrator.pkl")
            else:
                print(f"  Only {len(cal_df)} paired OOF rows — skipping calibration (need >= 800)")
        else:
            print("  No OOF predictions available — skipping calibration")

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
        # No trained model — use league-average xR and home-field win prob
        predict_df["is_home"] = predict_df["is_home"].astype(int)
        predict_df["xR"] = LEAGUE_AVG["avg_last5"]
        predict_df["win_prob"] = np.where(predict_df["is_home"] == 1, 0.54, 0.46)
        predict_df["our_odds"] = predict_df["win_prob"].apply(convert_to_odds)
        predictions = predict_df

    final_output = predictions.sort_values(
        ["game_pk", "is_home"], ascending=[True, True]
    ).reset_index(drop=True)[["game_pk", "game_date", "team", "starter", "xR", "win_prob", "our_odds", "is_home"]]

    # Merge odds data — include total_over_odds/total_under_odds so the totals EV
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
    final_output = _apply_market_probs(final_output)

    final_output["total_play"] = final_output.apply(flag_total_play, axis=1)
    final_output["ev_flag"] = final_output.apply(flag_ev, axis=1)
    final_output["run_line_ev_flag"] = final_output.apply(flag_runline_ev, axis=1)

    # --- Kelly sizing ---
    # Moneyline Kelly: model prob vs book moneyline
    ml_kelly = final_output.apply(
        lambda row: compute_kelly_row(row["win_prob"], row["moneyline"]), axis=1
    )
    final_output["kelly_full_ml"] = ml_kelly.apply(lambda x: x[0])
    final_output["kelly_quarter_ml"] = ml_kelly.apply(lambda x: x[1])

    # Run line Kelly: model p_cover vs book spread odds
    rl_kelly = final_output.apply(
        lambda row: compute_kelly_row(row.get("p_cover"), row.get("spread_odds")), axis=1
    )
    final_output["kelly_full_rl"] = rl_kelly.apply(lambda x: x[0])
    final_output["kelly_quarter_rl"] = rl_kelly.apply(lambda x: x[1])

    # Totals Kelly: model p_over vs book over odds, model p_under vs book under odds
    final_output["kelly_full_total"] = final_output.apply(
        lambda row: (
            kelly_fraction(row.get("p_over"), american_to_decimal(row.get("total_over_odds")))
            if row.get("total_play") == "Over"
            else kelly_fraction(row.get("p_under"), american_to_decimal(row.get("total_under_odds")))
            if row.get("total_play") == "Under"
            else 0.0
        ),
        axis=1,
    )
    final_output["kelly_quarter_total"] = final_output["kelly_full_total"].apply(
        lambda x: round(x * 0.25, 6) if pd.notna(x) else np.nan
    )

    # Confidence from quarter-Kelly (replaces ad-hoc edge differences)
    final_output["ml_confidence"] = final_output["kelly_quarter_ml"]
    final_output["run_line_confidence"] = final_output["kelly_quarter_rl"]

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
