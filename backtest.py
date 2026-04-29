"""
Walk-forward backtesting for the MLB expected runs model.

Fetches historical data, runs walk-forward validation, and reports metrics.
All in-memory — no production DB writes.

Usage:
    python backtest.py --season 2025
    python backtest.py --season 2024 --window-days 14
"""

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from backend.data.mlb_api import fetch_schedule_range
from backend.data.fangraphs import (
    _compute_pitcher_stats,
    _identify_starters,
)
from backend.data.savant import _static_park_factors
from backend.model import FEATURE_COLS
from backend.features import LEAGUE_AVG
from backend.simulation import poisson_win_prob
from backend.team_mappings import normalize_team

CACHE_DIR = Path("backtest_cache")


def fetch_season_statcast(year: int) -> pd.DataFrame:
    """Fetch full-season Statcast data, cached to parquet."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"statcast_{year}.parquet"

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  Cached Statcast {year}: {len(df)} pitches")
        return df

    from pybaseball import statcast

    print(f"  Fetching {year} Statcast data (this takes ~20-30 min)...")
    all_chunks = []
    for month in range(3, 11):
        start = date(year, month, 1)
        end = date(year, month + 1, 1) - timedelta(days=1) if month < 10 else date(year, 10, 31)
        try:
            chunk = statcast(start_dt=str(start), end_dt=str(end))
            if not chunk.empty:
                all_chunks.append(chunk)
                print(f"    {start.strftime('%b')}: {len(chunk)} pitches")
        except Exception as e:
            print(f"    {start.strftime('%b')}: failed ({e})")

    if not all_chunks:
        return pd.DataFrame()

    df = pd.concat(all_chunks, ignore_index=True)
    if "game_type" in df.columns:
        df = df[df["game_type"] == "R"]

    df.to_parquet(cache_path, index=False)
    print(f"  Cached {len(df)} pitches to {cache_path}")
    return df


def compute_features_for_window(
    games_df: pd.DataFrame,
    statcast_df: pd.DataFrame,
    park_factors: dict,
    cutoff_date: str,
) -> pd.DataFrame:
    """Compute model features for games, using only data available before cutoff_date.

    Returns DataFrame with one row per team per game, with all 11 features + actual_runs.
    """
    cutoff = pd.to_datetime(cutoff_date)
    games_df = games_df.copy()
    games_df["game_date"] = pd.to_datetime(games_df["game_date"])

    # Split into training data (before cutoff) and prediction targets (at/after cutoff)
    completed = games_df[
        (games_df["game_date"] < cutoff) &
        (games_df["status"] == "Final") &
        (games_df["home_score"].notna())
    ]

    # Filter Statcast to before cutoff
    sc = statcast_df.copy()
    sc["game_date"] = pd.to_datetime(sc["game_date"])
    sc_before = sc[sc["game_date"] < cutoff]

    if sc_before.empty or completed.empty:
        return pd.DataFrame()

    # Compute pitcher stats from Statcast up to cutoff
    starter_ids = _identify_starters(sc_before)
    pitcher_stats = _compute_pitcher_stats(sc_before, pitcher_ids=starter_ids)

    # Compute bullpen stats
    reliever_ids = set(sc_before["pitcher"].unique()) - starter_ids
    reliever_stats = _compute_pitcher_stats(sc_before, pitcher_ids=reliever_ids)

    bullpen_stats = pd.DataFrame()
    if not reliever_stats.empty:
        bullpen_stats = reliever_stats.groupby("team").apply(
            lambda g: pd.Series({
                "xfip_bullpen": np.average(g["xfip"].dropna(), weights=g.loc[g["xfip"].notna(), "ip"])
                if g["xfip"].notna().any() else np.nan,
                "bullpen_k_9": np.average(g["k_9"].dropna(), weights=g.loc[g["k_9"].notna(), "ip"])
                if g["k_9"].notna().any() else np.nan,
            })
        ).reset_index()

    # Compute batting stats by split
    pa_df = sc_before[sc_before["events"].notna()].copy()
    batting_splits = {}
    if "p_throws" in pa_df.columns:
        for split_name, hand in [("vs_rhp", "R"), ("vs_lhp", "L")]:
            split_df = pa_df[pa_df["p_throws"] == hand]
            for side, topbot in [("home", "Bot"), ("away", "Top")]:
                team_col = f"{side}_team"
                if team_col not in split_df.columns:
                    continue
                side_df = split_df[split_df["inning_topbot"] == topbot]
                for team, g in side_df.groupby(team_col):
                    team = normalize_team(team)
                    pa = len(g)
                    ab = pa - g["events"].isin(["walk", "intent_walk", "hit_by_pitch", "sac_fly", "sac_bunt", "catcher_interf"]).sum()
                    hits = g["events"].isin(["single", "double", "triple", "home_run"]).sum()
                    doubles = (g["events"] == "double").sum()
                    triples = (g["events"] == "triple").sum()
                    hr = (g["events"] == "home_run").sum()
                    bb = g["events"].isin(["walk", "intent_walk"]).sum()
                    hbp = (g["events"] == "hit_by_pitch").sum()
                    so = (g["events"] == "strikeout").sum()
                    sf = (g["events"] == "sac_fly").sum()

                    avg = hits / ab if ab > 0 else 0
                    obp = (hits + bb + hbp) / (ab + bb + hbp + sf) if (ab + bb + hbp + sf) > 0 else 0
                    slg = (hits - doubles - triples - hr + 2 * doubles + 3 * triples + 4 * hr) / ab if ab > 0 else 0
                    ops = obp + slg
                    iso = slg - avg
                    k_pct = (so / pa * 100) if pa > 0 else 0

                    key = (team, split_name)
                    if key not in batting_splits:
                        batting_splits[key] = {"pa": 0, "ops_sum": 0, "iso_sum": 0, "k_pct_sum": 0}
                    batting_splits[key]["pa"] += pa
                    batting_splits[key]["ops_sum"] += ops * pa
                    batting_splits[key]["iso_sum"] += iso * pa
                    batting_splits[key]["k_pct_sum"] += k_pct * pa

    batting_df_rows = []
    for (team, split), v in batting_splits.items():
        if v["pa"] > 0:
            batting_df_rows.append({
                "team": team, "split": split,
                "ops": v["ops_sum"] / v["pa"],
                "iso": v["iso_sum"] / v["pa"],
                "k_pct": v["k_pct_sum"] / v["pa"],
            })
    batting_df = pd.DataFrame(batting_df_rows)

    # Rolling averages from completed games
    home_runs = completed[["game_date", "home_team", "home_score"]].rename(
        columns={"home_team": "team", "home_score": "runs"}
    )
    away_runs = completed[["game_date", "away_team", "away_score"]].rename(
        columns={"away_team": "team", "away_score": "runs"}
    )
    team_games = pd.concat([home_runs, away_runs]).sort_values(["team", "game_date"])

    def get_rolling(team, game_date):
        past = team_games[(team_games["team"] == team) & (team_games["game_date"] < game_date)]
        last5 = past.tail(5)["runs"]
        last10 = past.tail(10)["runs"]
        return (
            last5.mean() if not last5.empty else np.nan,
            last10.mean() if not last10.empty else np.nan,
        )

    # Build feature rows for ALL completed games (for training)
    rows = []
    all_games = games_df[games_df["status"] == "Final"].copy()

    for _, game in all_games.iterrows():
        gd = game["game_date"]
        gpk = game["game_pk"]

        for side in ["home", "away"]:
            team = game[f"{side}_team"]
            pitcher_id = game.get(f"{side}_pitcher_id")
            pitcher_name = game.get(f"{side}_pitcher_name")
            opp_side = "away" if side == "home" else "home"
            opp_hand = game.get(f"{opp_side}_pitcher_hand")
            is_home = 1 if side == "home" else 0
            actual = game[f"{side}_score"]

            # Pitcher stats
            xfip = LEAGUE_AVG["xfip"]
            whip = LEAGUE_AVG["whip"]  # stored as "whip" in LEAGUE_AVG but mapped to starter_whip
            if not pitcher_stats.empty and pitcher_id is not None:
                ps = pitcher_stats[pitcher_stats["pitcher_id"] == pitcher_id]
                if not ps.empty:
                    xfip = ps.iloc[0]["xfip"] if pd.notna(ps.iloc[0]["xfip"]) else xfip
                    whip = ps.iloc[0]["whip"] if pd.notna(ps.iloc[0]["whip"]) else whip

            # Bullpen stats
            xfip_bp = LEAGUE_AVG["xfip_bullpen"]
            bp_k9 = LEAGUE_AVG["bullpen_k_9"]
            if not bullpen_stats.empty:
                bs = bullpen_stats[bullpen_stats["team"] == team]
                if not bs.empty:
                    xfip_bp = bs.iloc[0]["xfip_bullpen"] if pd.notna(bs.iloc[0]["xfip_bullpen"]) else xfip_bp
                    bp_k9 = bs.iloc[0]["bullpen_k_9"] if pd.notna(bs.iloc[0]["bullpen_k_9"]) else bp_k9

            # Batting stats (vs opponent handedness)
            split = "vs_rhp" if opp_hand == "R" else "vs_lhp"
            b_ops = LEAGUE_AVG["batting_ops"]
            b_iso = LEAGUE_AVG["batting_iso"]
            b_kpct = LEAGUE_AVG["batting_k_pct"]
            if not batting_df.empty:
                bt = batting_df[(batting_df["team"] == team) & (batting_df["split"] == split)]
                if not bt.empty:
                    b_ops = bt.iloc[0]["ops"] if pd.notna(bt.iloc[0]["ops"]) else b_ops
                    b_iso = bt.iloc[0]["iso"] if pd.notna(bt.iloc[0]["iso"]) else b_iso
                    b_kpct = bt.iloc[0]["k_pct"] if pd.notna(bt.iloc[0]["k_pct"]) else b_kpct

            # Rolling averages
            avg5, avg10 = get_rolling(team, gd)
            if pd.isna(avg5):
                avg5 = LEAGUE_AVG["avg_last5"]
            if pd.isna(avg10):
                avg10 = LEAGUE_AVG["avg_last10"]

            # Park factor
            pf = park_factors.get(game.get("home_team", ""), 100)

            rows.append({
                "game_pk": gpk,
                "game_date": gd,
                "team": team,
                "starter": pitcher_name,
                "is_home": is_home,
                "xfip": xfip,
                "starter_whip": whip,
                "xfip_bullpen": xfip_bp,
                "bullpen_k_9": bp_k9,
                "batting_ops": b_ops,
                "batting_iso": b_iso,
                "batting_k_pct": b_kpct,
                "avg_last5": avg5,
                "avg_last10": avg10,
                "park_factor": pf,
                "actual_runs": actual,
            })

    return pd.DataFrame(rows)


def run_backtest(season: int, window_days: int = 7):
    """Run walk-forward backtest for a season."""
    t0 = time.time()
    print(f"\nBacktest — {season} season (window={window_days} days)")
    print("=" * 60)

    # 1. Fetch games
    print("\n>> Fetching schedule...")
    games = fetch_schedule_range(date(season, 3, 20), date(season, 10, 31))
    games = games[games["status"] == "Final"]
    print(f"  {len(games)} completed games")

    if games.empty:
        print("No completed games found. Exiting.")
        return

    # 2. Fetch Statcast
    print("\n>> Fetching Statcast data...")
    statcast = fetch_season_statcast(season)
    if statcast.empty:
        print("No Statcast data. Exiting.")
        return

    # 3. Park factors (static fallback)
    pf_df = _static_park_factors(season)
    park_factors = dict(zip(pf_df["team"], pf_df["park_factor"]))

    # 4. Compute features for entire season
    print("\n>> Computing features...")
    season_end = date(season, 10, 1)
    all_features = compute_features_for_window(games, statcast, park_factors, str(season_end))
    print(f"  {len(all_features)} feature rows for {all_features['game_pk'].nunique()} games")

    if all_features.empty:
        print("No features computed. Exiting.")
        return

    all_features["game_date"] = pd.to_datetime(all_features["game_date"])

    # 5. Walk-forward validation
    print("\n>> Walk-forward validation...")
    # Start predictions after ~30 days of data (need training samples)
    first_date = all_features["game_date"].min()
    start_pred = first_date + timedelta(days=45)
    end_date = all_features["game_date"].max()

    current = start_pred
    all_predictions = []
    window_metrics = []

    while current <= end_date:
        window_end = current + timedelta(days=window_days)

        # Training: all data before current
        train = all_features[all_features["game_date"] < current].copy()
        # Test: games in [current, window_end)
        test = all_features[
            (all_features["game_date"] >= current) &
            (all_features["game_date"] < window_end)
        ].copy()

        train = train.dropna(subset=["actual_runs"])
        test = test.dropna(subset=["actual_runs"])

        if len(train) < 20 or test.empty:
            current = window_end
            continue

        # Train model
        X_train = train[FEATURE_COLS].copy()
        y_train = train["actual_runs"].astype(float)
        X_test = test[FEATURE_COLS].copy()
        y_test = test["actual_runs"].astype(float)

        # Fill NaN with league averages
        fill_map = {**LEAGUE_AVG, "starter_whip": LEAGUE_AVG.get("whip", 1.30)}
        for col, fallback in fill_map.items():
            if col in X_train.columns:
                X_train[col] = X_train[col].fillna(fallback)
            if col in X_test.columns:
                X_test[col] = X_test[col].fillna(fallback)

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            min_child_weight=3,
            random_state=42,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        test = test.copy()
        test["predicted_runs"] = preds

        # Compute win probabilities per game
        test["predicted_win"] = False
        test["actual_win"] = False
        for gpk, group in test.groupby("game_pk"):
            if len(group) != 2:
                continue
            rows = group.sort_values("is_home")
            away_xr = max(rows.iloc[0]["predicted_runs"], 0.5)
            home_xr = max(rows.iloc[1]["predicted_runs"], 0.5)
            p_home = poisson_win_prob(home_xr, away_xr)

            home_idx = rows.index[rows["is_home"] == 1]
            away_idx = rows.index[rows["is_home"] == 0]
            test.loc[home_idx, "win_prob"] = p_home
            test.loc[away_idx, "win_prob"] = 1 - p_home

            actual_home = rows.iloc[1]["actual_runs"]
            actual_away = rows.iloc[0]["actual_runs"]
            test.loc[home_idx, "actual_win"] = actual_home > actual_away
            test.loc[away_idx, "actual_win"] = actual_away > actual_home
            test.loc[home_idx, "predicted_win"] = p_home > 0.5
            test.loc[away_idx, "predicted_win"] = p_home <= 0.5

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        win_acc = (test["predicted_win"] == test["actual_win"]).mean()

        window_metrics.append({
            "window_start": current.strftime("%Y-%m-%d"),
            "n_games": test["game_pk"].nunique(),
            "n_rows": len(test),
            "train_size": len(train),
            "mae": mae,
            "rmse": rmse,
            "win_accuracy": win_acc,
        })

        all_predictions.append(test)
        current = window_end

    if not window_metrics:
        print("No evaluation windows completed. Exiting.")
        return

    # 6. Report results
    metrics_df = pd.DataFrame(window_metrics)
    preds_df = pd.concat(all_predictions, ignore_index=True)

    print(f"\n{'=' * 60}")
    print(f"BACKTEST RESULTS — {season}")
    print(f"{'=' * 60}")
    print(f"\nWindows: {len(metrics_df)}")
    print(f"Total games evaluated: {preds_df['game_pk'].nunique()}")
    print(f"Total predictions: {len(preds_df)}")
    print(f"\nOverall Metrics:")
    print(f"  MAE:          {metrics_df['mae'].mean():.3f}")
    print(f"  RMSE:         {metrics_df['rmse'].mean():.3f}")
    print(f"  Win Accuracy: {metrics_df['win_accuracy'].mean():.1%}")

    # Calibration: binned win_prob vs actual win rate
    if "win_prob" in preds_df.columns:
        preds_df["wp_bin"] = pd.cut(preds_df["win_prob"], bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0])
        cal = preds_df.groupby("wp_bin", observed=True).agg(
            count=("actual_win", "count"),
            actual_rate=("actual_win", "mean"),
            avg_pred=("win_prob", "mean"),
        ).reset_index()
        print(f"\nCalibration:")
        print(f"  {'Bin':<12} {'Count':>6} {'Predicted':>10} {'Actual':>10}")
        for _, r in cal.iterrows():
            print(f"  {str(r['wp_bin']):<12} {r['count']:>6} {r['avg_pred']:>10.1%} {r['actual_rate']:>10.1%}")

    # Per-window summary
    print(f"\nPer-Window Metrics:")
    print(f"  {'Window':<12} {'Games':>6} {'Train':>6} {'MAE':>6} {'RMSE':>6} {'Win%':>6}")
    for _, m in metrics_df.iterrows():
        print(f"  {m['window_start']:<12} {m['n_games']:>6} {m['train_size']:>6} "
              f"{m['mae']:>6.3f} {m['rmse']:>6.3f} {m['win_accuracy']:>6.1%}")

    # Save results
    results_path = f"backtest_results_{season}.csv"
    preds_df.to_csv(results_path, index=False)
    print(f"\nRaw predictions saved to {results_path}")

    elapsed = time.time() - t0
    print(f"\nBacktest completed in {elapsed:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLB model walk-forward backtest")
    parser.add_argument("--season", type=int, default=2025, help="Season year to backtest")
    parser.add_argument("--window-days", type=int, default=7, help="Days per evaluation window")
    args = parser.parse_args()
    run_backtest(args.season, args.window_days)
