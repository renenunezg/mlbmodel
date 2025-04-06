import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set the DATABASE_URL using the provided parameters
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/mlbmodel")
engine = create_engine(DATABASE_URL)

def load_training_data():
    starters = pd.read_sql_table("probable_starters", con=engine)
    starters = starters.rename(columns={"team_abbr": "team"})
    from team_mappings import TEAM_NAME_MAP
    starters["team"] = starters["team"].map(TEAM_NAME_MAP).fillna(starters["team"])

    # Merge in pitcher xFIP from starting_pitchers
    sp_stats = pd.read_sql_table("starting_pitchers", con=engine)
    starters = pd.merge(starters, sp_stats[["pitcher", "xFIP"]], how="left", left_on="pitcher_name", right_on="pitcher")
    
    # Merge in bullpen xFIP
    bullpen = pd.read_sql_table("bullpen", con=engine)
    starters = pd.merge(starters, bullpen[["team", "xFIP"]], on="team", how="left", suffixes=("", "_bullpen"))
    starters = starters.rename(columns={"xFIP_bullpen": "xFIP_bullpen"})
    
    # Load batting splits
    vs_r = pd.read_sql_table("team_batting_vs_rhp", con=engine)[["team", "wRC+", "wOBA"]].rename(columns={"wRC+": "wRC+_vs_r", "wOBA": "wOBA_vs_r"})
    vs_l = pd.read_sql_table("team_batting_vs_lhp", con=engine)[["team", "wRC+", "wOBA"]].rename(columns={"wRC+": "wRC+_vs_l", "wOBA": "wOBA_vs_l"})
    
    # Merge batting splits for both RHP and LHP
    starters = pd.merge(starters, vs_r, on="team", how="left")
    starters = pd.merge(starters, vs_l, on="team", how="left")
    
    # Load runs per game
    runs = pd.read_sql_table("runs_per_game", con=engine)[["team", "home", "away"]]
    starters = pd.merge(starters, runs, on="team", how="left")
    
    # Self merge to get opponent handedness
    opp = starters[["game_id", "team", "handedness"]].rename(columns={"team": "opp_team", "handedness": "opp_handedness"})
    starters = pd.merge(starters, opp, on="game_id")
    starters = starters[starters["team"] != starters["opp_team"]]
    
    # Select correct wRC+ based on opponent handedness
    starters["batting_wrc_plus"] = np.where(
        starters["opp_handedness"] == "R", starters["wRC+_vs_r"], starters["wRC+_vs_l"]
    )
    
    # Load park factors
    parks = pd.read_sql_table("park_factors", con=engine)
    home_teams = starters.groupby("game_id", group_keys=False).apply(lambda x: x.iloc[1:2])[["game_id", "team"]].rename(columns={"team": "home_team"})
    starters = pd.merge(starters, home_teams, on="game_id", how="left")
    home_parks = pd.merge(home_teams, parks, left_on="home_team", right_on="team", how="left")[["game_id", "park_factor"]]
    starters = pd.merge(starters, home_parks, on="game_id", how="left")
    
    # Filter for games with exactly 2 teams
    starters = starters.groupby("game_id").filter(lambda x: len(x) == 2)
    
    # Sort values and compute expected runs
    starters = starters.sort_values(["game_id", "team"]).reset_index(drop=True)
    starters["expected_runs"] = np.where(
        starters["team"] == starters["home_team"],
        starters["home"],
        starters["away"]
    )
    
    return starters[["game_id", "team", "pitcher_name", "xFIP", "opp_handedness", "park_factor", "batting_wrc_plus", "xFIP_bullpen", "expected_runs"]].rename(columns={"pitcher_name": "starter"})

def preprocess_data(df):
    """
    Convert columns to numeric.
    """
    df['batting_wrc_plus'] = pd.to_numeric(df['batting_wrc_plus'], errors='coerce')
    df['xFIP'] = pd.to_numeric(df['xFIP'], errors='coerce')
    df['park_factor'] = pd.to_numeric(df['park_factor'], errors='coerce')
    df['xFIP_bullpen'] = pd.to_numeric(df['xFIP_bullpen'], errors='coerce')
    return df

def train_model(X, y):
    """
    Train an XGBoost regressor to predict expected runs.
    """
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X, y)
    return model

def convert_to_odds(p):
    """
    Convert win probability to American odds.
    """
    if p < 0.5:
        return round(((1 - p) / p) * 100)
    elif p > 0.5:
        return round(- (p / (1 - p)) * 100) if p < 1 else -1000
    else:
        return 100  # even odds

def compute_predictions(df, model):
    """
    Use the trained model to predict expected runs (xR) for each team,
    then compute win probabilities and simulated American odds for each game.
    """
    X = df[['batting_wrc_plus', 'park_factor', 'xFIP_bullpen']].copy()
    X['adj_pitch_factor'] = (df['xFIP'] * df['park_factor']) / df['park_factor'].mean()
    df['xR'] = model.predict(X[['batting_wrc_plus', 'adj_pitch_factor', 'xFIP_bullpen']])
    df['xR'] = df['xR'].round(2)
    
    def compute_game_stats(group):
        total_xR = group['xR'].sum()
        group['win_prob'] = group['xR'] / total_xR if total_xR > 0 else 1.0 / len(group)
        group['win_prob'] = group['win_prob'].clip(upper=0.999, lower=0.001)
        group['win_prob'] = group['win_prob'].round(2)
        group['our_odds'] = group['win_prob'].apply(convert_to_odds)
        return group

    df = df.groupby('game_id', group_keys=False).apply(compute_game_stats).reset_index(drop=True)
    df['team_count'] = df.groupby('game_id')['team'].transform('count')
    return df

def main():
    df = load_training_data()
    df = preprocess_data(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["expected_runs"])
    
    X = df[['batting_wrc_plus', 'park_factor', 'xFIP_bullpen']].copy()
    X['adj_pitch_factor'] = (df['xFIP'] * df['park_factor']) / df['park_factor'].mean()
    X = X[['batting_wrc_plus', 'adj_pitch_factor', 'xFIP_bullpen']]
    y = pd.to_numeric(df['expected_runs'], errors='coerce')
    
    model = train_model(X, y)
    model.save_model("xgb_model.json")
    print("Model trained and saved to xgb_model.json")
    
    print("\nSanity check on predicted xR:")
    print(df['expected_runs'].describe())
    
    predictions = compute_predictions(df, model)
    final_output = predictions.sort_values(['game_id', 'team']).reset_index(drop=True)[['game_id', 'team', 'starter', 'xR', 'win_prob', 'our_odds']]
    
    print("\nFinal simulated predictions:")
    print(final_output)
    
    odds_df = pd.read_sql_table("odds", con=engine)
    odds_df = odds_df.drop_duplicates(subset=["team"], keep="first")
    final_output = pd.merge(final_output, odds_df[["team", "bet365_ml", "total", "run_line"]], on="team", how="left")
    print("\nMerged predictions with odds:")
    print(final_output)

if __name__ == "__main__":
    main()