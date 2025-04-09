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
    
    # Load batting splits
    vs_r = pd.read_sql_table("team_batting_vs_rhp", con=engine)[["team", "wRC+", "wOBA"]].rename(columns={"wRC+": "wRC+_vs_r", "wOBA": "wOBA_vs_r"})
    vs_l = pd.read_sql_table("team_batting_vs_lhp", con=engine)[["team", "wRC+", "wOBA"]].rename(columns={"wRC+": "wRC+_vs_l", "wOBA": "wOBA_vs_l"})
    starters = pd.merge(starters, vs_r, on="team", how="left")
    starters = pd.merge(starters, vs_l, on="team", how="left")
    
    # Self merge to get opponent handedness
    opp = starters[["game_id", "team", "handedness"]].rename(columns={"team": "opp_team", "handedness": "opp_handedness"})
    starters = pd.merge(starters, opp, on="game_id")
    starters = starters[starters["team"] != starters["opp_team"]]
    
    # Select correct batting wRC+ based on opponent handedness
    starters["batting_wrc_plus"] = np.where(
        starters["opp_handedness"] == "R", starters["wRC+_vs_r"], starters["wRC+_vs_l"]
    )
    
    # Load park factors and merge them in based on home team
    parks = pd.read_sql_table("park_factors", con=engine)
    home_teams = starters[starters["is_home"] == True][["game_id", "team"]].rename(columns={"team": "home_team"})
    starters = pd.merge(starters, home_teams, on="game_id", how="left")
    home_parks = pd.merge(home_teams, parks, left_on="home_team", right_on="team", how="left")[["game_id", "park_factor"]]
    starters = pd.merge(starters, home_parks, on="game_id", how="left")
    
    # Load actual game results, filtering out games without scores
    game_results = pd.read_sql_table("game_results", con=engine)
    game_results = game_results.dropna(subset=["away_score", "home_score"])
    
    # Merge game results into starters to obtain game date and scores
    starters = pd.merge(starters, game_results[['game_id', 'away_team', 'away_score', 'home_team', 'home_score']], on="game_id", how="left")
    
    # Set expected_runs based on whether the team is home or away
    starters["expected_runs"] = np.where(
        starters["is_home"] == True,
        starters["home_score"],
        starters["away_score"]
    )
    
    # Compute rolling averages (last 5 and last 10 games) for each team using historical game results
    game_results['date'] = pd.to_datetime(game_results['date'])
    starters['date'] = pd.to_datetime(starters['date'])
    
    home_games = game_results[['date', 'home_team', 'home_score']].rename(columns={'home_team': 'team', 'home_score': 'runs'})
    away_games = game_results[['date', 'away_team', 'away_score']].rename(columns={'away_team': 'team', 'away_score': 'runs'})
    team_games = pd.concat([home_games, away_games]).sort_values(['team', 'date']).reset_index(drop=True)
    
    def get_rolling_averages(team, current_date):
        games = team_games[(team_games['team'] == team) & (team_games['date'] < current_date)]
        last5 = games.tail(5)['runs']
        return pd.Series({
            'avg_last5': last5.mean() if not last5.empty else np.nan,
            'avg_last10': games.tail(10)['runs'].mean() if not games.empty else np.nan,
            'std_last5': last5.std() if not last5.empty else np.nan
        })
    
    starters[['avg_last5', 'avg_last10', 'std_last5']] = starters.apply(lambda row: get_rolling_averages(row['team'], row['date']), axis=1)
    
    # Filter for games with exactly 2 teams
    starters = starters.groupby("game_id").filter(lambda x: len(x) == 2)
    
    # Sort values by game_id and is_home to preserve correct home/away order
    starters = starters.sort_values(["game_id", "is_home"]).reset_index(drop=True)
    
    return starters[["game_id", "date", "team", "pitcher_name", "is_home", "xFIP", "batting_wrc_plus", "park_factor", "xFIP_bullpen", "expected_runs", "avg_last5", "avg_last10", "std_last5"]].rename(columns={"pitcher_name": "starter"})

def preprocess_data(df):
    """
    Convert columns to numeric.
    """
    df['xFIP'] = pd.to_numeric(df['xFIP'], errors='coerce')
    df['xFIP_bullpen'] = pd.to_numeric(df['xFIP_bullpen'], errors='coerce')
    df['avg_last5'] = pd.to_numeric(df['avg_last5'], errors='coerce')
    df['avg_last10'] = pd.to_numeric(df['avg_last10'], errors='coerce')
    df['std_last5'] = pd.to_numeric(df['std_last5'], errors='coerce')
    df['batting_wrc_plus'] = pd.to_numeric(df['batting_wrc_plus'], errors='coerce')
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

def american_to_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def flag_ev(row, threshold=0.03):
    try:
        our_prob = american_to_prob(row['our_odds'])
        book_prob = american_to_prob(pd.to_numeric(row['bet365_ml'], errors='coerce'))
        edge = our_prob - book_prob
        return row['team'] if edge >= threshold else "No Play"
    except:
        return "No Play"

def flag_runline_ev(row, threshold=0.03):
    try:
        book_prob = american_to_prob(row['run_line_odds'])
        model_prob = max(row['win_prob'] - 0.10, 0)  # crude discount to estimate cover chance
        edge = model_prob - book_prob
        return row['team'] if edge >= threshold else "No Play"
    except:
        return "No Play"

def compute_predictions(df, model):
    """
    Use the trained model to predict expected runs (xR) for each team,
    then compute win probabilities and simulated American odds for each game.
    """
    X = df[['xFIP_bullpen', 'avg_last5', 'avg_last10', 'batting_wrc_plus']].copy()
    df["adj_pitch_factor"] = df["xFIP"] * (df["park_factor"] / 100)
    X['adj_pitch_factor'] = df['xFIP'] * (df['park_factor'] / 100)
    df['xR'] = model.predict(X[['adj_pitch_factor', 'xFIP_bullpen', 'avg_last5', 'avg_last10', 'batting_wrc_plus']])
    df['xR'] = df['xR'].round(2)
    
    def compute_game_stats(group):
        total_xR = group['xR'].sum()
        group['win_prob'] = group['xR'] / total_xR if total_xR > 0 else 1.0 / len(group)
        group['win_prob'] = group['win_prob'].clip(upper=0.999, lower=0.001)
        group['win_prob'] = group['win_prob'].round(2)
        group['our_odds'] = group['win_prob'].apply(convert_to_odds)
        return group

    df = df.groupby('game_id', group_keys=False).apply(compute_game_stats).reset_index(drop=True)
    return df

def main():
    df = load_training_data()
    df = preprocess_data(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["expected_runs"])
    
    df["adj_pitch_factor"] = df["xFIP"] * (df["park_factor"] / 100)
    X = df[['adj_pitch_factor', 'xFIP_bullpen', 'avg_last5', 'avg_last10', 'batting_wrc_plus']].copy()
    y = pd.to_numeric(df['expected_runs'], errors='coerce')
    
    model = train_model(X, y)
    model.save_model("xgb_model.json")
    print("Model trained and saved to xgb_model.json")
    
    print("\nSanity check on predicted xR:")
    print(df['expected_runs'].describe())
    
    predictions = compute_predictions(df, model)
    final_output = predictions.sort_values(['game_id', 'is_home'], ascending=[True, True]).reset_index(drop=True)[['game_id', 'date', 'team', 'starter', 'xR', 'win_prob', 'our_odds']]
    
    odds_df = pd.read_sql_table("odds", con=engine)
    odds_df = odds_df.drop_duplicates(subset=["team"], keep="first")
    final_output = pd.merge(final_output, odds_df[["team", "bet365_ml", "total", "run_line", "run_line_raw"]], on="team", how="left")
    
    final_output["run_line_odds"] = final_output["run_line_raw"].str.extract(r'[\+\-]\d+\.\d+\s+([\+\-]?\d+)')
    final_output["run_line_odds"] = pd.to_numeric(final_output["run_line_odds"], errors="coerce")

    # Compute additional columns
    our_totals = final_output.groupby('game_id')['xR'].sum().round(2).rename("our_total")
    final_output = final_output.merge(our_totals, on='game_id', how='left')
    final_output['total'] = pd.to_numeric(final_output['total'], errors='coerce')
    final_output['total_diff'] = (final_output['our_total'] - final_output['total']).round(2)

    def flag_total_play(row):
        if row['total_diff'] >= 1:
            return 'Over'
        elif row['total_diff'] <= -1:
            return 'Under'
        else:
            return 'No Play'

    final_output['total_play'] = final_output.apply(flag_total_play, axis=1)
    final_output['ev_flag'] = final_output.apply(flag_ev, axis=1)
    final_output['run_line_ev_flag'] = final_output.apply(flag_runline_ev, axis=1)

    final_output['ml_confidence'] = final_output.apply(
        lambda row: round(row['win_prob'] - american_to_prob(pd.to_numeric(row['bet365_ml'], errors='coerce')), 3),
        axis=1
    )

    final_output['run_line_confidence'] = final_output.apply(
        lambda row: round(max(row['win_prob'] - 0.10, 0) - american_to_prob(row['run_line_odds']), 3),
        axis=1
    )

    final_output["high_variance_flag"] = predictions["std_last5"].apply(lambda x: "Yes" if x > 4.0 else "No")

    print("\nFinal output with flags:")
    print(final_output)
    
    # Insert into model_outputs table
    final_output = final_output.rename(columns={"xR": "expected_runs"})
    final_output.to_sql("model_outputs", con=engine, if_exists="replace", index=False)
    
    # Also append to model_outputs_season for evaluation
    # Round float columns to 2 decimals using the same method as model_outputs table
    float_cols_to_format = ["expected_runs", "win_prob", "our_total"]
    for col in float_cols_to_format:
        if col in final_output.columns:
            final_output[col] = final_output[col].map(lambda x: round(x, 2) if pd.notnull(x) else x)

    final_output = final_output.round(2)
    final_output.to_sql("model_outputs_season", con=engine, if_exists="append", index=False)

    print("\n✅ model_outputs table updated.")

if __name__ == "__main__":
    main()