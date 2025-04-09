import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# Load model outputs and game results
model_df = pd.read_sql_table("model_outputs", con=engine)
results_df = pd.read_sql_table("game_results", con=engine)

# Reshape game_results into one row per team
home_df = results_df.rename(columns={"home_team": "team", "home_score": "actual_runs"})[["date", "game_id", "team", "actual_runs"]]
away_df = results_df.rename(columns={"away_team": "team", "away_score": "actual_runs"})[["date", "game_id", "team", "actual_runs"]]
all_results = pd.concat([home_df, away_df], ignore_index=True)

# Merge model predictions with actual results
eval_df = pd.merge(model_df, all_results, on=["game_id", "team"], how="inner")

# Compute evaluation metrics
eval_df["abs_error"] = abs(eval_df["xR"] - eval_df["actual_runs"])
eval_df["squared_error"] = (eval_df["xR"] - eval_df["actual_runs"]) ** 2
eval_df["sMAPE"] = 100 * abs(eval_df["xR"] - eval_df["actual_runs"]) / ((abs(eval_df["xR"]) + abs(eval_df["actual_runs"])) / 2)

# Print evaluation summary
print("\n📊 Model Evaluation Metrics:")
print(f"MAE: {eval_df['abs_error'].mean():.3f}")
print(f"MSE: {eval_df['squared_error'].mean():.3f}")
print(f"sMAPE: {eval_df['sMAPE'].mean():.2f}%")

# Optional: show a few rows
print("\n🔍 Sample comparison:")
print(eval_df[["team", "xR", "actual_runs", "abs_error", "sMAPE"]].head())
