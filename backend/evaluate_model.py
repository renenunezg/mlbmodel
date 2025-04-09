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

results_df = results_df.dropna(subset=["home_score", "away_score"])

# Reshape game_results into one row per team
home_df = results_df.rename(columns={"home_team": "team", "home_score": "actual_runs"})[["date", "game_id", "team", "actual_runs"]]
away_df = results_df.rename(columns={"away_team": "team", "away_score": "actual_runs"})[["date", "game_id", "team", "actual_runs"]]
all_results = pd.concat([home_df, away_df], ignore_index=True)

# Merge model predictions with actual results
eval_df = pd.merge(model_df, all_results, on=["date", "team"], how="inner")

# Add actual_win column (1 if team won, 0 if lost)
eval_df["actual_win"] = eval_df.groupby("date")["actual_runs"].transform(lambda x: (x == x.max()).astype(int))

# Create a binary prediction column based on win probability
eval_df["pred_win"] = (eval_df["win_prob"] > 0.5).astype(int)

# Compute evaluation metrics
eval_df["abs_error"] = abs(eval_df["expected_runs"] - eval_df["actual_runs"])
eval_df["squared_error"] = (eval_df["expected_runs"] - eval_df["actual_runs"]) ** 2
eval_df["sMAPE"] = 100 * abs(eval_df["expected_runs"] - eval_df["actual_runs"]) / ((abs(eval_df["expected_runs"]) + abs(eval_df["actual_runs"])) / 2)

# Compute overall accuracy
accuracy = (eval_df["pred_win"] == eval_df["actual_win"]).mean()
print(f"Win Prediction Accuracy: {accuracy:.2%}")

# Group by team for per-team accuracy
team_accuracy = eval_df.groupby("team").apply(lambda df: (df["pred_win"] == df["actual_win"]).mean()).reset_index(name="accuracy")
print("\n📈 Per-Team Win Prediction Accuracy:")
print(team_accuracy.sort_values("accuracy", ascending=False).head(10))

# Print evaluation summary
print("\n📊 Model Evaluation Metrics:")
print(f"MAE: {eval_df['abs_error'].mean():.3f}")
print(f"MSE: {eval_df['squared_error'].mean():.3f}")
print(f"sMAPE: {eval_df['sMAPE'].mean():.2f}%")

# Optional: show a few rows
print("\n🔍 Sample comparison:")
print(eval_df[["team", "expected_runs", "actual_runs", "abs_error", "sMAPE"]].head())

print(f"\n✅ Total rows evaluated: {len(eval_df)}")
