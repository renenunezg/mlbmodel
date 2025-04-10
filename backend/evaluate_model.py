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
model_df = pd.read_sql_table("model_outputs_season", con=engine)
results_df = pd.read_sql_table("game_results", con=engine)

results_df = results_df.dropna(subset=["home_score", "away_score"])
results_df["winning_team"] = np.where(results_df["home_score"] > results_df["away_score"],
                                       results_df["home_team"],
                                       results_df["away_team"])

# Reshape game_results into one row per team
home_df = results_df.rename(columns={"home_team": "team", "home_score": "actual_runs"})[["date", "game_id", "team", "actual_runs", "winning_team"]]
away_df = results_df.rename(columns={"away_team": "team", "away_score": "actual_runs"})[["date", "game_id", "team", "actual_runs", "winning_team"]]
home_df["actual_margin"] = results_df["home_score"] - results_df["away_score"]
away_df["actual_margin"] = results_df["away_score"] - results_df["home_score"]
all_results = pd.concat([home_df, away_df], ignore_index=True)

# Merge model predictions with actual results
eval_df = pd.merge(model_df, all_results[["game_id", "date", "team", "actual_runs", "winning_team", "actual_margin"]], on=["date", "team"], how="inner")

# Skip any rows where the actual result isn't available
eval_df = eval_df.dropna(subset=["actual_runs"])

# If no games have been played yet, exit with notice
if eval_df.empty:
    print("🚫 No games with completed scores to evaluate yet.")
    exit()

# Add actual_win column (1 if team won, 0 if lost)
eval_df["actual_win"] = (eval_df["team"] == eval_df["winning_team"]).astype(int)
print("DEBUG: actual_win assignment:")
print(eval_df[["date", "team", "actual_runs", "winning_team", "actual_win"]].drop_duplicates())

# Create a binary prediction column based on win probability
eval_df["pred_win"] = (eval_df["win_prob"] > 0.5).astype(int)

# Compute evaluation metrics
eval_df["abs_error"] = abs(eval_df["expected_runs"] - eval_df["actual_runs"])
eval_df["squared_error"] = (eval_df["expected_runs"] - eval_df["actual_runs"]) ** 2
eval_df["sMAPE"] = 100 * abs(eval_df["expected_runs"] - eval_df["actual_runs"]) / ((abs(eval_df["expected_runs"]) + abs(eval_df["actual_runs"])) / 2)

# Compute favorite win rate (based on win_prob > 0.5)
favorite_accuracy = (eval_df["pred_win"] == eval_df["actual_win"]).mean()
print(f"Favorite Accuracy (win_prob > 0.5): {favorite_accuracy:.2%}")

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

# Moneyline pick accuracy
print("DEBUG: Complete eval_df 'ev_flag' value counts:", eval_df["ev_flag"].value_counts())
ml_plays_all = eval_df[eval_df["ev_flag"] != "No Play"]
print("DEBUG: After filtering ev_flag != 'No Play', count =", len(ml_plays_all))
print("DEBUG: Unique 'ev_flag' values in ml_plays_all:", ml_plays_all["ev_flag"].unique())
ml_plays = ml_plays_all[ml_plays_all["ev_flag"] == ml_plays_all["team"]]
print("DEBUG: After filtering ev_flag == team, count =", len(ml_plays))
print("DEBUG: ml_plays ev_flag value counts:", ml_plays["ev_flag"].value_counts())
if "game_id" in ml_plays.columns:
    ml_plays = ml_plays.drop_duplicates(subset=["game_id"])
else:
    ml_plays = ml_plays.drop_duplicates(subset=["date", "team", "ev_flag"])

ml_correct = ml_plays[ml_plays["actual_win"] == 1].shape[0]
ml_incorrect = ml_plays[ml_plays["actual_win"] == 0].shape[0]
ml_total = ml_correct + ml_incorrect
ml_accuracy = ml_correct / ml_total if ml_total > 0 else np.nan
print(f"\nMoneyline Pick Accuracy: {ml_correct}/{ml_total} ({ml_accuracy:.2%})")

print("\n🔎 Moneyline Picks Evaluated:")
print(ml_plays[["date", "team", "ev_flag", "actual_win", "win_prob"]])

# Evaluate Run Line Picks:
# Define a function to determine if a run line pick is correct.
# Logic:
#   - If run_line is negative (e.g., -1.5): pick wins if actual_margin is at least 2 runs.
#   - If run_line is positive (e.g., +1.5): pick wins if the team won or lost by exactly 1 run.
def calc_run_line_pick(row):
    try:
        run_line_value = float(row["run_line"])
    except:
        return np.nan
    margin = row.get("actual_margin", np.nan)
    if pd.isna(margin):
        return np.nan
    # For negative run_line (e.g., -1.5): pick wins if actual_margin is at least 2 runs.
    if run_line_value < 0:
        return 1 if margin >= 2 else 0
    # For positive run_line (e.g., +1.5): pick wins if the team won or lost by at least 1 run.
    elif run_line_value > 0:
        return 1 if (row["actual_win"] == 1 or margin >= -1) else 0
    else:
        return np.nan

if "game_id" in eval_df.columns:
    runline_plays = eval_df[eval_df["run_line_ev_flag"] != "No Play"].groupby("game_id").first().reset_index()
else:
    runline_plays = eval_df[eval_df["run_line_ev_flag"] != "No Play"].drop_duplicates(subset=["date", "team", "run_line_ev_flag"]).copy()
runline_plays = runline_plays[runline_plays["run_line_ev_flag"] == runline_plays["team"]]
runline_plays["run_line_correct"] = runline_plays.apply(calc_run_line_pick, axis=1)
rl_correct = runline_plays[runline_plays["run_line_correct"] == 1].shape[0]
rl_incorrect = runline_plays[runline_plays["run_line_correct"] == 0].shape[0]
rl_total = rl_correct + rl_incorrect
run_line_accuracy = rl_correct / rl_total if rl_total > 0 else np.nan
print(f"Run Line Pick Accuracy: {rl_correct}/{rl_total} ({run_line_accuracy:.2%})")

# Evaluate Totals Picks:
# Compute game total using results_df (one row per game)
results_df["game_total"] = results_df["home_score"] + results_df["away_score"]

if "game_id" in eval_df.columns:
    # Group by game_id, total, and total_play to ensure one totals pick per game.
    totals_eval = eval_df[eval_df["total_play"].isin(["Over", "Under"])].groupby(["game_id", "total", "total_play"]).first().reset_index()
    # Merge game_total from results_df using game_id
    totals_eval = totals_eval.merge(results_df[["game_id", "game_total"]].drop_duplicates(subset=["game_id"]), on="game_id", how="left")
else:
    # Fallback: if game_id is not available, deduplicate using a combination of date, team, total, and total_play,
    # and merge on date.
    totals_eval = eval_df[eval_df["total_play"].isin(["Over", "Under"])].drop_duplicates(subset=["date", "team", "total", "total_play"]).copy()
    totals_eval = totals_eval.merge(results_df[["date", "game_total"]].drop_duplicates(subset=["date"]), on="date", how="left")

def calc_total_pick_row(row):
    try:
        model_total = float(row["total"])
    except:
        return np.nan
    game_total = row["game_total"]
    if pd.isna(game_total):
        return np.nan
    if row["total_play"].strip().lower() == "over":
        if game_total > model_total:
            return 1
        elif game_total == model_total:
            return np.nan  # push
        else:
            return 0
    elif row["total_play"].strip().lower() == "under":
        if game_total < model_total:
            return 1
        elif game_total == model_total:
            return np.nan  # push
        else:
            return 0
    else:
        return np.nan

totals_eval["total_pick_correct"] = totals_eval.apply(calc_total_pick_row, axis=1)

# Separate overs and unders
overs_eval = totals_eval[totals_eval["total_play"].str.strip().str.lower() == "over"].copy()
unders_eval = totals_eval[totals_eval["total_play"].str.strip().str.lower() == "under"].copy()
overs_eval = overs_eval.dropna(subset=["total_pick_correct"])
unders_eval = unders_eval.dropna(subset=["total_pick_correct"])

overs_total = len(overs_eval)
unders_total = len(unders_eval)
overs_correct = overs_eval["total_pick_correct"].sum()
unders_correct = unders_eval["total_pick_correct"].sum()
overs_accuracy = overs_correct / overs_total if overs_total > 0 else np.nan
unders_accuracy = unders_correct / unders_total if unders_total > 0 else np.nan

print(f"Totals Over Pick Accuracy: {overs_correct}/{overs_total} ({overs_accuracy:.2%})")
print(f"Totals Under Pick Accuracy: {unders_correct}/{unders_total} ({unders_accuracy:.2%})")
from sqlalchemy import Table, MetaData
from sqlalchemy.dialects.postgresql import insert

metadata = MetaData()
metadata.reflect(bind=engine)
model_evaluation = metadata.tables["model_evaluation"]

today = pd.to_datetime(eval_df["date"].max()).date()

# Daily totals
daily_total_predictions = eval_df.shape[0]
daily_total_correct = (eval_df["pred_win"] == eval_df["actual_win"]).sum()
daily_total_accuracy = daily_total_correct / daily_total_predictions

# Daily averages
average_total_diff = abs(eval_df["expected_runs"] - eval_df["actual_runs"]).mean()
average_win_prob = eval_df["win_prob"].mean()

# Insert or update daily evaluation
with engine.begin() as conn:
    conn.execute(
        insert(model_evaluation).values(
            date=today,
            total_correct=int(daily_total_correct),
            total_predictions=int(daily_total_predictions),
            total_accuracy=round(float(daily_total_accuracy), 2),
            ml_correct=int(ml_correct),
            ml_predictions=int(ml_total),
            ml_accuracy=round(float(ml_accuracy), 2),
            run_line_correct=int(rl_correct),
            run_line_predictions=int(rl_total),
            run_line_accuracy=round(float(run_line_accuracy), 2),
            average_total_diff=round(float(average_total_diff), 2),
            average_win_prob=round(float(average_win_prob), 2),
        ).on_conflict_do_update(
            index_elements=["date"],
            set_={
                "total_correct": int(daily_total_correct),
                "total_predictions": int(daily_total_predictions),
                "total_accuracy": round(float(daily_total_accuracy), 2),
                "ml_correct": int(ml_correct),
                "ml_predictions": int(ml_total),
                "ml_accuracy": round(float(ml_accuracy), 2),
                "run_line_correct": int(rl_correct),
                "run_line_predictions": int(rl_total),
                "run_line_accuracy": round(float(run_line_accuracy), 2),
                "average_total_diff": round(float(average_total_diff), 2),
                "average_win_prob": round(float(average_win_prob), 2),
            }
        )
    )
    # Fetch and display the inserted row to confirm
    result = conn.execute(model_evaluation.select().where(model_evaluation.c.date == today)).fetchone()
    print("\n🔎 Retrieved from DB:")
    print(dict(result._mapping))
