"""
Evaluate model predictions against actual game results.

Computes accuracy metrics for moneyline, run line, and totals picks,
then upserts a summary row into the model_evaluation table.

Usage:
    from backend.evaluate_model import main
    main()
"""

import pandas as pd
import numpy as np
from sqlalchemy import MetaData
from sqlalchemy.dialects.postgresql import insert

from backend.db import engine


def _calc_run_line_pick(row):
    try:
        spread_value = float(row["spread"])
    except Exception:
        return np.nan
    margin = row.get("actual_margin", np.nan)
    if pd.isna(margin):
        return np.nan
    if spread_value < 0:
        return 1 if margin >= abs(spread_value) else 0
    elif spread_value > 0:
        return 1 if (row["actual_win"] == 1 or margin >= -spread_value) else 0
    return np.nan


def _calc_total_pick(row):
    try:
        line = float(row["total"])
    except Exception:
        return np.nan
    actual = row["game_total"]
    if pd.isna(actual):
        return np.nan
    direction = row["total_play"].strip().lower()
    if direction == "over":
        return 1 if actual > line else (np.nan if actual == line else 0)
    elif direction == "under":
        return 1 if actual < line else (np.nan if actual == line else 0)
    return np.nan


def main():
    """Run full evaluation and write results to model_evaluation table."""
    model_df = pd.read_sql_table("model_outputs_season", con=engine)
    games_df = pd.read_sql_table("games", con=engine)

    games_df = games_df[games_df["status"] == "Final"].dropna(subset=["home_score", "away_score"])
    if games_df.empty:
        print("  No completed games to evaluate against.")
        return

    games_df["winning_team"] = np.where(
        games_df["home_score"] > games_df["away_score"],
        games_df["home_team"],
        games_df["away_team"],
    )

    # Reshape games: one row per team
    home_df = games_df.rename(columns={"home_team": "team", "home_score": "actual_runs"})[
        ["game_date", "game_pk", "team", "actual_runs", "winning_team"]
    ]
    away_df = games_df.rename(columns={"away_team": "team", "away_score": "actual_runs"})[
        ["game_date", "game_pk", "team", "actual_runs", "winning_team"]
    ]
    home_df["actual_margin"] = games_df["home_score"].values - games_df["away_score"].values
    away_df["actual_margin"] = games_df["away_score"].values - games_df["home_score"].values
    all_results = pd.concat([home_df, away_df], ignore_index=True)

    # Merge predictions with results
    eval_df = pd.merge(
        model_df,
        all_results[["game_pk", "game_date", "team", "actual_runs", "winning_team", "actual_margin"]],
        on=["game_pk", "team"],
        how="inner",
    ).dropna(subset=["actual_runs"])

    if eval_df.empty:
        print("  No predictions matched to completed games yet.")
        return

    # Win prediction accuracy
    eval_df["actual_win"] = (eval_df["team"] == eval_df["winning_team"]).astype(int)
    eval_df["pred_win"] = (eval_df["win_prob"] > 0.5).astype(int)

    accuracy = (eval_df["pred_win"] == eval_df["actual_win"]).mean()
    mae = abs(eval_df["expected_runs"] - eval_df["actual_runs"]).mean()
    print(f"  Win accuracy: {accuracy:.2%} | Runs MAE: {mae:.3f} | {len(eval_df)} predictions evaluated")

    # Moneyline picks
    ml_plays = eval_df[eval_df["ev_flag"] == eval_df["team"]].drop_duplicates(subset=["game_pk"])
    ml_correct = (ml_plays["actual_win"] == 1).sum()
    ml_total = len(ml_plays)
    ml_accuracy = ml_correct / ml_total if ml_total > 0 else np.nan
    print(f"  ML picks: {ml_correct}/{ml_total} ({ml_accuracy:.2%})" if ml_total > 0 else "  ML picks: none")

    # Run line picks
    rl_plays = eval_df[eval_df["run_line_ev_flag"] == eval_df["team"]].drop_duplicates(subset=["game_pk"])
    rl_plays["run_line_correct"] = rl_plays.apply(_calc_run_line_pick, axis=1)
    rl_correct = int(rl_plays["run_line_correct"].sum()) if not rl_plays.empty else 0
    rl_total = int(rl_plays["run_line_correct"].notna().sum())
    rl_accuracy = rl_correct / rl_total if rl_total > 0 else np.nan
    print(f"  RL picks: {rl_correct}/{rl_total} ({rl_accuracy:.2%})" if rl_total > 0 else "  RL picks: none")

    # Totals picks
    games_df["game_total"] = games_df["home_score"] + games_df["away_score"]
    totals_eval = (
        eval_df[eval_df["total_play"].isin(["Over", "Under"])]
        .groupby(["game_pk", "total", "total_play"])
        .first()
        .reset_index()
    )
    totals_eval = totals_eval.merge(
        games_df[["game_pk", "game_total"]].drop_duplicates(subset=["game_pk"]),
        on="game_pk",
        how="left",
    )
    totals_eval["total_pick_correct"] = totals_eval.apply(_calc_total_pick, axis=1)
    totals_correct = int(totals_eval["total_pick_correct"].sum())
    totals_total = int(totals_eval["total_pick_correct"].notna().sum())
    totals_accuracy = totals_correct / totals_total if totals_total > 0 else np.nan
    print(f"  Totals picks: {totals_correct}/{totals_total} ({totals_accuracy:.2%})" if totals_total > 0 else "  Totals picks: none")

    # Upsert into model_evaluation
    metadata = MetaData()
    metadata.reflect(bind=engine)
    model_evaluation = metadata.tables["model_evaluation"]

    eval_date = pd.to_datetime(eval_df["game_date"].max()).date()
    total_correct = int((eval_df["pred_win"] == eval_df["actual_win"]).sum())
    total_predictions = len(eval_df)

    row = {
        "date": eval_date,
        "total_correct": total_correct,
        "total_predictions": total_predictions,
        "total_accuracy": round(total_correct / total_predictions, 2),
        "ml_correct": int(ml_correct),
        "ml_predictions": int(ml_total),
        "ml_accuracy": round(float(ml_accuracy), 2) if pd.notna(ml_accuracy) else 0.0,
        "run_line_correct": int(rl_correct),
        "run_line_predictions": int(rl_total),
        "run_line_accuracy": round(float(rl_accuracy), 2) if pd.notna(rl_accuracy) else 0.0,
        "average_total_diff": round(float(mae), 2),
        "average_win_prob": round(float(eval_df["win_prob"].mean()), 2),
    }

    update_cols = {k: v for k, v in row.items() if k != "date"}

    with engine.begin() as conn:
        conn.execute(
            insert(model_evaluation)
            .values(**row)
            .on_conflict_do_update(index_elements=["date"], set_=update_cols)
        )

    print(f"  Evaluation written for {eval_date}")


if __name__ == "__main__":
    main()
