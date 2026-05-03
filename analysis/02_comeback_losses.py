"""Comeback-loss analysis.

Question: of the games where our model's pick lost, what fraction were
'comeback losses' - games where our pick was leading or tied through the
end of inning 5, and lost due to runs scored in innings 6-9+?

If the share is high, bullpen volatility is hurting the model and a first-5
or bullpen-aware framing might be worth exploring.

Also produces a per-inning run distribution (how scoring is distributed
across innings - context for the bullpen story).

Read-only.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis._common import output_dir, read_sql, style_plot, write_summary
from analysis._linescores import fetch_linescores


def load_picks() -> pd.DataFrame:
    """One row per game with the model's moneyline pick (higher win_prob team)."""
    query = """
        SELECT
            mos.game_pk,
            g.game_date,
            mos.team,
            CASE WHEN mos.team = g.home_team THEN 1 ELSE 0 END AS is_home,
            mos.win_prob,
            CASE WHEN mos.team = g.home_team THEN g.home_score
                 ELSE g.away_score END AS team_runs,
            CASE WHEN mos.team = g.home_team THEN g.away_score
                 ELSE g.home_score END AS opp_runs,
            g.home_team,
            g.away_team
        FROM model_outputs_season mos
        JOIN games g ON g.game_pk = mos.game_pk
        WHERE g.home_score IS NOT NULL
          AND g.away_score IS NOT NULL
          AND mos.win_prob IS NOT NULL
        ORDER BY g.game_date, mos.game_pk
    """
    df = read_sql(query)
    # Reduce to the model's pick per game = team with the higher stored win_prob.
    picks = df.sort_values("win_prob", ascending=False).groupby("game_pk").head(1).copy()
    picks["pick_won"] = picks["team_runs"] > picks["opp_runs"]
    picks["pick_lost"] = picks["team_runs"] < picks["opp_runs"]
    return picks.reset_index(drop=True)


def compute_inning_state(linescores: pd.DataFrame, picks: pd.DataFrame, through: int = 5) -> pd.DataFrame:
    """Per game: pick's runs and opp's runs through inning `through`."""
    # Aggregate runs through inning N for home/away
    inn = linescores[linescores["inning"] <= through]
    agg = inn.groupby("game_pk").agg(home_runs_thru=("home_runs", "sum"),
                                     away_runs_thru=("away_runs", "sum")).reset_index()
    df = picks.merge(agg, on="game_pk", how="left")
    df["pick_runs_thru"] = np.where(df["is_home"] == 1, df["home_runs_thru"], df["away_runs_thru"])
    df["opp_runs_thru"]  = np.where(df["is_home"] == 1, df["away_runs_thru"], df["home_runs_thru"])
    df["pick_leading_thru5"] = df["pick_runs_thru"] > df["opp_runs_thru"]
    df["pick_tied_thru5"]    = df["pick_runs_thru"] == df["opp_runs_thru"]
    df["pick_trailing_thru5"]= df["pick_runs_thru"] < df["opp_runs_thru"]
    return df


def plot_comeback_breakdown(df: pd.DataFrame, path: Path) -> None:
    style_plot()
    losses = df[df["pick_lost"]].copy()
    cats = {
        "Leading thru 5": int(losses["pick_leading_thru5"].sum()),
        "Tied thru 5":    int(losses["pick_tied_thru5"].sum()),
        "Trailing thru 5":int(losses["pick_trailing_thru5"].sum()),
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d62728", "#ff7f0e", "#7f7f7f"]
    bars = ax.bar(cats.keys(), cats.values(), color=colors)
    total = sum(cats.values()) or 1
    for bar, v in zip(bars, cats.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{v}\n({v/total*100:.1f}%)", ha="center", fontsize=10)
    ax.set_ylabel("Number of model losses")
    ax.set_title(f"Where the model's losses come from (n={total})")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_runs_by_inning(linescores: pd.DataFrame, path: Path) -> None:
    style_plot()
    by_inn = linescores.groupby("inning").agg(
        home_runs=("home_runs", "sum"),
        away_runs=("away_runs", "sum"),
        n_games=("game_pk", "nunique"),
    ).reset_index()
    by_inn["total_runs"] = by_inn["home_runs"] + by_inn["away_runs"]
    by_inn["runs_per_game"] = by_inn["total_runs"] / by_inn["n_games"]
    by_inn = by_inn[by_inn["inning"] <= 9]

    fig, ax = plt.subplots()
    ax.bar(by_inn["inning"], by_inn["runs_per_game"], color="#1f77b4", alpha=0.85)
    ax.set_xlabel("Inning"); ax.set_ylabel("Runs per game (both teams)")
    ax.set_title("Runs scored per inning (regulation only)")
    ax.set_xticks(range(1, 10))
    for _, row in by_inn.iterrows():
        ax.text(row["inning"], row["runs_per_game"] + 0.02,
                f"{row['runs_per_game']:.2f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return by_inn


def plot_runs_distribution_by_inning(linescores: pd.DataFrame, path: Path) -> None:
    """Box plot of runs per inning across games."""
    style_plot()
    df = linescores[linescores["inning"] <= 9].copy()
    df["both"] = df["home_runs"] + df["away_runs"]
    data = [df[df["inning"] == i]["both"].to_numpy() for i in range(1, 10)]
    fig, ax = plt.subplots()
    bp = ax.boxplot(data, tick_labels=range(1, 10), showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#1f77b4"); patch.set_alpha(0.6)
    ax.set_xlabel("Inning"); ax.set_ylabel("Runs in that inning (both teams)")
    ax.set_title("Distribution of runs per inning (no outliers shown)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main() -> None:
    out = output_dir("02_comeback_losses")

    print("Loading model picks...")
    picks = load_picks()
    print(f"  {len(picks)} completed games with predictions")

    print("Fetching line scores...")
    ls = fetch_linescores(picks["game_pk"].astype(int).tolist())
    games_with_ls = picks["game_pk"].isin(ls["game_pk"].unique())
    print(f"  {games_with_ls.sum()} / {len(picks)} games have line score data")

    df = compute_inning_state(ls, picks[games_with_ls].copy(), through=5)

    # Confidence bucket
    df["confidence"] = pd.cut(
        df["win_prob"], bins=[0.4, 0.55, 0.65, 0.75, 1.0],
        labels=["50-55%", "55-65%", "65-75%", "75%+"], include_lowest=True,
    )

    losses = df[df["pick_lost"]].copy()
    n_losses = len(losses)
    n_comeback = int(losses["pick_leading_thru5"].sum())
    n_tied = int(losses["pick_tied_thru5"].sum())
    n_trailing = int(losses["pick_trailing_thru5"].sum())

    plot_comeback_breakdown(df, out / "comeback_breakdown.png")
    by_inn = plot_runs_by_inning(ls, out / "runs_per_inning.png")
    plot_runs_distribution_by_inning(ls, out / "runs_distribution_by_inning.png")
    by_inn.to_csv(out / "runs_per_inning.csv", index=False)

    # Comeback rate by confidence bucket
    conf_breakdown = (
        losses.groupby("confidence", observed=True)
              .agg(n_losses=("pick_lost", "sum"),
                   n_comeback=("pick_leading_thru5", "sum"))
              .reset_index()
    )
    conf_breakdown["comeback_rate"] = conf_breakdown["n_comeback"] / conf_breakdown["n_losses"]
    conf_breakdown.to_csv(out / "comeback_rate_by_confidence.csv", index=False)

    # Save game-level detail for follow-up
    df[[
        "game_pk", "game_date", "team", "win_prob", "pick_won", "pick_lost",
        "pick_runs_thru", "opp_runs_thru", "pick_leading_thru5",
        "team_runs", "opp_runs",
    ]].to_csv(out / "games_detail.csv", index=False)

    overall_loss_rate = df["pick_lost"].mean()
    pct_comeback = n_comeback / n_losses * 100 if n_losses else 0
    pct_tied = n_tied / n_losses * 100 if n_losses else 0
    pct_trailing = n_trailing / n_losses * 100 if n_losses else 0

    summary = (
        f"- Sample: **{len(df)}** completed games with line scores\n"
        f"- Model pick lost: **{n_losses}** games ({overall_loss_rate*100:.1f}%)\n\n"
        "## Where do the losses come from?\n\n"
        f"| State through inning 5 | Losses | Share of losses |\n"
        f"|---|---|---|\n"
        f"| Pick was leading | {n_comeback} | **{pct_comeback:.1f}%** |\n"
        f"| Pick was tied    | {n_tied} | {pct_tied:.1f}% |\n"
        f"| Pick was trailing | {n_trailing} | {pct_trailing:.1f}% |\n\n"
        f"**{pct_comeback:.1f}%** of model losses were *comeback losses* - pick led after 5, lost the game.\n\n"
        "## Comeback-loss rate by model confidence\n\n"
        f"| Confidence | Losses | Comeback losses | Comeback rate |\n|---|---|---|---|\n"
        + "".join(
            f"| {r['confidence']} | {int(r['n_losses'])} | {int(r['n_comeback'])} | {r['comeback_rate']*100:.1f}% |\n"
            for _, r in conf_breakdown.iterrows()
        )
        + "\n"
        "## Files\n\n"
        "- `comeback_breakdown.png` - bar chart of where losses come from\n"
        "- `runs_per_inning.png` - total runs/game by inning (regulation)\n"
        "- `runs_distribution_by_inning.png` - boxplot of runs per inning\n"
        "- `comeback_rate_by_confidence.csv` - does it concentrate in high-confidence picks?\n"
        "- `games_detail.csv` - per-game detail for follow-up\n"
    )
    write_summary(out, "Comeback losses & inning run distribution", summary)
    print(f"\nLosses: {n_losses}  | Comeback losses: {n_comeback} ({pct_comeback:.1f}%)")


if __name__ == "__main__":
    main()
