"""Bullpen fatigue analysis.

Question: do model picks fare worse when the OPPOSING team's bullpen is
fatigued (heavy reliever IP in the prior 1-3 days)? If yes, that's a feature
the model should add.

Procedure:
  1. For each team, build a daily timeline of reliever outs from boxscores.
  2. For each (game_pk, team) row, compute prior-1d / prior-2d / prior-3d
     reliever outs for the OPPOSING team's bullpen (since that's what our
     team will face today).
  3. Bucket games by opp bullpen fatigue (low/medium/high IP load) and
     measure model pick win rate + ROI per bucket.
  4. Also do the inverse: when OUR pick has a heavily-used bullpen behind
     them, do they cough up more leads (comeback losses)?

Read-only.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis._boxscores import fetch_boxscores
from analysis._common import output_dir, read_sql, style_plot, write_summary


def load_picks() -> pd.DataFrame:
    query = """
        SELECT
            mos.game_pk,
            g.game_date,
            mos.team,
            CASE WHEN mos.team = g.home_team THEN 1 ELSE 0 END AS is_home,
            mos.win_prob,
            mos.moneyline,
            CASE WHEN mos.team = g.home_team THEN g.home_score
                 ELSE g.away_score END AS team_runs,
            CASE WHEN mos.team = g.home_team THEN g.away_score
                 ELSE g.home_score END AS opp_runs,
            g.home_team,
            g.away_team
        FROM model_outputs_season mos
        JOIN games g ON g.game_pk = mos.game_pk
        WHERE g.home_score IS NOT NULL AND g.away_score IS NOT NULL
          AND mos.win_prob IS NOT NULL
        ORDER BY g.game_date, mos.game_pk
    """
    df = read_sql(query)
    picks = df.sort_values("win_prob", ascending=False).groupby("game_pk").head(1).copy()
    picks["pick_won"] = picks["team_runs"] > picks["opp_runs"]
    picks["pick_lost"] = picks["team_runs"] < picks["opp_runs"]
    picks["opponent"] = np.where(picks["is_home"] == 1, picks["away_team"], picks["home_team"])
    return picks.reset_index(drop=True)


def build_team_bullpen_timeline(boxscores: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, team) with reliever_outs that day."""
    bx = boxscores.merge(games[["game_pk", "game_date", "home_team", "away_team"]], on="game_pk")
    bx["team"] = np.where(bx["team_side"] == "home", bx["home_team"], bx["away_team"])
    daily = bx.groupby(["game_date", "team"]).agg(
        reliever_outs=("reliever_outs", "sum"),
        n_games=("game_pk", "nunique"),
    ).reset_index()
    return daily


def compute_prior_outs(picks: pd.DataFrame, daily: pd.DataFrame, days: int) -> pd.Series:
    """For each pick row, return opponent's reliever outs in the prior `days` days."""
    daily = daily.copy()
    daily["game_date"] = pd.to_datetime(daily["game_date"])

    out = []
    for _, r in picks.iterrows():
        game_date = pd.to_datetime(r["game_date"])
        opp = r["opponent"]
        window = daily[(daily["team"] == opp)
                       & (daily["game_date"] >= game_date - pd.Timedelta(days=days))
                       & (daily["game_date"] < game_date)]
        out.append(int(window["reliever_outs"].sum()))
    return pd.Series(out, index=picks.index, dtype=int)


def bucket_by_quantile(s: pd.Series, n_buckets: int = 3) -> pd.Series:
    """Return labels 'low' / 'mid' / 'high' by tertile."""
    try:
        return pd.qcut(s, n_buckets, labels=["low", "mid", "high"], duplicates="drop")
    except ValueError:
        return pd.cut(s, n_buckets, labels=["low", "mid", "high"])


def plot_winrate_by_bucket(df: pd.DataFrame, col: str, title: str, path: Path) -> None:
    style_plot()
    g = df.groupby(col, observed=True).agg(
        n=("pick_won", "size"),
        win_rate=("pick_won", "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(g[col].astype(str), g["win_rate"] * 100, color="#1f77b4", alpha=0.85)
    for bar, wr, n in zip(bars, g["win_rate"], g["n"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{wr*100:.1f}%\nn={n}", ha="center", fontsize=9)
    ax.axhline(50, color="gray", linestyle="--", alpha=0.5, label="50% (coin flip)")
    ax.set_ylabel("Pick win rate (%)")
    ax.set_xlabel(col.replace("_", " ").title())
    ax.set_title(title)
    ax.set_ylim(0, max(70, g["win_rate"].max() * 100 + 8))
    ax.legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main() -> None:
    out = output_dir("03_bullpen_fatigue")

    print("Loading picks + games...")
    picks = load_picks()
    games = read_sql("SELECT game_pk, game_date, home_team, away_team FROM games WHERE home_score IS NOT NULL")
    games["game_date"] = pd.to_datetime(games["game_date"])

    print(f"  {len(picks)} picks across {picks['game_date'].nunique()} days")

    print("Fetching boxscores (reliever IP)...")
    bx = fetch_boxscores(picks["game_pk"].astype(int).tolist())
    print(f"  {bx['game_pk'].nunique()} boxscores loaded")

    daily = build_team_bullpen_timeline(bx, games)
    print(f"  {len(daily)} (date, team) bullpen-day rows")

    # Compute opponent bullpen fatigue at game time
    picks["opp_bp_outs_1d"] = compute_prior_outs(picks, daily, days=1)
    picks["opp_bp_outs_2d"] = compute_prior_outs(picks, daily, days=2)
    picks["opp_bp_outs_3d"] = compute_prior_outs(picks, daily, days=3)

    # And our pick's own bullpen fatigue (relevant for comeback losses)
    picks["own_team"] = picks["team"]
    own_outs = []
    for d in (1, 2, 3):
        col = f"own_bp_outs_{d}d"
        # Reuse compute_prior_outs by swapping opponent → own team
        tmp = picks.rename(columns={"opponent": "_orig_opp", "team": "opponent"}).copy()
        picks[col] = compute_prior_outs(tmp, daily, days=d)

    # Bucket and plot — opp fatigue → our pick win rate
    picks["opp_fatigue_2d"] = bucket_by_quantile(picks["opp_bp_outs_2d"])
    picks["own_fatigue_2d"] = bucket_by_quantile(picks["own_bp_outs_2d"])

    plot_winrate_by_bucket(
        picks, "opp_fatigue_2d",
        "Pick win rate by OPPONENT bullpen fatigue (prior 2 days)",
        out / "winrate_by_opp_fatigue.png",
    )
    plot_winrate_by_bucket(
        picks, "own_fatigue_2d",
        "Pick win rate by OWN bullpen fatigue (prior 2 days)",
        out / "winrate_by_own_fatigue.png",
    )

    # Detailed table per bucket and window
    rows = []
    for window in (1, 2, 3):
        bcol = f"opp_bp_outs_{window}d"
        bucket = bucket_by_quantile(picks[bcol])
        picks[f"opp_fatigue_{window}d"] = bucket
        g = picks.groupby(bucket, observed=True).agg(
            n=("pick_won", "size"),
            win_rate=("pick_won", "mean"),
            avg_outs=(bcol, "mean"),
        ).reset_index().rename(columns={bcol: "bucket"})
        g["window_days"] = window
        g["axis"] = "opponent"
        rows.append(g)
    summary_df = pd.concat(rows, ignore_index=True)
    summary_df.to_csv(out / "fatigue_buckets.csv", index=False)

    # ROI by opp fatigue bucket (moneyline)
    picks["dec_odds"] = np.where(
        picks["moneyline"].fillna(0) > 0,
        picks["moneyline"] / 100 + 1,
        100 / picks["moneyline"].abs() + 1,
    )
    picks["pnl"] = np.where(picks["pick_won"], picks["dec_odds"] - 1, -1)
    roi_table = picks.groupby("opp_fatigue_2d", observed=True).agg(
        n=("pick_won", "size"),
        win_rate=("pick_won", "mean"),
        roi=("pnl", "mean"),
    ).reset_index()
    roi_table.to_csv(out / "roi_by_opp_fatigue.csv", index=False)

    # Build summary text
    op = roi_table.set_index("opp_fatigue_2d")
    summary = (
        f"- Sample: **{len(picks)}** completed games with boxscore data\n\n"
        "## Pick win rate by OPPONENT bullpen fatigue (prior 2 days)\n\n"
        "| Opp BP fatigue | n | Pick win rate | ROI per 1u flat-bet |\n"
        "|---|---|---|---|\n"
    )
    for label in ["low", "mid", "high"]:
        if label in op.index:
            r = op.loc[label]
            summary += f"| {label} | {int(r['n'])} | {r['win_rate']*100:.1f}% | {r['roi']*100:+.2f}% |\n"
    summary += (
        "\n*If 'high' is materially better than 'low', opponent bullpen fatigue is a "
        "real edge the model isn't capturing.*\n\n"
        "## Files\n"
        "- `winrate_by_opp_fatigue.png`\n"
        "- `winrate_by_own_fatigue.png`\n"
        "- `fatigue_buckets.csv` — full breakdown across 1d/2d/3d windows\n"
        "- `roi_by_opp_fatigue.csv` — ROI per fatigue bucket\n"
    )
    write_summary(out, "Bullpen fatigue impact", summary)

    print(f"\nDone. Outputs at {out}")
    print(roi_table.to_string(index=False))


if __name__ == "__main__":
    main()
