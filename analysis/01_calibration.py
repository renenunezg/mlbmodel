"""Calibration analysis: raw negative-binomial win probs vs isotonic-calibrated.

The production model fits isotonic regression on each training run and applies it
to the negative-binomial win probabilities derived from the predicted xR pair.
What gets stored in `model_outputs_season.win_prob` is the *calibrated* value.

This script:
  1. Pulls completed games (we have actual_runs for both teams).
  2. Recomputes the *raw* NB win prob from the stored predicted xR pair.
  3. Treats the stored `win_prob` as the *calibrated* value.
  4. Compares the two via reliability curves, Brier score, and log loss.

Read-only. Does not touch the production model or database.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from analysis._common import output_dir, read_sql, style_plot, write_summary
from backend.simulation import win_prob


def load_games() -> pd.DataFrame:
    """One row per (game_pk, team) for completed games with predictions.

    Joins model_outputs_season → games (for final scores) → probable_starters
    (for is_home, since model_outputs_season doesn't carry it).
    """
    query = """
        SELECT
            mos.game_pk,
            g.game_date,
            mos.team,
            CASE WHEN mos.team = g.home_team THEN 1 ELSE 0 END AS is_home,
            mos.expected_runs AS xr,
            mos.win_prob AS win_prob_calibrated,
            CASE WHEN mos.team = g.home_team THEN g.home_score
                 ELSE g.away_score END AS actual_runs
        FROM model_outputs_season mos
        JOIN games g ON g.game_pk = mos.game_pk
        WHERE g.home_score IS NOT NULL
          AND g.away_score IS NOT NULL
          AND mos.expected_runs IS NOT NULL
          AND mos.win_prob IS NOT NULL
        ORDER BY g.game_date, mos.game_pk
    """
    return read_sql(query)


def add_raw_win_prob(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute the raw NB win prob from each game's xR pair."""
    df = df.copy()
    df["win_prob_raw"] = np.nan
    for game_pk, group in df.groupby("game_pk"):
        if len(group) != 2:
            continue
        rows = group.sort_values("is_home", ascending=True)
        lambda_away = max(rows.iloc[0]["xr"], 0.5)
        lambda_home = max(rows.iloc[1]["xr"], 0.5)
        p_home = win_prob(lambda_home, lambda_away)
        df.loc[rows.index[0], "win_prob_raw"] = 1.0 - p_home
        df.loc[rows.index[1], "win_prob_raw"] = p_home
    df["win_prob_raw"] = df["win_prob_raw"].clip(0.05, 0.95)
    return df


def add_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each (game_pk, team) row with 1 if won, 0 if lost, NaN if tied."""
    df = df.copy()
    df["won"] = np.nan
    for game_pk, group in df.groupby("game_pk"):
        if len(group) != 2:
            continue
        rows = group.sort_values("is_home")
        away_runs = rows.iloc[0]["actual_runs"]
        home_runs = rows.iloc[1]["actual_runs"]
        if pd.isna(away_runs) or pd.isna(home_runs):
            continue
        if home_runs > away_runs:
            df.loc[rows.index[0], "won"] = 0.0
            df.loc[rows.index[1], "won"] = 1.0
        elif away_runs > home_runs:
            df.loc[rows.index[0], "won"] = 1.0
            df.loc[rows.index[1], "won"] = 0.0
    return df


def reliability_curve(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10):
    """Equal-width reliability curve. Returns (bin_mid, observed_rate, count)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        out.append({
            "bin_mid": (bins[b] + bins[b + 1]) / 2,
            "predicted_mean": probs[mask].mean(),
            "observed_rate": outcomes[mask].mean(),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(out)


def plot_reliability(curves: dict[str, pd.DataFrame], path: Path) -> None:
    style_plot()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    colors = {"raw": "#d62728", "calibrated": "#2ca02c"}
    for name, df in curves.items():
        ax.plot(
            df["predicted_mean"], df["observed_rate"],
            marker="o", linewidth=2, markersize=7,
            color=colors.get(name, None), label=name.capitalize(),
        )
        for _, row in df.iterrows():
            ax.annotate(
                f"n={row['count']}",
                (row["predicted_mean"], row["observed_rate"]),
                fontsize=7, alpha=0.55, xytext=(4, 4), textcoords="offset points",
            )
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Observed win rate")
    ax.set_title("Reliability curve — raw vs isotonic-calibrated")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_distribution(df: pd.DataFrame, path: Path) -> None:
    style_plot()
    fig, ax = plt.subplots()
    ax.hist(df["win_prob_raw"], bins=30, alpha=0.55, label="Raw NB", color="#d62728")
    ax.hist(df["win_prob_calibrated"], bins=30, alpha=0.55, label="Calibrated", color="#2ca02c")
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Count of (game, team) rows")
    ax.set_title("Distribution of predicted win probabilities")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    out = output_dir("01_calibration")

    print("Loading completed games with predictions...")
    df = load_games()
    print(f"  {len(df)} (game, team) rows across {df['game_pk'].nunique()} games")

    df = add_raw_win_prob(df)
    df = add_outcomes(df)
    df = df.dropna(subset=["win_prob_raw", "win_prob_calibrated", "won"]).copy()
    print(f"  {len(df)} rows after dropping ties / missing predictions")

    raw_curve = reliability_curve(df["win_prob_raw"].to_numpy(), df["won"].to_numpy())
    cal_curve = reliability_curve(df["win_prob_calibrated"].to_numpy(), df["won"].to_numpy())
    raw_curve.to_csv(out / "reliability_raw.csv", index=False)
    cal_curve.to_csv(out / "reliability_calibrated.csv", index=False)

    plot_reliability({"raw": raw_curve, "calibrated": cal_curve}, out / "reliability.png")
    plot_distribution(df, out / "win_prob_distribution.png")

    metrics = {
        "n_rows": len(df),
        "n_games": df["game_pk"].nunique(),
        "raw_brier": brier_score_loss(df["won"], df["win_prob_raw"]),
        "calibrated_brier": brier_score_loss(df["won"], df["win_prob_calibrated"]),
        "raw_log_loss": log_loss(df["won"], df["win_prob_raw"].clip(1e-3, 1 - 1e-3)),
        "calibrated_log_loss": log_loss(df["won"], df["win_prob_calibrated"].clip(1e-3, 1 - 1e-3)),
        "raw_mean": df["win_prob_raw"].mean(),
        "calibrated_mean": df["win_prob_calibrated"].mean(),
        "actual_win_rate": df["won"].mean(),
    }
    pd.DataFrame([metrics]).to_csv(out / "metrics.csv", index=False)

    summary = (
        f"- Sample: **{metrics['n_games']:,}** completed games "
        f"(**{metrics['n_rows']:,}** team-game rows, ties excluded)\n"
        f"- Actual win rate: **{metrics['actual_win_rate']:.3f}** (sanity: should be ~0.5)\n\n"
        "## Brier score (lower is better)\n\n"
        f"| Variant | Brier | Log loss | Mean prob |\n"
        f"|---|---|---|---|\n"
        f"| Raw NB | {metrics['raw_brier']:.4f} | {metrics['raw_log_loss']:.4f} | {metrics['raw_mean']:.3f} |\n"
        f"| Isotonic-calibrated | {metrics['calibrated_brier']:.4f} | {metrics['calibrated_log_loss']:.4f} | {metrics['calibrated_mean']:.3f} |\n\n"
        f"**Brier delta:** {metrics['raw_brier'] - metrics['calibrated_brier']:+.4f} "
        f"({(metrics['raw_brier'] - metrics['calibrated_brier']) / metrics['raw_brier'] * 100:+.2f}% improvement)\n\n"
        "## Files\n\n"
        "- `reliability.png` — reliability curves overlaid for raw vs calibrated\n"
        "- `win_prob_distribution.png` — histogram of predicted probs\n"
        "- `reliability_{raw,calibrated}.csv` — per-bin observed vs predicted\n"
        "- `metrics.csv` — Brier/log-loss summary\n"
    )
    write_summary(out, "Calibration: raw NB vs isotonic", summary)
    print(f"\nWrote outputs to {out}")
    print(f"  Brier raw: {metrics['raw_brier']:.4f}  calibrated: {metrics['calibrated_brier']:.4f}")


if __name__ == "__main__":
    main()
