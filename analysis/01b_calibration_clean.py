"""Clean calibration diagnostic.

Production stores `win_prob` as calibrated by whatever isotonic regressor was
fit on the day of prediction - historical values are a mishmash of many
calibrators, and each was fit on a barely-sufficient OOF sample.

This script asks the cleaner question: if we fit ONE isotonic calibrator
correctly via cross-validation on the full historical sample, does it improve
on raw NB win probabilities?

Procedure:
  1. Recompute raw NB win prob from stored expected_runs pairs (same as 01).
  2. 5-fold GroupKFold by game_pk (keeps both team rows of a game in same fold).
  3. In each fold: fit IsotonicRegression on the other 4 folds, predict on held-out.
  4. Compare raw vs CV-calibrated reliability + Brier + log loss.

Read-only. Does not touch production model or DB.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold

from analysis._common import output_dir, read_sql, style_plot, write_summary
from backend.simulation import win_prob


def load_games() -> pd.DataFrame:
    query = """
        SELECT
            mos.game_pk,
            g.game_date,
            mos.team,
            CASE WHEN mos.team = g.home_team THEN 1 ELSE 0 END AS is_home,
            mos.expected_runs AS xr,
            CASE WHEN mos.team = g.home_team THEN g.home_score
                 ELSE g.away_score END AS actual_runs
        FROM model_outputs_season mos
        JOIN games g ON g.game_pk = mos.game_pk
        WHERE g.home_score IS NOT NULL
          AND g.away_score IS NOT NULL
          AND mos.expected_runs IS NOT NULL
        ORDER BY g.game_date, mos.game_pk
    """
    return read_sql(query)


def add_raw_win_prob(df: pd.DataFrame) -> pd.DataFrame:
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
    df = df.copy()
    df["won"] = np.nan
    for game_pk, group in df.groupby("game_pk"):
        if len(group) != 2:
            continue
        rows = group.sort_values("is_home")
        a, h = rows.iloc[0]["actual_runs"], rows.iloc[1]["actual_runs"]
        if pd.isna(a) or pd.isna(h):
            continue
        if h > a:
            df.loc[rows.index[0], "won"] = 0.0; df.loc[rows.index[1], "won"] = 1.0
        elif a > h:
            df.loc[rows.index[0], "won"] = 1.0; df.loc[rows.index[1], "won"] = 0.0
    return df


def cv_isotonic(df: pd.DataFrame, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold predictions from CV-fit isotonic regressors.

    GroupKFold by game_pk keeps both rows of a game in the same fold so the
    two complementary probs aren't split across train/test.

    Renormalization is intentionally NOT applied here - we want to isolate
    'does isotonic itself help?' from 'does the renormalization step undo it?'.
    """
    probs = df["win_prob_raw"].to_numpy()
    won = df["won"].to_numpy()
    groups = df["game_pk"].to_numpy()
    out = np.full_like(probs, np.nan)
    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in gkf.split(probs, won, groups):
        cal = IsotonicRegression(y_min=0.05, y_max=0.95, out_of_bounds="clip")
        cal.fit(probs[train_idx], won[train_idx])
        out[test_idx] = cal.predict(probs[test_idx])
    return out


def cv_isotonic_renormalized(df: pd.DataFrame, n_splits: int = 5) -> np.ndarray:
    """Same as cv_isotonic but with the production renormalization step applied."""
    probs = cv_isotonic(df, n_splits=n_splits).copy()
    df = df.assign(_p=probs)
    out = probs.copy()
    for game_pk, group in df.groupby("game_pk"):
        if len(group) != 2:
            continue
        idx = group.index.to_numpy()
        positions = [df.index.get_loc(i) for i in idx]
        total = group["_p"].sum()
        if total > 0:
            for pos, val in zip(positions, group["_p"] / total):
                out[pos] = val
    return np.clip(out, 0.05, 0.95)


def reliability_curve(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_mid": (bins[b] + bins[b + 1]) / 2,
            "predicted_mean": probs[mask].mean(),
            "observed_rate": outcomes[mask].mean(),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def plot_reliability(curves: dict, path: Path) -> None:
    style_plot()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    palette = {"raw": "#d62728", "cv_isotonic": "#1f77b4", "cv_isotonic_renormalized": "#2ca02c"}
    for name, df in curves.items():
        ax.plot(
            df["predicted_mean"], df["observed_rate"],
            marker="o", linewidth=2, markersize=7,
            color=palette.get(name), label=name.replace("_", " ").title(),
        )
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted win probability"); ax.set_ylabel("Observed win rate")
    ax.set_title("Reliability - raw vs CV-fit isotonic (with/without renormalization)")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main() -> None:
    out = output_dir("01b_calibration_clean")
    print("Loading...")
    df = load_games()
    df = add_raw_win_prob(df)
    df = add_outcomes(df)
    df = df.dropna(subset=["win_prob_raw", "won"]).reset_index(drop=True)
    print(f"  {len(df)} rows / {df['game_pk'].nunique()} games")

    df["win_prob_iso_cv"] = cv_isotonic(df)
    df["win_prob_iso_renorm"] = cv_isotonic_renormalized(df)

    variants = {
        "raw": df["win_prob_raw"],
        "cv_isotonic": df["win_prob_iso_cv"],
        "cv_isotonic_renormalized": df["win_prob_iso_renorm"],
    }
    metrics = []
    curves = {}
    for name, probs in variants.items():
        p = probs.to_numpy() if hasattr(probs, "to_numpy") else probs
        metrics.append({
            "variant": name,
            "brier": brier_score_loss(df["won"], p),
            "log_loss": log_loss(df["won"], np.clip(p, 1e-3, 1 - 1e-3)),
            "mean_prob": float(np.mean(p)),
        })
        curve = reliability_curve(p, df["won"].to_numpy())
        curve.to_csv(out / f"reliability_{name}.csv", index=False)
        curves[name] = curve

    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    plot_reliability(curves, out / "reliability.png")

    m = {x["variant"]: x for x in metrics}
    summary = (
        f"- Sample: **{df['game_pk'].nunique()}** games, **{len(df)}** team-game rows (CV-fit isotonic, GroupKFold by game_pk)\n\n"
        "## Brier score (lower better)\n\n"
        f"| Variant | Brier | Log loss | Mean prob |\n"
        f"|---|---|---|---|\n"
        f"| Raw NB | {m['raw']['brier']:.4f} | {m['raw']['log_loss']:.4f} | {m['raw']['mean_prob']:.3f} |\n"
        f"| CV isotonic (no renorm) | {m['cv_isotonic']['brier']:.4f} | {m['cv_isotonic']['log_loss']:.4f} | {m['cv_isotonic']['mean_prob']:.3f} |\n"
        f"| CV isotonic + renorm (prod-style) | {m['cv_isotonic_renormalized']['brier']:.4f} | {m['cv_isotonic_renormalized']['log_loss']:.4f} | {m['cv_isotonic_renormalized']['mean_prob']:.3f} |\n\n"
        "## Interpretation\n\n"
        "- Compare **raw vs cv_isotonic**: does cleanly-fit calibration help at all on this sample?\n"
        "- Compare **cv_isotonic vs cv_isotonic_renormalized**: is the renormalization step (suspect #2) the thing that's hurting?\n"
    )
    write_summary(out, "Clean calibration diagnostic", summary)
    for v in metrics:
        print(f"  {v['variant']:30s}  Brier={v['brier']:.4f}  LogLoss={v['log_loss']:.4f}")


if __name__ == "__main__":
    main()
