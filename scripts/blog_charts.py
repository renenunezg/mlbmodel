"""Generate publication-quality charts for blog posts from live Supabase data."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from sqlalchemy import text

from backend.db import engine

OUT_DIR = Path("analysis/blog_charts")

ACCENT = "#2d7a4f"
GRID_COLOR = "#e5e5e5"
AXIS_COLOR = "#d4d4d8"
TICK_COLOR = "#52525b"


def _apply_style(fig, ax):
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.tick_params(colors=TICK_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TICK_COLOR)
    ax.yaxis.label.set_color(TICK_COLOR)
    ax.title.set_color(TICK_COLOR)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_edgecolor(AXIS_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, linestyle="--")


def load_evaluations() -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT * FROM model_evaluation ORDER BY date ASC"),
            conn,
        )
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_calibration() -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(
            text(
                "SELECT * FROM model_calibration "
                "ORDER BY date DESC LIMIT 10"
            ),
            conn,
        )
    return df


def accuracy_over_time(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_style(fig, ax)

    ax.axhline(0.5, color="#475569", linewidth=0.8, linestyle="--", label="50% baseline")
    ax.plot(df["date"], df["ml_accuracy"], color="#22c55e", linewidth=1.8, label="Moneyline")
    ax.plot(df["date"], df["run_line_accuracy"], color="#3b82f6", linewidth=1.8, label="Run line")
    ax.plot(df["date"], df["totals_accuracy"], color="#a855f7", linewidth=1.8, label="Totals")
    ax.plot(df["date"], df["total_accuracy"], color="#d4a96a", linewidth=1.4, linestyle="--", label="Pick accuracy")

    ax.set_title("Prediction Accuracy Over Time", fontsize=13, pad=12)
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    fig.autofmt_xdate(rotation=30)

    legend = ax.legend(fontsize=9, framealpha=0.8, edgecolor=GRID_COLOR, labelcolor=TICK_COLOR)
    legend.get_frame().set_facecolor("white")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "accuracy_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def equity_curve(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    _apply_style(fig, ax)

    ax.axhline(0, color="#475569", linewidth=0.8, linestyle="--")
    ax.fill_between(
        df["date"],
        df["equity_end_units"],
        0,
        where=df["equity_end_units"] >= 0,
        color="#22c55e",
        alpha=0.25,
    )
    ax.fill_between(
        df["date"],
        df["equity_end_units"],
        0,
        where=df["equity_end_units"] < 0,
        color="#ef4444",
        alpha=0.25,
    )
    ax.plot(df["date"], df["equity_end_units"], color="#22c55e", linewidth=2)

    ax.set_title("Equity Curve (units, flat stakes)", fontsize=13, pad=12)
    ax.set_ylabel("P&L (u)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    fig.autofmt_xdate(rotation=30)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "equity_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def calibration_chart(cal: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    _apply_style(fig, ax)

    ax.plot([0, 1], [0, 1], color="#475569", linewidth=1, linestyle="--", label="Perfect calibration")
    ax.scatter(
        cal["predicted_mean"],
        cal["observed_rate"],
        color="#3b82f6",
        s=60,
        zorder=3,
        label="Model",
    )
    ax.plot(
        cal.sort_values("predicted_mean")["predicted_mean"],
        cal.sort_values("predicted_mean")["observed_rate"],
        color="#3b82f6",
        linewidth=1.4,
    )

    ax.set_title("Win Probability Calibration", fontsize=13, pad=12)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed win rate")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    legend = ax.legend(fontsize=9, framealpha=0.8, edgecolor=GRID_COLOR, labelcolor=TICK_COLOR)
    legend.get_frame().set_facecolor("white")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def brier_over_time(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    _apply_style(fig, ax)

    # Lower is better; shade the "good" region below 0.25 (random classifier baseline)
    ax.axhline(0.25, color="#475569", linewidth=0.8, linestyle="--", label="Random baseline (0.25)")
    ax.fill_between(df["date"], df["brier_score"], 0.25, where=df["brier_score"] <= 0.25, color="#22c55e", alpha=0.15)
    ax.plot(df["date"], df["brier_score"], color="#3b82f6", linewidth=1.8, label="Brier score")

    ax.set_title("Brier Score Over Time (lower is better)", fontsize=13, pad=12)
    ax.set_ylabel("Brier score")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    fig.autofmt_xdate(rotation=30)

    legend = ax.legend(fontsize=9, framealpha=0.8, edgecolor=GRID_COLOR, labelcolor=TICK_COLOR)
    legend.get_frame().set_facecolor("white")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "brier_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def mae_over_time(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    _apply_style(fig, ax)

    ax.plot(df["date"], df["mae"], color="#f59e0b", linewidth=1.8, label="MAE (runs)")
    ax.fill_between(df["date"], df["mae"], df["mae"].mean(), where=df["mae"] <= df["mae"].mean(), color="#f59e0b", alpha=0.12)

    mean_val = df["mae"].mean()
    ax.axhline(mean_val, color="#475569", linewidth=0.8, linestyle="--", label=f"Season avg ({mean_val:.2f})")

    ax.set_title("Mean Absolute Error Over Time", fontsize=13, pad=12)
    ax.set_ylabel("MAE (runs)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    fig.autofmt_xdate(rotation=30)

    legend = ax.legend(fontsize=9, framealpha=0.8, edgecolor=GRID_COLOR, labelcolor=TICK_COLOR)
    legend.get_frame().set_facecolor("white")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "mae_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def roi_by_segment(df: pd.DataFrame) -> None:
    latest = df.iloc[-1]

    segments = [
        ("Unders", latest["unders_roi"]),
        ("Overs", latest["overs_roi"]),
        ("Run line", latest["roi_run_line"]),
        ("Underdogs", latest["roi_underdogs"]),
        ("Favorites", latest["roi_favorites"]),
    ]
    labels = [s[0] for s in segments]
    values = [s[1] for s in segments]
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in values]

    fig, ax = plt.subplots(figsize=(8, 4))
    _apply_style(fig, ax)

    bars = ax.barh(labels, values, color=colors, height=0.55)
    ax.axvline(0, color="#475569", linewidth=0.8)

    for bar, val in zip(bars, values):
        x = bar.get_width()
        offset = 0.003 if x >= 0 else -0.003
        ha = "left" if x >= 0 else "right"
        ax.text(
            x + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.1%}",
            va="center",
            ha=ha,
            fontsize=9,
            color=TICK_COLOR,
        )

    ax.set_title("ROI by Bet Segment (season to date)", fontsize=13, pad=12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("ROI")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "roi_by_segment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    evals = load_evaluations()
    cal = load_calibration()

    accuracy_over_time(evals)
    equity_curve(evals)
    calibration_chart(cal)
    roi_by_segment(evals)
    brier_over_time(evals)
    mae_over_time(evals)

    print(f"Saved 6 charts to {OUT_DIR}/")


if __name__ == "__main__":
    main()
