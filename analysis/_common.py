"""Shared utilities for analysis scripts.

Read-only - do not write to the database from analysis code.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from backend.db import engine

ANALYSIS_DIR = Path(__file__).parent
OUTPUTS_DIR = ANALYSIS_DIR / "outputs"


def output_dir(script_name: str) -> Path:
    d = OUTPUTS_DIR / script_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_sql(query: str, params: dict | None = None) -> pd.DataFrame:
    return pd.read_sql(query, engine, params=params or {})


ACCENT = "#2d7a4f"
GRID_COLOR = "#e5e5e5"
AXIS_COLOR = "#d4d4d8"
TICK_COLOR = "#52525b"


def style_plot():
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 110,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": AXIS_COLOR,
        "axes.grid": True,
        "axes.grid.axis": "both",
        "grid.color": GRID_COLOR,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "xtick.color": TICK_COLOR,
        "ytick.color": TICK_COLOR,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "font.family": "monospace",
        "font.size": 11,
        "text.color": TICK_COLOR,
    })


def write_summary(out_dir: Path, title: str, body: str) -> None:
    (out_dir / "summary.md").write_text(f"# {title}\n\n{body}\n")
