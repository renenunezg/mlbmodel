"""Shared utilities for analysis scripts.

Read-only — do not write to the database from analysis code.
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


def style_plot():
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 11,
    })


def write_summary(out_dir: Path, title: str, body: str) -> None:
    (out_dir / "summary.md").write_text(f"# {title}\n\n{body}\n")
