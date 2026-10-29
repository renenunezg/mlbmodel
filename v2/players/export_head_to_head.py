"""Export every published grade-versus-projection cell with the projection captures behind it.

One CSV row per cell of the published forward evaluation (mlb.player_value_evaluation.forward,
or the evaluation block of a local model file): pool, comparison, window, season or bucket,
system, both rank correlations, the gap and its 95% interval, wins, checkpoints and mean n, plus
the Wayback capture id of every cached FanGraphs page that cell used (``season:snapshot``;
``season:missing`` when no page is cached). Read-only.

Usage:
    python -m v2.players.export_head_to_head --out head_to_head.csv [--model PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from v2.players.forward import CACHE_DIR, ROS_SYSTEMS, projection_page
from v2.players.model import MODEL_VERSION, ValueModel

CELL_KEYS = ("rho_grade", "rho_system", "diff", "ci95", "wins", "checkpoints", "mean_n")
WINDOWS = ("preseason", "in_season", "by_season_preseason", "rest_of_season_snapshots", "buckets")


def _snapshot(path: Path) -> str:
    return json.loads(path.read_text())["snapshot"] if path.exists() else "missing"


def captures(system: str, role: str, seasons: list[int]) -> str:
    """Capture ids of the pages a cell compared against: preseason pages, or every dated rest-of-season snapshot."""
    if system in ROS_SYSTEMS:
        pages = sorted(CACHE_DIR.glob(f"fangraphs_projection_{ROS_SYSTEMS[system]}_{role}_*_*.json"))
        pages = [p for p in pages if int(p.stem.split("_")[-2]) in seasons]
        return ";".join(f"{p.stem.split('_')[-1]}:{_snapshot(p)}" for p in pages)
    return ";".join(f"{season}:{_snapshot(projection_page(system, role, season))}" for season in seasons)


def cells(forward: dict) -> pd.DataFrame:
    seasons = [int(s) for s in forward["seasons"]]
    rows = []

    def emit(pool, comparison, spec, window, slice_, system, cell):
        role = "bat" if pool == "hitter" else "pit"
        used = [int(slice_)] if window == "by_season_preseason" else seasons
        low, high = cell["ci95"]
        rows.append({
            "pool": pool, "comparison": comparison, "target": spec.get("target"),
            "system_statistic": spec.get("system_statistic"), "window": window, "slice": slice_, "system": system,
            "rho_grade": cell["rho_grade"], "rho_system": cell["rho_system"], "diff": cell["diff"],
            "ci95_low": low, "ci95_high": high, "wins": cell["wins"], "checkpoints": cell["checkpoints"],
            "mean_n": cell["mean_n"], "captures": captures(system, role, used),
        })

    def walk(pool, comparison, spec, window, node, slice_=""):
        for key, value in node.items():
            if not isinstance(value, dict):
                continue
            if set(CELL_KEYS) <= set(value):
                emit(pool, comparison, spec, window, slice_, key, value)
            else:
                walk(pool, comparison, spec, window, value, f"{slice_}/{key}".strip("/"))

    for pool, comparisons in forward["pools"].items():
        for comparison, spec in comparisons.items():
            for window in WINDOWS:
                if isinstance(spec.get(window), dict):
                    walk(pool, comparison, spec, window, spec[window])
    return pd.DataFrame(rows)


def published_forward() -> tuple[str, dict]:
    from sqlalchemy import text

    from backend.db import engine

    with engine.connect() as conn:
        fitted_at, forward = conn.execute(text(
            "SELECT fitted_at, forward FROM player_value_evaluation WHERE model_version = :v"),
            {"v": MODEL_VERSION}).one()
    return str(fitted_at), forward


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path, help="Local model JSON instead of the published row")
    args = ap.parse_args()
    if args.model:
        model = ValueModel.load(args.model)
        fitted_at, forward = model.fitted_at, model.evaluation["forward"]
    else:
        fitted_at, forward = published_forward()
    frame = cells(forward).assign(model_version=MODEL_VERSION, fitted_at=fitted_at,
                                  computed_at=forward.get("computed_at"))
    frame.to_csv(args.out, index=False)
    print(f"{len(frame)} cells -> {args.out}")


if __name__ == "__main__":
    main()
