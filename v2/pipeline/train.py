"""Refit all three PyMC models (batter, pitcher, park) via fit_all.py.

Usage:
    python -m v2.pipeline.train --mode full [--archive]

--archive moves the current posteriors to posteriors/archive/{date}/ before
refitting so the prior trace can be restored if something goes wrong.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

POSTERIORS_DIR = Path(__file__).resolve().parents[1] / "bayesian" / "posteriors"
ARCHIVE_DIR = POSTERIORS_DIR / "archive"


def _archive_posteriors() -> None:
    tag = date.today().isoformat()
    dest = ARCHIVE_DIR / tag
    dest.mkdir(parents=True, exist_ok=True)
    for nc in POSTERIORS_DIR.glob("*.nc"):
        shutil.copy2(nc, dest / nc.name)
    diag = POSTERIORS_DIR / "diagnostics.json"
    if diag.exists():
        shutil.copy2(diag, dest / diag.name)
    print(f"[train] archived posteriors to {dest}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "incremental"], default="full")
    ap.add_argument("--archive", action="store_true")
    ap.add_argument("--start-year", type=int, default=2024)
    ap.add_argument("--end-year", type=int, default=date.today().year)
    args = ap.parse_args()

    if args.mode == "incremental":
        raise NotImplementedError("incremental warm-start is deferred to v2.1")

    if args.archive:
        _archive_posteriors()

    cmd = [
        sys.executable, "-m", "v2.bayesian.fit_all",
        "--start-year", str(args.start_year),
        "--end-year", str(args.end_year),
        "--save-traces",
    ]
    print(f"[train] running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
