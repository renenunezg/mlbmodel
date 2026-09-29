"""Fail closed before publishing or consuming a production simulator bundle."""
from __future__ import annotations

import hashlib
import json
from datetime import date

import numpy as np
import pandas as pd
import xarray as xr

from v2.bayesian._common import POSTERIORS_DIR, evaluate_gate
from v2.simulator.baserunner import TABLE_VERSION, load_advancement_table, load_out_subtype_table
from v2.simulator.gb_quartiles import TABLES_DIR, load_gb_quartiles
from v2.simulator.posteriors import load_posterior_draws


def verify_artifacts() -> dict:
    diagnostics = json.loads((POSTERIORS_DIR / "diagnostics.json").read_text())
    window = diagnostics["training_window"]
    if window.get("game_type") != "R" or not diagnostics["all_gates_passed"]:
        raise ValueError("Artifacts must pass diagnostics and use regular-season training only")
    cutoff = date.fromisoformat(window["max_date"][:10])
    if cutoff >= date.today():
        raise ValueError("Production training cutoff must precede the scoring date")
    for name in ("batter", "pitcher", "park"):
        block = diagnostics[name]
        if not evaluate_gate(block["max_rhat"], block["min_ess_bulk"]) or block["n_divergent"]:
            raise ValueError(f"Unacceptable {name} sampling diagnostics")
    files = [POSTERIORS_DIR / "diagnostics.json"]
    for name in ("batter_skill", "pitcher_skill", "park_effects"):
        path = POSTERIORS_DIR / f"{name}.nc"
        files.append(path)
        with xr.open_dataset(path, group="posterior") as trace:
            if (trace.attrs.get("training_game_type") != "R"
                    or trace.attrs.get("training_max_date") != window["max_date"]):
                raise ValueError(f"Unverified training provenance in {name}")
    for draw in load_posterior_draws(np.random.default_rng(0), K=2):
        for name in ("intercept", "batter_offset", "platoon_offset", "pitcher_offset", "park_log"):
            if not np.isfinite(getattr(draw, name)).all():
                raise ValueError(f"Nonfinite posterior values in {name}")
    for name in ("advancement", "out_subtype"):
        path = TABLES_DIR / f"{name}.parquet"
        files.append(path)
        frame = pd.read_parquet(path)
        if frame.empty or not frame["table_version"].eq(TABLE_VERSION).all():
            raise ValueError(f"Incompatible {name} table")
        if pd.to_datetime(frame["training_max_date"]).max().date() > cutoff:
            raise ValueError(f"{name} training extends beyond the posterior cutoff")
    load_advancement_table()
    load_out_subtype_table()
    load_gb_quartiles()
    files.append(TABLES_DIR / "gb_quartiles.parquet")
    return {
        "model_version": "sim-v3", "training_window": window,
        "diagnostics": {name: diagnostics[name] for name in ("batter", "pitcher", "park")},
        "sha256": {str(path.relative_to(POSTERIORS_DIR.parents[2])):
                   hashlib.file_digest(path.open("rb"), "sha256").hexdigest() for path in files},
    }


if __name__ == "__main__":
    print(json.dumps(verify_artifacts(), indent=2))
