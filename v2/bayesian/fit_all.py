"""Phase 2 orchestrator: fit batter, pitcher, and park models in sequence.

Loads the PA frame once, fits each skill posterior, then writes a consolidated
diagnostics.json. Exits 0 only if all three models clear the gate.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import arviz as az

from v2.data.pa_dataset import load_pa_dataset
from v2.bayesian._common import POSTERIORS_DIR, evaluate_gate, write_diagnostics
from v2.bayesian import batter_skill, pitcher_skill, park_effects

SCHEMA_VERSION = 1


def _diag_block(idata: az.InferenceData, summarize_fn, fit_seconds: float) -> dict:
    diag = summarize_fn(idata)
    n_div = int(idata.sample_stats["diverging"].sum().item()) if "diverging" in idata.sample_stats else 0
    return {
        **diag,
        "n_divergent": n_div,
        "fit_seconds": fit_seconds,
        "fit_minutes": fit_seconds / 60,
        "gate_passed": evaluate_gate(diag["max_rhat"], diag["min_ess_bulk"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2024)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=2500)
    parser.add_argument("--park-draws", type=int, default=4000,
                        help="Park stage gets more draws since its parameter set is tiny "
                             "but autocorrelates without enough samples.")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--output-dir", type=Path, default=POSTERIORS_DIR)
    parser.add_argument("--save-traces", action="store_true")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Optional PA subsample for smoke runs.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fit_all] loading PAs {args.start_year}-{args.end_year}...")
    t0 = time.time()
    pa_df = load_pa_dataset(args.start_year, args.end_year)
    print(f"  loaded {len(pa_df):,} PAs in {time.time()-t0:.1f}s")
    if args.subsample:
        pa_df = pa_df.sample(args.subsample, random_state=args.seed).reset_index(drop=True)
        print(f"  subsampled to {len(pa_df):,} PAs")

    print(f"[fit_all] === pitcher ===")
    pit_pa, dropped = pitcher_skill.filter_position_player_pitching(pa_df)
    print(f"  filtered position-player-pitching: dropped {len(dropped)}")
    pit_idata, pit_meta, pit_elapsed = pitcher_skill.fit(
        pit_pa, draws=args.draws, tune=args.tune, chains=args.chains, random_seed=args.seed
    )
    pit_diag = _diag_block(pit_idata, pitcher_skill.summarize, pit_elapsed)
    print(f"  rhat={pit_diag['max_rhat']:.4f}  ess={pit_diag['min_ess_bulk']:.0f}  "
          f"div={pit_diag['n_divergent']}  gate={pit_diag['gate_passed']}  ({pit_elapsed/60:.1f} min)")

    print(f"[fit_all] === batter ===")
    pitcher_intercept = pit_idata.posterior["intercept"].mean(("chain", "draw")).values
    bat_idata, bat_meta, bat_elapsed = batter_skill.fit(
        pa_df, frozen_intercept=pitcher_intercept,
        draws=args.draws, tune=args.tune, chains=args.chains, random_seed=args.seed
    )
    bat_diag = _diag_block(bat_idata, batter_skill.summarize, bat_elapsed)
    print(f"  rhat={bat_diag['max_rhat']:.4f}  ess={bat_diag['min_ess_bulk']:.0f}  "
          f"div={bat_diag['n_divergent']}  gate={bat_diag['gate_passed']}  anchored=True  ({bat_elapsed/60:.1f} min)")

    print(f"[fit_all] === park ===")
    woba_pred, pa_for_park = park_effects.predict_woba_per_pa(pa_df, bat_idata, pit_idata)
    venue_df = park_effects.venue_residuals(pa_for_park, woba_pred)
    park_idata, park_meta, park_elapsed = park_effects.fit(
        venue_df, draws=args.park_draws, tune=args.tune, chains=args.chains, random_seed=args.seed
    )
    park_diag = _diag_block(park_idata, park_effects.summarize, park_elapsed)
    print(f"  rhat={park_diag['max_rhat']:.4f}  ess={park_diag['min_ess_bulk']:.0f}  "
          f"div={park_diag['n_divergent']}  gate={park_diag['gate_passed']}  ({park_elapsed/60:.1f} min)")

    if args.save_traces:
        bat_idata.to_netcdf(args.output_dir / "batter_skill.nc")
        pit_idata.to_netcdf(args.output_dir / "pitcher_skill.nc")
        park_idata.to_netcdf(args.output_dir / "park_effects.nc")
        print(f"  saved traces to {args.output_dir}")

    all_passed = bat_diag["gate_passed"] and pit_diag["gate_passed"] and park_diag["gate_passed"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "training_window": {
            "start_year": args.start_year,
            "end_year": args.end_year,
            "n_pa": int(len(pa_df)),
        },
        "sampler": {"chains": args.chains, "draws": args.draws, "tune": args.tune},
        "batter": {**bat_diag, "n_batters": bat_meta["n_batters"],
                   "n_pa_vs_lhp": bat_meta["n_pa_vs_lhp"], "n_pa_vs_rhp": bat_meta["n_pa_vs_rhp"]},
        "pitcher": {**pit_diag, "n_pitchers": pit_meta["n_pitchers"],
                    "n_sp": pit_meta["n_sp"], "n_rp": pit_meta["n_rp"],
                    "n_dropped_position_player_pitchers": len(dropped)},
        "park": {**park_diag, "venues": park_meta["venues"],
                 "n_per_venue": park_meta["n_per_venue"]},
        "all_gates_passed": all_passed,
    }
    write_diagnostics(args.output_dir / "diagnostics.json", payload)
    print(f"[fit_all] wrote {args.output_dir / 'diagnostics.json'}  all_gates_passed={all_passed}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
