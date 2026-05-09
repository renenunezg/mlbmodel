"""Sweep FORM_SIGMA values against the Phase 4 variance gate, with K-draw posteriors.

Phase 5 introduced per-game posterior draws in score_games (load_posterior_draws,
N_DRAWS=30 batches). The original FORM_SIGMA=0.18 was calibrated against
point-estimate posteriors. With real posterior parameter uncertainty now in the
loop, we expect to need a smaller sigma — possibly zero. This script reruns the
Phase 4 acceptance gate (200 stratified 2025 games, runs/team-game mean and var
within 5% of actual) at multiple sigma values and reports.

Usage:
    env/bin/python -m v2.tools.calibrate_form_sigma
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from v2.data.pa_dataset import EVENT_TO_OUTCOME, NON_PA_EVENTS
from v2.simulator import (
    BullpenQueue,
    GameInputs,
    build_queues_from_cache,
    load_advancement_table,
    load_out_subtype_table,
    load_posterior_draws,
    simulate_game,
)
from v2.tests.test_game_sim import (  # reuse helpers
    _build_game_inputs,
    _build_pa_frame,
    _classify_roles_2025,
)


SIGMAS = [0.0, 0.05, 0.10, 0.13, 0.15, 0.18]
N_DRAWS = 30
N_SIMS_PER_DRAW = 33  # → 990 sims/game (matches gate budget)
N_GAMES = 200
SEED = 20260509


def main():
    pa = _build_pa_frame(2025)
    role_lookup = _classify_roles_2025(pa)
    queues = build_queues_from_cache(2025)
    games = _build_game_inputs(pa, queues, role_lookup)
    if len(games) > N_GAMES:
        step = len(games) // N_GAMES
        games = games[::step][:N_GAMES]
    print(f"games for gate: {len(games)}")

    df_runs = pd.read_parquet(
        Path("cache/statcast_2025.parquet"),
        columns=["game_pk", "inning_topbot", "bat_score", "post_bat_score", "events"],
    )
    df_runs = df_runs[df_runs["events"].notna() & ~df_runs["events"].isin(NON_PA_EVENTS)]
    df_runs["runs"] = (df_runs["post_bat_score"].fillna(0) - df_runs["bat_score"].fillna(0)).clip(0, 4)
    actual_per_team = df_runs.groupby(["game_pk", "inning_topbot"])["runs"].sum().reset_index()
    actual_runs = actual_per_team["runs"].to_numpy(dtype=np.float64)
    actual_mean = float(actual_runs.mean())
    actual_var = float(actual_runs.var())
    print(f"actual 2025: mean={actual_mean:.4f}  var={actual_var:.4f}")

    rng = np.random.default_rng(SEED)
    draws = load_posterior_draws(rng, K=N_DRAWS)
    adv = load_advancement_table()
    sub_table = load_out_subtype_table()

    print(f"\n{'sigma':>6} | {'sim_mean':>9} {'mean_rel%':>10} | {'sim_var':>8} {'var_rel%':>9} | {'time':>6}")
    print("-" * 70)
    for sigma in SIGMAS:
        rng = np.random.default_rng(SEED)
        t0 = time.time()
        sim_runs = []
        for gp, gi, _ in games:
            for pm_k in draws:
                h, a = simulate_game(
                    rng, pm_k, adv, sub_table, gi,
                    n_sims=N_SIMS_PER_DRAW,
                    role_lookup=role_lookup,
                    form_sigma=sigma,
                )
                sim_runs.append(h)
                sim_runs.append(a)
        all_runs = np.concatenate(sim_runs)
        sim_mean = float(all_runs.mean())
        sim_var = float(all_runs.var())
        mean_rel = (sim_mean - actual_mean) / actual_mean * 100
        var_rel = (sim_var - actual_var) / actual_var * 100
        elapsed = time.time() - t0
        gate = "PASS" if abs(mean_rel) < 5 and abs(var_rel) < 5 else "FAIL"
        print(f"{sigma:>6.3f} | {sim_mean:>9.4f} {mean_rel:>+9.2f}% | {sim_var:>8.4f} {var_rel:>+8.2f}% | {elapsed:>5.0f}s  {gate}")


if __name__ == "__main__":
    main()
