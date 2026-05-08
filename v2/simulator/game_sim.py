"""Game-level Monte Carlo simulator: vectorized across N parallel sims of one game.

Vectorization strategy: for one game, run N sims in lockstep — at each tick (one PA in
ALL sims simultaneously), call simulate_pa_batch on the (N,) batch, then advance
baserunner state via AdvancementTable.sample. Per-sim state arrays are int64.

Termination & extras:
- regulation: 9 innings; game ends when bottom of inning ≥ 9 starts and home leads,
  or when home goes ahead at any PA in bottom of 9th+ (walkoff).
- extras: half-innings ≥ 10 start with state=0b010 (ghost runner on 2B).

Phase 4 known approximations (see CLAUDE.md / build_advancement_table.py docstrings):
- out_subtype is sampled from P(subtype | state, outs); ignores batter/pitcher tendencies
- mid-PA stolen bases / wild pitches not modeled
- bullpen queue uses actual-game RPs (backtest); not rest-aware
- relievers stay until they cross 9-outs OR 3-runs-allowed thresholds
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from v2.simulator.baserunner import (
    AdvancementTable,
    OutSubtypeTable,
    sample_subtypes_for_outs,
)
from v2.simulator.bullpen import BullpenQueue, should_pull_starter
from v2.simulator.pa_sim import _build_full_logits, _sample_categorical, _softmax, pa_logits_batch
from v2.simulator.posteriors import K_FREE, PosteriorMeans

GHOST_RUNNER_STATE = 2  # runner on 2B only

# reliever swap thresholds (after a reliever is in)
RELIEVER_PULL_OUTS = 9
RELIEVER_PULL_RUNS = 3

# Per-game form noise on logits. Mimics day-to-day variation and posterior
# parameter uncertainty that point-estimate posteriors miss. Applied to all 8
# outcomes (zero-sum across the vector to avoid mean-shifting the run rate).
# Sigma calibrated against the variance gate.
FORM_SIGMA = 0.18


@dataclass
class GameInputs:
    home_lineup: np.ndarray   # (9,) batter_ids, batting order 1..9
    away_lineup: np.ndarray
    home_queue: BullpenQueue  # home pitching staff (used in top halves)
    away_queue: BullpenQueue  # away pitching staff (used in bottom halves)
    venue: str                # 3-letter home_team code
    home_p_throws_lookup: dict[int, str]  # pitcher_id → "L"/"R"
    away_p_throws_lookup: dict[int, str]


def _queue_arrays(q: BullpenQueue, throws: dict[int, str], roles: dict[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pitcher_ids, throws_is_lhp_int, role_int) — one entry per slot in queue."""
    ids = [q.starter] + q.relievers
    if not ids:
        ids = [0]
    arr_ids = np.array(ids, dtype=np.int64)
    arr_lhp = np.array([1 if throws.get(int(p), "R") == "L" else 0 for p in ids], dtype=np.int64)
    # default: starter gets role=0, all subsequent role=1
    arr_role = np.array(
        [roles.get(int(p), 0 if i == 0 else 1) for i, p in enumerate(ids)],
        dtype=np.int64,
    )
    return arr_ids, arr_lhp, arr_role


def simulate_game(
    rng: np.random.Generator,
    pm: PosteriorMeans,
    adv: AdvancementTable,
    sub_table: OutSubtypeTable,
    inputs: GameInputs,
    n_sims: int = 1000,
    role_lookup: dict[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (home_runs, away_runs), each (n_sims,) int64."""
    role_lookup = role_lookup or {}

    # ---- precompute pitcher arrays per side ----
    home_pids, home_lhp, home_role = _queue_arrays(inputs.home_queue, inputs.home_p_throws_lookup, role_lookup)
    away_pids, away_lhp, away_role = _queue_arrays(inputs.away_queue, inputs.away_p_throws_lookup, role_lookup)
    home_max_idx = len(home_pids) - 1
    away_max_idx = len(away_pids) - 1

    home_arr = np.asarray(inputs.home_lineup, dtype=np.int64)
    away_arr = np.asarray(inputs.away_lineup, dtype=np.int64)
    venue_arr = np.array([inputs.venue], dtype=object).repeat(n_sims)

    # Per-sim "form" noise across all 8 outcomes (one per side). Zero-summed to
    # avoid systematic level-shift on the run rate. Held constant across the game.
    n_out = K_FREE + 1
    away_form_full = rng.normal(0.0, FORM_SIGMA, size=(n_sims, n_out))
    away_form_full -= away_form_full.mean(axis=1, keepdims=True)
    home_form_full = rng.normal(0.0, FORM_SIGMA, size=(n_sims, n_out))
    home_form_full -= home_form_full.mean(axis=1, keepdims=True)

    # ---- per-sim state ----
    state = np.zeros(n_sims, dtype=np.int64)
    outs = np.zeros(n_sims, dtype=np.int64)
    inning = np.ones(n_sims, dtype=np.int64)
    is_top = np.ones(n_sims, dtype=np.bool_)
    home_runs = np.zeros(n_sims, dtype=np.int64)
    away_runs = np.zeros(n_sims, dtype=np.int64)
    li_home = np.zeros(n_sims, dtype=np.int64)
    li_away = np.zeros(n_sims, dtype=np.int64)
    p_idx_home = np.zeros(n_sims, dtype=np.int64)
    p_idx_away = np.zeros(n_sims, dtype=np.int64)
    # since-last-swap counters per side
    p_outs_h = np.zeros(n_sims, dtype=np.int64)
    p_outs_a = np.zeros(n_sims, dtype=np.int64)
    p_runs_h = np.zeros(n_sims, dtype=np.int64)
    p_runs_a = np.zeros(n_sims, dtype=np.int64)
    p_pa_h = np.zeros(n_sims, dtype=np.int64)
    p_pa_a = np.zeros(n_sims, dtype=np.int64)
    done = np.zeros(n_sims, dtype=np.bool_)

    MAX_PAS = 250  # safety cap (real games end well before this)
    for _ in range(MAX_PAS):
        if done.all():
            break
        active = ~done

        # ---- batter & pitcher per active sim ----
        batter_ids = np.where(is_top, away_arr[li_away % 9], home_arr[li_home % 9])
        pitcher_ids = np.where(is_top, home_pids[p_idx_home], away_pids[p_idx_away])
        roles = np.where(is_top, home_role[p_idx_home], away_role[p_idx_away])
        vs_lhp = np.where(is_top, home_lhp[p_idx_home], away_lhp[p_idx_away]).astype(np.bool_)

        # Build logits once, add per-sim zero-sum form noise across all 8 outcomes,
        # softmax, sample.
        full_logits = pa_logits_batch(pm, batter_ids, pitcher_ids, vs_lhp, roles, venue_arr)
        form = np.where(is_top[:, None], away_form_full, home_form_full)
        full_logits = full_logits + form
        probs = _softmax(full_logits)
        outcomes = _sample_categorical(rng, probs)
        subtypes = sample_subtypes_for_outs(rng, outcomes, state, outs, sub_table)
        new_state, runs_scored, outs_added = adv.sample(rng, state, outs, outcomes, subtypes)

        # ---- post outcome (only on active sims) ----
        # batting team scores
        away_runs = np.where(active &  is_top, away_runs + runs_scored, away_runs)
        home_runs = np.where(active & ~is_top, home_runs + runs_scored, home_runs)
        # pitching side counters
        p_outs_h = np.where(active &  is_top, p_outs_h + outs_added, p_outs_h)
        p_outs_a = np.where(active & ~is_top, p_outs_a + outs_added, p_outs_a)
        p_runs_h = np.where(active &  is_top, p_runs_h + runs_scored, p_runs_h)
        p_runs_a = np.where(active & ~is_top, p_runs_a + runs_scored, p_runs_a)
        p_pa_h   = np.where(active &  is_top, p_pa_h + 1, p_pa_h)
        p_pa_a   = np.where(active & ~is_top, p_pa_a + 1, p_pa_a)

        outs = np.where(active, outs + outs_added, outs)
        state = np.where(active, new_state, state)
        li_away = np.where(active &  is_top, li_away + 1, li_away)
        li_home = np.where(active & ~is_top, li_home + 1, li_home)

        # ---- walkoff (bottom of 9+, home now ahead) ----
        walkoff = active & (~is_top) & (inning >= 9) & (home_runs > away_runs)
        done = done | walkoff

        # ---- inning-end ----
        inning_end = active & (outs >= 3)
        if inning_end.any():
            flipped_to_bot = inning_end &  is_top
            flipped_to_top = inning_end & ~is_top
            # flip halves
            new_is_top = is_top.copy()
            new_is_top[flipped_to_bot] = False
            new_is_top[flipped_to_top] = True
            is_top = new_is_top
            inning = np.where(flipped_to_top, inning + 1, inning)
            outs = np.where(inning_end, 0, outs)
            # ghost runner if entering half-inning ≥ 10
            entering_extras = inning_end & (inning >= 10)
            state = np.where(entering_extras, GHOST_RUNNER_STATE, np.where(inning_end, 0, state))

            # skip bottom of 9+ if home leads
            skip_bot = flipped_to_bot & (inning >= 9) & (home_runs > away_runs)
            done = done | skip_bot
            # game over if just finished bottom of 9+ with non-tied score
            game_over = flipped_to_top & (inning - 1 >= 9) & (home_runs != away_runs)
            done = done | game_over

        # ---- pitcher swap (vectorized rule check on per-pitcher counters) ----
        # starter pull: starter is in iff p_idx == 0
        # vectorized version of should_pull_starter
        def pull_mask(p_idx, p_outs, p_runs, p_pa):
            starter_in = p_idx == 0
            starter_pull = starter_in & (
                (p_outs >= 18)
                | (p_runs >= 6)
                | ((p_outs >= 12) & (p_runs >= 4))
                | (p_pa >= 24)
            )
            reliever_in = p_idx > 0
            reliever_pull = reliever_in & ((p_outs >= RELIEVER_PULL_OUTS) | (p_runs >= RELIEVER_PULL_RUNS))
            return starter_pull | reliever_pull

        swap_h = active & pull_mask(p_idx_home, p_outs_h, p_runs_h, p_pa_h) & (p_idx_home < home_max_idx)
        swap_a = active & pull_mask(p_idx_away, p_outs_a, p_runs_a, p_pa_a) & (p_idx_away < away_max_idx)

        if swap_h.any():
            p_idx_home = np.where(swap_h, p_idx_home + 1, p_idx_home)
            p_outs_h = np.where(swap_h, 0, p_outs_h)
            p_runs_h = np.where(swap_h, 0, p_runs_h)
            p_pa_h = np.where(swap_h, 0, p_pa_h)
        if swap_a.any():
            p_idx_away = np.where(swap_a, p_idx_away + 1, p_idx_away)
            p_outs_a = np.where(swap_a, 0, p_outs_a)
            p_runs_a = np.where(swap_a, 0, p_runs_a)
            p_pa_a = np.where(swap_a, 0, p_pa_a)

    return home_runs, away_runs
