"""Unit tests for the baserunner advancement + out-subtype lookups."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from v2.simulator.baserunner import (
    OUTCOME_TO_IDX,
    SUBTYPE_TO_IDX,
    NA_SUBTYPE_IDX,
    load_advancement_table,
    load_out_subtype_table,
    sample_subtypes_for_outs,
)

skip_if_no_tables = pytest.mark.skipif(
    not (Path("v2/simulator/tables/advancement.parquet").exists()
         and Path("v2/simulator/tables/out_subtype.parquet").exists()),
    reason="advancement tables not built (run build_advancement_table.py first)",
)


@skip_if_no_tables
def test_hr_clears_bases_and_scores_all():
    """HR override: state=7 (bases loaded) → 4 runs, new_state=0, outs_added=0 every draw."""
    adv = load_advancement_table()
    rng = np.random.default_rng(0)
    n = 500
    state = np.full(n, 7, dtype=np.int64)
    outs = np.zeros(n, dtype=np.int64)
    outcome = np.full(n, OUTCOME_TO_IDX["HR"], dtype=np.int64)
    subtype = np.full(n, NA_SUBTYPE_IDX, dtype=np.int64)
    ns, ru, oa = adv.sample(rng, state, outs, outcome, subtype)
    assert (ns == 0).all()
    assert (ru == 4).all()
    assert (oa == 0).all()


@skip_if_no_tables
def test_walk_with_bases_loaded_forces_run():
    """BB with bases loaded → exactly 1 run (forced from 3rd), state stays full."""
    adv = load_advancement_table()
    rng = np.random.default_rng(3)
    n = 500
    state = np.full(n, 7, dtype=np.int64)
    outs = np.zeros(n, dtype=np.int64)
    outcome = np.full(n, OUTCOME_TO_IDX["BB"], dtype=np.int64)
    subtype = np.full(n, NA_SUBTYPE_IDX, dtype=np.int64)
    ns, ru, oa = adv.sample(rng, state, outs, outcome, subtype)
    # walk on bases loaded ALWAYS scores 1 (occasionally more if there's a wild pitch + walk, but rare)
    assert (ru >= 1).all()
    assert (ru == 1).mean() > 0.95
    assert (oa == 0).all()


@skip_if_no_tables
def test_subtype_sampler_only_outs():
    """Non-OUT outcomes get _NA_ subtype; OUT outcomes get a real subtype."""
    sub_table = load_out_subtype_table()
    rng = np.random.default_rng(4)
    n = 1000
    state = np.full(n, 1, dtype=np.int64)  # runner on 1B
    outs = np.zeros(n, dtype=np.int64)
    outcome = np.array([OUTCOME_TO_IDX["1B"]] * 500 + [OUTCOME_TO_IDX["OUT"]] * 500, dtype=np.int64)
    b_q = np.full(n, 2, dtype=np.int64)
    p_q = np.full(n, 2, dtype=np.int64)
    subt = sample_subtypes_for_outs(rng, outcome, state, outs, b_q, p_q, sub_table)
    assert (subt[:500] == NA_SUBTYPE_IDX).all()
    assert (subt[500:] != NA_SUBTYPE_IDX).all()
    # at runner-on-1st 0-outs, GIDP should be a meaningful slice (~10-15%)
    gidp_idx = SUBTYPE_TO_IDX["gidp"]
    gidp_share = (subt[500:] == gidp_idx).mean()
    assert 0.03 < gidp_share < 0.30, f"GIDP share {gidp_share:.3f} outside [3%, 30%]"


@skip_if_no_tables
def test_high_gb_pitcher_more_gidp():
    """At runner-on-1B, 0 outs: a high-GB pitcher (p_q=3) should induce more
    GIDP than a low-GB pitcher (p_q=0) against a neutral batter. Direction check
    on the stratification - the whole point of the variance fix."""
    sub_table = load_out_subtype_table()
    rng = np.random.default_rng(7)
    n = 20_000
    state = np.full(n, 1, dtype=np.int64)
    outs = np.zeros(n, dtype=np.int64)
    outcome = np.full(n, OUTCOME_TO_IDX["OUT"], dtype=np.int64)
    b_q = np.full(n, 2, dtype=np.int64)
    gidp_idx = SUBTYPE_TO_IDX["gidp"]

    lo = sample_subtypes_for_outs(
        rng, outcome, state, outs, b_q, np.zeros(n, np.int64), sub_table
    )
    hi = sample_subtypes_for_outs(
        rng, outcome, state, outs, b_q, np.full(n, 3, np.int64), sub_table
    )
    lo_gidp = (lo == gidp_idx).mean()
    hi_gidp = (hi == gidp_idx).mean()
    assert hi_gidp > lo_gidp, f"high-GB GIDP {hi_gidp:.3f} !> low-GB {lo_gidp:.3f}"


@skip_if_no_tables
def test_aggregate_runs_per_pa_sane():
    """Sample 50k random PAs uniformly across (state, outs, outcome). Aggregate runs/PA
    should be in a sane MLB-like range. Sanity check on the lookup wiring, not on per-cell accuracy."""
    adv = load_advancement_table()
    sub_table = load_out_subtype_table()
    rng = np.random.default_rng(5)
    n = 50_000

    # MLB-realistic outcome mix (rough): K 22%, BB 8%, HBP 1%, 1B 14%, 2B 4%, 3B 0.5%, HR 3%, OUT 47.5%
    probs = np.array([0.22, 0.08, 0.01, 0.14, 0.04, 0.005, 0.03, 0.475])
    outcomes = rng.choice(8, size=n, p=probs)
    # state distribution rough: empty 55%, 1B 18%, 2B 8%, 1B+2B 6%, 3B 1%, 1B+3B 3%, 2B+3B 1%, loaded 8%
    state_probs = np.array([0.55, 0.18, 0.08, 0.06, 0.01, 0.03, 0.01, 0.08])
    state = rng.choice(8, size=n, p=state_probs)
    outs = rng.choice(3, size=n)

    b_q = rng.integers(0, 4, size=n).astype(np.int64)
    p_q = rng.integers(0, 4, size=n).astype(np.int64)
    subtypes = sample_subtypes_for_outs(rng, outcomes, state, outs, b_q, p_q, sub_table)
    _, runs, _ = adv.sample(rng, state, outs, outcomes, subtypes)
    rpa = runs.mean()
    assert 0.08 < rpa < 0.18, f"runs/PA {rpa:.4f} outside sane range"
