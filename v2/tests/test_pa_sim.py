"""Phase 3 acceptance gate + sanity tests for the PA simulator."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

POSTERIOR_FILES = [
    Path("v2/bayesian/posteriors/batter_skill.nc"),
    Path("v2/bayesian/posteriors/pitcher_skill.nc"),
    Path("v2/bayesian/posteriors/park_effects.nc"),
]
CACHE_2025 = Path("cache/statcast_2025.parquet")

skip_if_no_posteriors = pytest.mark.skipif(
    not all(p.exists() for p in POSTERIOR_FILES),
    reason="Phase 2 posterior NetCDF files not present",
)
skip_if_no_2025_cache = pytest.mark.skipif(
    not CACHE_2025.exists(), reason="2025 statcast cache not present",
)


# ---------- helpers ------------------------------------------------------


def _load():
    from v2.simulator import load_posteriors
    return load_posteriors()


def _classify_roles_2025(pa_df: pd.DataFrame) -> dict[int, int]:
    """Map pitcher_id -> 0 (SP) or 1 (RP) using the same heuristic as
    pitcher_skill.classify_roles. 0=SP slot, 1=RP slot in pitcher_offset."""
    inning1 = pa_df[pa_df["inning"] == 1].copy()
    starter_side = np.where(inning1["inning_topbot"] == "Top", "away", "home")
    inning1 = inning1.assign(side=starter_side)
    starters = (
        inning1.groupby(["game_pk", "side"])["pitcher"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
        .dropna()
        .astype("int64")
    )
    starts = starters.value_counts()
    games = pa_df.groupby("pitcher")["game_pk"].nunique()
    share = (starts / games).fillna(0.0)
    sp = set(share[share >= 0.5].index)
    return {int(p): (0 if int(p) in sp else 1) for p in games.index}


# ---------- sanity tests -------------------------------------------------


@skip_if_no_posteriors
def test_intercepts_match_within_tolerance():
    pm = _load()
    # Loose tolerance: the two fits use slightly different PA pools so a small
    # gap is expected. The binding check is the 1pp acceptance gate below.
    assert pm.intercept_diff < 0.30, (
        f"batter and pitcher intercepts disagree by {pm.intercept_diff:.4f}"
    )


@skip_if_no_posteriors
def test_unknown_actor_falls_back_to_league_mean():
    """An unknown batter+pitcher at a neutral park must produce probs equal to
    softmax(intercept) (with OUT logit = 0)."""
    from v2.simulator.pa_sim import _build_full_logits, _softmax
    pm = _load()
    expected = _softmax(_build_full_logits(pm.intercept[None, :]))[0]

    from v2.simulator import pa_probs_batch
    probs = pa_probs_batch(
        pm,
        batter_ids=np.array([-1], dtype=np.int64),
        pitcher_ids=np.array([-1], dtype=np.int64),
        vs_lhp=np.array([False]),
        roles=np.array([0], dtype=np.int64),
        venues=np.array(["__UNKNOWN__"]),
    )[0]
    np.testing.assert_allclose(probs, expected, atol=1e-12)


# ---------- acceptance gate ----------------------------------------------


@skip_if_no_posteriors
@skip_if_no_2025_cache
def test_league_replay_within_1pp():
    """Replay 2025 PAs through the simulator. Aggregate K%/BB%/HR%/BABIP must
    land within 1pp of actual MLB 2025 rates."""
    from v2.data.pa_dataset import OUTCOMES, load_pa_dataset
    from v2.simulator import load_posteriors, simulate_pa_batch

    pa = load_pa_dataset(2025, 2025)
    pm = load_posteriors()

    role_map = _classify_roles_2025(pa)
    roles = pa["pitcher"].map(role_map).fillna(1).astype(np.int64).to_numpy()
    batter_ids = pa["batter"].astype("int64").to_numpy()
    pitcher_ids = pa["pitcher"].astype("int64").to_numpy()
    vs_lhp = (pa["p_throws"].to_numpy() == "L")
    venues = pa["home_team"].astype(str).to_numpy()

    rng = np.random.default_rng(20260508)
    sim = simulate_pa_batch(rng, pm, batter_ids, pitcher_ids, vs_lhp, roles, venues)

    actual_codes = pa["outcome"].map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy()

    def rates(codes: np.ndarray) -> dict[str, float]:
        c = np.bincount(codes, minlength=len(OUTCOMES))
        n = c.sum()
        K, BB, HBP, S, D, T, HR, OUT = c  # noqa: E741
        babip_denom = n - K - BB - HBP - HR
        return {
            "K%":   K / n,
            "BB%":  BB / n,
            "HR%":  HR / n,
            "BABIP": (S + D + T) / babip_denom if babip_denom else 0.0,
        }

    sim_r = rates(sim)
    act_r = rates(actual_codes)
    diffs = {k: abs(sim_r[k] - act_r[k]) for k in sim_r}
    print(f"\n  PAs: {len(pa):,}")
    for k in sim_r:
        print(f"  {k:6s}  sim={sim_r[k]:.4f}  actual={act_r[k]:.4f}  diff={diffs[k]:.4f}")

    for k, d in diffs.items():
        assert d < 0.01, f"{k} diff {d:.4f} exceeds 1pp gate"
