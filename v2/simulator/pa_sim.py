"""Per-PA outcome simulator.

Given (batter_id, pitcher_id, vs_lhp, role, venue), returns one of the 8
OUTCOMES sampled from the categorical distribution implied by combining the
batter, pitcher, and park posterior means.

Logit assembly (free outcomes only, OUT is the reference at logit 0):

    logits[k]  = intercept[k]
               + batter_offset[b, k]
               + vs_lhp * platoon_offset[b, k]
               + pitcher_offset[p, role, k]
               + park_log[v] * WOBA_WEIGHTS[k]

then softmax over [logits, 0] across the 8 outcomes (with the 0 spliced into
the OUT slot at REF_IDX).
"""
from __future__ import annotations

import numpy as np

from v2.simulator.posteriors import (
    K_FREE,
    PosteriorMeans,
    REF_IDX,
    WOBA_VEC_FREE,
)
from v2.data.pa_dataset import OUTCOMES

# Indices into the K_FREE-vector for outcomes that need standalone slicing.
NON_REF_LABELS = [OUTCOMES[i] for i in range(len(OUTCOMES)) if i != REF_IDX]


def _build_full_logits(free_logits: np.ndarray) -> np.ndarray:
    """Splice the OUT reference (logit=0) back in at REF_IDX."""
    n = free_logits.shape[0]
    full = np.zeros((n, len(OUTCOMES)), dtype=np.float64)
    free_pos = 0
    for i in range(len(OUTCOMES)):
        if i == REF_IDX:
            continue
        full[:, i] = free_logits[:, free_pos]
        free_pos += 1
    return full


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _sample_categorical(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    """Vectorized categorical: one draw per row using cumsum + uniform."""
    cdf = np.cumsum(probs, axis=1)
    u = rng.random(size=(probs.shape[0], 1))
    return (u < cdf).argmax(axis=1).astype(np.int64)


def pa_logits_batch(
    pm: PosteriorMeans,
    batter_ids: np.ndarray,
    pitcher_ids: np.ndarray,
    vs_lhp: np.ndarray,
    roles: np.ndarray,
    venues: np.ndarray,
) -> np.ndarray:
    """Return (N, 8) full logits for a batch of PAs."""
    b = pm.encode_batter(batter_ids)
    p = pm.encode_pitcher(pitcher_ids)
    v = pm.encode_venue(venues)
    roles = np.asarray(roles, dtype=np.int64)
    vs_lhp = np.asarray(vs_lhp, dtype=np.float64)[:, None]

    free = (
        pm.intercept[None, :]
        + pm.batter_offset[b]
        + vs_lhp * pm.platoon_offset[b]
        + pm.pitcher_offset[p, roles]
        + pm.park_log[v][:, None] * WOBA_VEC_FREE[None, :]
    )
    return _build_full_logits(free)


def pa_probs_batch(
    pm: PosteriorMeans,
    batter_ids: np.ndarray,
    pitcher_ids: np.ndarray,
    vs_lhp: np.ndarray,
    roles: np.ndarray,
    venues: np.ndarray,
) -> np.ndarray:
    return _softmax(pa_logits_batch(pm, batter_ids, pitcher_ids, vs_lhp, roles, venues))


def simulate_pa_batch(
    rng: np.random.Generator,
    pm: PosteriorMeans,
    batter_ids: np.ndarray,
    pitcher_ids: np.ndarray,
    vs_lhp: np.ndarray,
    roles: np.ndarray,
    venues: np.ndarray,
) -> np.ndarray:
    """Sample one outcome per row. Returns int array of OUTCOMES indices."""
    probs = pa_probs_batch(pm, batter_ids, pitcher_ids, vs_lhp, roles, venues)
    return _sample_categorical(rng, probs)


def simulate_pa(
    rng: np.random.Generator,
    pm: PosteriorMeans,
    batter_id: int,
    pitcher_id: int,
    vs_lhp: bool,
    role: int,
    venue: str,
) -> int:
    """Single-PA convenience wrapper."""
    return int(
        simulate_pa_batch(
            rng,
            pm,
            np.array([batter_id], dtype=np.int64),
            np.array([pitcher_id], dtype=np.int64),
            np.array([vs_lhp], dtype=np.bool_),
            np.array([role], dtype=np.int64),
            np.array([venue]),
        )[0]
    )
