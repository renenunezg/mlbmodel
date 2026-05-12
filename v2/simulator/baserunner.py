"""Empirical baserunner advancement and out-subtype sampling.

Two parquet tables (built by `build_advancement_table.py`):
- advancement.parquet: P(new_state, runs, outs_added | state, outs, outcome, subtype)
- out_subtype.parquet: P(subtype | state, outs)  (only meaningful when outcome=OUT)

Both lookups are vectorized: take (N,) arrays of the lookup key, return (N,) arrays of samples.

Known approximation (Phase 4): out_subtype is conditioned only on (state, outs), not on
batter or pitcher tendencies. League-mean GIDP/sac_fly rates per state.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from v2.data.pa_dataset import OUTCOMES

TABLES_DIR = Path(__file__).resolve().parent / "tables"
OUTCOME_TO_IDX = {o: i for i, o in enumerate(OUTCOMES)}
OUT_IDX = OUTCOME_TO_IDX["OUT"]

# Subtype string ↔ int. Keep a stable order; "_NA_" reserved for non-OUT outcomes.
SUBTYPE_ORDER = (
    "_NA_", "field_out", "force_out", "gidp", "dp", "tp", "fc",
    "sac_fly", "sac_fly_dp", "sac_bunt", "sac_bunt_dp", "roe", "ci", "k_dp",
)
SUBTYPE_TO_IDX = {s: i for i, s in enumerate(SUBTYPE_ORDER)}
N_SUBTYPES = len(SUBTYPE_ORDER)
NA_SUBTYPE_IDX = SUBTYPE_TO_IDX["_NA_"]

N_STATES = 8
N_OUTS = 3
N_OUTCOMES = len(OUTCOMES)


def _key_advancement(state: np.ndarray, outs: np.ndarray, outcome: np.ndarray, subtype: np.ndarray) -> np.ndarray:
    """Flatten (state, outs, outcome, subtype) → single int64 key."""
    return ((state * N_OUTS + outs) * N_OUTCOMES + outcome) * N_SUBTYPES + subtype


def _key_subtype(state: np.ndarray, outs: np.ndarray) -> np.ndarray:
    return state * N_OUTS + outs


@dataclass
class AdvancementTable:
    """Pre-baked flat lookup for advancement.

    For each unique (state, outs, outcome, subtype) key we store:
    - cdf[key]: cumulative probabilities, length = max number of distinct (new_state, runs, outs_added) entries.
    - new_state[key], runs[key], outs_added[key]: aligned arrays of outcomes for that key.
    - lengths[key]: how many entries are valid for that key.
    """
    starts: np.ndarray       # (N_KEYS+1,) offsets into the flat arrays
    cdf: np.ndarray          # (TOTAL,) cumulative probs per key, last entry per key is 1.0
    new_state: np.ndarray    # (TOTAL,) int8
    runs: np.ndarray         # (TOTAL,) int8
    outs_added: np.ndarray   # (TOTAL,) int8

    def sample(
        self,
        rng: np.random.Generator,
        state: np.ndarray,
        outs: np.ndarray,
        outcome: np.ndarray,
        subtype: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (new_state, runs, outs_added), each (N,) int."""
        keys = _key_advancement(
            state.astype(np.int64),
            outs.astype(np.int64),
            outcome.astype(np.int64),
            subtype.astype(np.int64),
        )
        starts = self.starts[keys]
        ends = self.starts[keys + 1]
        lengths = ends - starts

        if (lengths == 0).any():
            missing = keys[lengths == 0][:5]
            raise KeyError(
                f"Advancement table missing {(lengths == 0).sum()} key(s); first: {missing.tolist()}"
            )

        u = rng.random(size=len(keys))
        results_ns = np.empty(len(keys), dtype=np.int64)
        results_ru = np.empty(len(keys), dtype=np.int64)
        results_oa = np.empty(len(keys), dtype=np.int64)
        for i in range(len(keys)):
            s, e = starts[i], ends[i]
            j = s + np.searchsorted(self.cdf[s:e], u[i])
            j = min(j, e - 1)
            results_ns[i] = self.new_state[j]
            results_ru[i] = self.runs[j]
            results_oa[i] = self.outs_added[j]
        return results_ns, results_ru, results_oa


@dataclass
class OutSubtypeTable:
    """P(subtype | state, outs). Same flat-lookup shape as AdvancementTable."""
    starts: np.ndarray       # (N_STATES * N_OUTS + 1,)
    cdf: np.ndarray
    subtype: np.ndarray      # int subtype indices

    def sample(self, rng: np.random.Generator, state: np.ndarray, outs: np.ndarray) -> np.ndarray:
        keys = _key_subtype(state.astype(np.int64), outs.astype(np.int64))
        starts = self.starts[keys]
        ends = self.starts[keys + 1]
        if (ends - starts == 0).any():
            missing = keys[ends - starts == 0][:5]
            raise KeyError(f"Subtype table missing keys: {missing.tolist()}")
        u = rng.random(size=len(keys))
        out = np.empty(len(keys), dtype=np.int64)
        for i in range(len(keys)):
            s, e = starts[i], ends[i]
            j = s + np.searchsorted(self.cdf[s:e], u[i])
            j = min(j, e - 1)
            out[i] = self.subtype[j]
        return out


def _build_flat_lookup(df: pd.DataFrame, key_cols: list[str], n_keys: int, prob_col: str = "prob") -> tuple:
    """Take a long-format prob table, group by key_cols, return (starts, cdf, *value_arrays)."""
    # Make a single int key column
    if len(key_cols) == 4:
        key = _key_advancement(
            df[key_cols[0]].to_numpy(np.int64),
            df[key_cols[1]].to_numpy(np.int64),
            df[key_cols[2]].to_numpy(np.int64),
            df[key_cols[3]].to_numpy(np.int64),
        )
    else:  # 2 cols (subtype table)
        key = _key_subtype(df[key_cols[0]].to_numpy(np.int64), df[key_cols[1]].to_numpy(np.int64))
    df = df.assign(_key=key).sort_values(["_key", prob_col], ascending=[True, False]).reset_index(drop=True)

    counts = np.bincount(df["_key"].to_numpy(), minlength=n_keys)
    starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    # cdf within each key group
    cdf = np.empty(len(df), dtype=np.float64)
    for k in range(n_keys):
        s, e = starts[k], starts[k + 1]
        if e > s:
            cdf[s:e] = np.cumsum(df[prob_col].iloc[s:e].to_numpy())
            # ensure last entry is exactly 1 to avoid float drift
            cdf[e - 1] = 1.0
    return starts, cdf, df


def load_advancement_table() -> AdvancementTable:
    df = pd.read_parquet(TABLES_DIR / "advancement.parquet")
    df["subtype_idx"] = df["subtype_key"].map(SUBTYPE_TO_IDX).astype(np.int64)
    n_keys = N_STATES * N_OUTS * N_OUTCOMES * N_SUBTYPES
    starts, cdf, df = _build_flat_lookup(
        df, ["state", "outs", "outcome_idx", "subtype_idx"], n_keys
    )
    return AdvancementTable(
        starts=starts,
        cdf=cdf,
        new_state=df["new_state"].to_numpy(np.int64),
        runs=df["runs"].to_numpy(np.int64),
        outs_added=df["outs_added"].to_numpy(np.int64),
    )


def load_out_subtype_table() -> OutSubtypeTable:
    df = pd.read_parquet(TABLES_DIR / "out_subtype.parquet")
    df["subtype_idx"] = df["subtype_key"].map(SUBTYPE_TO_IDX).astype(np.int64)
    n_keys = N_STATES * N_OUTS
    starts, cdf, df = _build_flat_lookup(df, ["state", "outs"], n_keys)
    return OutSubtypeTable(
        starts=starts,
        cdf=cdf,
        subtype=df["subtype_idx"].to_numpy(np.int64),
    )


def sample_subtypes_for_outs(
    rng: np.random.Generator,
    outcome_idx: np.ndarray,
    state: np.ndarray,
    outs: np.ndarray,
    sub_table: OutSubtypeTable,
) -> np.ndarray:
    """Return a (N,) subtype-index array. Non-OUT rows get _NA_ (index 0)."""
    out = np.full(len(outcome_idx), NA_SUBTYPE_IDX, dtype=np.int64)
    mask = outcome_idx == OUT_IDX
    if mask.any():
        out[mask] = sub_table.sample(rng, state[mask], outs[mask])
    return out
