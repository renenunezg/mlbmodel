"""Shared utilities for the Phase 2 Bayesian skill layer."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from v2.data.pa_dataset import OUTCOMES

POSTERIORS_DIR = Path(__file__).resolve().parent / "posteriors"

# FanGraphs canonical wOBA weights.
WOBA_WEIGHTS = {
    "K":   0.0,
    "OUT": 0.0,
    "BB":  0.69,
    "HBP": 0.72,
    "1B":  0.89,
    "2B":  1.27,
    "3B":  1.62,
    "HR":  2.10,
}


@dataclass(frozen=True)
class ActorIndex:
    ids: np.ndarray
    n: int

    def encode(self, raw: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.ids, raw)

    @classmethod
    def from_series(cls, s: pd.Series) -> "ActorIndex":
        ids = np.sort(s.unique())
        return cls(ids=ids, n=len(ids))


def encode_outcomes(outcome_col: pd.Series) -> np.ndarray:
    cat_to_code = {o: i for i, o in enumerate(OUTCOMES)}
    return outcome_col.map(cat_to_code).to_numpy(dtype=np.int32)


def league_log_p(outcome_codes: np.ndarray) -> np.ndarray:
    counts = np.bincount(outcome_codes, minlength=len(OUTCOMES))
    p = counts / counts.sum()
    p = np.clip(p, 1e-6, None)
    return np.log(p)


def write_diagnostics(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def evaluate_gate(
    rhat_max: float,
    ess_min: float,
    threshold_rhat: float = 1.01,
    threshold_ess: float = 400,
) -> bool:
    return bool(rhat_max < threshold_rhat and ess_min > threshold_ess)
