"""Empirical weather logit shift (Option A).

Coefficients from v2/tools/fit_weather_coeffs.py on 2024-25 (363,853 PAs, park
+ batter-rate + pitcher-rate controls). Only statistically significant,
correct-sign effects are wired - insignificant noise is deliberately left out
(forcing it in would add variance the wrong way):

  HR : wind_signal +0.0105 (p<.001), temp_f_c +0.0103 (p<.001)
  2B : temp_f_c    +0.0031 (p<.001)   [wind ~0, dropped]
  3B : nothing      (no usable signal at 0.36% base rate)

Applied as an additive shift on the full 8-outcome logits, AFTER the park
shift, BEFORE softmax. wind_signal = wind_speed_mph * signed out-component;
temp_c = temp_f - 70. Dome games carry wind_signal=0 and temp_c=0 upstream.
"""
from __future__ import annotations

import numpy as np

from v2.data.pa_dataset import OUTCOMES

TEMP_CENTER = 70.0

_IDX = {o: i for i, o in enumerate(OUTCOMES)}

# (8,) coefficient vectors aligned to OUTCOMES order; zero everywhere we chose
# not to wire a term.
_WIND_VEC = np.zeros(len(OUTCOMES))
_TEMP_VEC = np.zeros(len(OUTCOMES))
_WIND_VEC[_IDX["HR"]] = 0.010451
_TEMP_VEC[_IDX["HR"]] = 0.010337
_TEMP_VEC[_IDX["2B"]] = 0.003147


def apply_weather_shift(
    full_logits: np.ndarray,
    wind_signal: np.ndarray,
    temp_c: np.ndarray,
) -> np.ndarray:
    """In-place add of the weather shift onto (N, 8) full logits.

    wind_signal, temp_c: (N,) per-row arrays (constant across a game's PAs).
    """
    full_logits += wind_signal[:, None] * _WIND_VEC[None, :]
    full_logits += temp_c[:, None] * _TEMP_VEC[None, :]
    return full_logits
