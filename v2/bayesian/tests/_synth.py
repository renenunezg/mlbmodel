"""Synthetic PA generator for fast Bayesian skill-model unit tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from v2.data.pa_dataset import OUTCOMES

REF_IDX = OUTCOMES.index("OUT")
NON_REF_IDX = [i for i in range(len(OUTCOMES)) if i != REF_IDX]
LEAGUE_LOG_P = np.log(np.array([0.22, 0.085, 0.011, 0.14, 0.045, 0.005, 0.034, 0.46]))
LEAGUE_LOGITS = LEAGUE_LOG_P[NON_REF_IDX] - LEAGUE_LOG_P[REF_IDX]


def _softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _logit_to_full(logit_free: np.ndarray) -> np.ndarray:
    n = logit_free.shape[0]
    full = np.zeros((n, len(OUTCOMES)))
    free_iter = iter(range(len(NON_REF_IDX)))
    for i in range(len(OUTCOMES)):
        if i == REF_IDX:
            continue
        full[:, i] = logit_free[:, next(free_iter)]
    return full


def synth_batter_pa(
    n_batters: int = 60,
    pa_per_cell_mean: int = 120,
    sigma_main: float = 0.3,
    sigma_platoon: float = 0.2,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    K = len(NON_REF_IDX)
    main = rng.normal(0, sigma_main, (n_batters, K))
    platoon = rng.normal(0, sigma_platoon, (n_batters, K))
    rows = []
    for b in range(n_batters):
        for hand_code, hand_str in enumerate(("R", "L")):
            n_pa = max(20, int(rng.poisson(pa_per_cell_mean)))
            logit_free = LEAGUE_LOGITS + main[b] + (hand_code * platoon[b])
            full = _logit_to_full(logit_free[None, :])[0]
            p = _softmax(full)
            outcomes_drawn = rng.choice(len(OUTCOMES), size=n_pa, p=p)
            for o in outcomes_drawn:
                rows.append({
                    "batter": 100000 + b,
                    "pitcher": 200000 + (b % 5),
                    "stand": "R",
                    "p_throws": hand_str,
                    "outcome": OUTCOMES[o],
                    "home_team": "LAD",
                    "game_pk": 1,
                    "inning": 1,
                    "inning_topbot": "Top",
                })
    df = pd.DataFrame(rows)
    truth = {"main": main, "platoon": platoon}
    return df, truth


def synth_pitcher_pa(
    n_sp: int = 30,
    n_rp: int = 30,
    pa_per_sp: int = 400,
    pa_per_rp: int = 100,
    sigma_sp: float = 0.4,
    sigma_rp: float = 0.2,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    K = len(NON_REF_IDX)
    sp_eff = rng.normal(0, sigma_sp, (n_sp, K))
    rp_eff = rng.normal(0, sigma_rp, (n_rp, K))
    rows = []
    for i, eff in enumerate(sp_eff):
        n_pa = max(50, int(rng.poisson(pa_per_sp)))
        logit_free = LEAGUE_LOGITS + eff
        p = _softmax(_logit_to_full(logit_free[None, :])[0])
        for o in rng.choice(len(OUTCOMES), size=n_pa, p=p):
            rows.append({
                "batter": 100000 + (i % 10),
                "pitcher": 300000 + i,
                "p_throws": "R",
                "outcome": OUTCOMES[o],
                "home_team": "LAD", "game_pk": 1000 + i,
                "inning": 1, "inning_topbot": "Top",
            })
    for i, eff in enumerate(rp_eff):
        n_pa = max(20, int(rng.poisson(pa_per_rp)))
        logit_free = LEAGUE_LOGITS + eff
        p = _softmax(_logit_to_full(logit_free[None, :])[0])
        for o in rng.choice(len(OUTCOMES), size=n_pa, p=p):
            rows.append({
                "batter": 100000 + (i % 10),
                "pitcher": 400000 + i,
                "p_throws": "R",
                "outcome": OUTCOMES[o],
                "home_team": "LAD", "game_pk": 5000 + i,
                "inning": 7, "inning_topbot": "Top",
            })
    df = pd.DataFrame(rows)
    truth = {"sp_eff": sp_eff, "rp_eff": rp_eff}
    return df, truth
