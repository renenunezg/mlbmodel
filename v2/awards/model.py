"""A conditional logit per award: a softmax over each race's candidates.

The ballot races fit vote shares and their winners are read through a sharpening
exponent; Gold Gloves and the reliever award fit winners, since no votes exist.
The penalty and exponent come from leaving a training season out in turn, so a
backtest season never tunes its own forecast.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import log_softmax, softmax

PENALTIES = np.logspace(-2, 2, 9)
MAX_SHARPEN = 8.0


@dataclass(frozen=True)
class Race:
    season: int
    x: np.ndarray
    y: np.ndarray  # Share of the vote, summing to one.
    winner: int  # Index of the winner within the pool, -1 when he fell outside it.


@dataclass(frozen=True)
class Fit:
    beta: np.ndarray
    scale: np.ndarray
    penalty: float
    sharpen: float

    def share(self, x: np.ndarray) -> np.ndarray:
        return softmax((x / self.scale) @ self.beta)

    def win(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.sharpen * ((x / self.scale) @ self.beta))


def _coefficients(races: list[Race], scale: np.ndarray, penalty: float) -> np.ndarray:
    scaled = [(r.x / scale, r.y) for r in races]

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = penalty / 2 * beta @ beta, penalty * beta
        for x, y in scaled:
            logits = x @ beta
            value -= y @ log_softmax(logits)
            gradient = gradient + x.T @ (softmax(logits) - y)
        return value, gradient

    result = minimize(objective, np.zeros(scale.size), jac=True, method="L-BFGS-B")
    if not result.success and "ABNORMAL" in str(result.message):
        raise RuntimeError(f"Award fit failed: {result.message}")
    return result.x


def _scale(races: list[Race]) -> np.ndarray:
    deviation = np.vstack([r.x for r in races]).std(axis=0)
    return np.where(deviation > 0, deviation, 1.0)


def _held_out(races: list[Race], scale: np.ndarray, penalty: float) -> list[tuple[Race, np.ndarray]]:
    out = []
    for season in sorted({r.season for r in races}):
        beta = _coefficients([r for r in races if r.season != season], scale, penalty)
        out.extend((r, (r.x / scale) @ beta) for r in races if r.season == season)
    return out


def fit(races: list[Race]) -> Fit:
    scale = _scale(races)
    best, best_score, best_held = PENALTIES[0], np.inf, []
    for penalty in PENALTIES:
        held = _held_out(races, scale, penalty)
        score = -sum(r.y @ log_softmax(logits) for r, logits in held)
        if score < best_score:
            best, best_score, best_held = penalty, score, held
    won = [(r, logits) for r, logits in best_held if r.winner >= 0]
    sharpen = minimize_scalar(lambda t: -sum(log_softmax(t * logits)[r.winner] for r, logits in won),
                              bounds=(0.25, MAX_SHARPEN), method="bounded").x
    return Fit(_coefficients(races, scale, best), scale, float(best), float(sharpen))


def backtest(races: list[Race], first_season: int, baseline_column: int) -> dict:
    """Train on earlier seasons only. A winner outside the pool is a miss."""
    rows = []
    for season in sorted({r.season for r in races if r.season >= first_season}):
        model = fit([r for r in races if r.season < season])
        for race in (r for r in races if r.season == season):
            probabilities = model.win(race.x)
            won = race.winner >= 0
            rows.append({"season": season, "hit": won and int(probabilities.argmax()) == race.winner,
                         "baseline_hit": won and int(race.x[:, baseline_column].argmax()) == race.winner,
                         "log_loss": -np.log(max(probabilities[race.winner], 1e-6)) if won else -np.log(1e-6),
                         "uniform_log_loss": np.log(len(race.y)), "in_pool": won,
                         "win_probability": float(probabilities.max())})
    n = len(rows)
    return {"races": n, "seasons": sorted({r["season"] for r in rows}), "hits": sum(r["hit"] for r in rows),
            "baseline_hits": sum(r["baseline_hit"] for r in rows), "in_pool": sum(r["in_pool"] for r in rows),
            "log_loss": float(np.mean([r["log_loss"] for r in rows])),
            "uniform_log_loss": float(np.mean([r["uniform_log_loss"] for r in rows])),
            "mean_top_probability": float(np.mean([r["win_probability"] for r in rows]))}
