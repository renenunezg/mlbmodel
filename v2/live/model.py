"""Deterministic win expectancy from empirical remaining-inning run distributions.

Fit only complete, uncensored half innings. Counts shrink toward the corresponding
base/out distribution. Future half innings are convolved, with tied extra innings
solved as a geometric continuation rather than stopped at an arbitrary inning.
This is a league-average game-state model, not a player or market model.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GameState:
    inning: int
    top: bool
    outs: int
    bases: int
    home_score: int
    away_score: int
    balls: int = 0
    strikes: int = 0
    automatic_runner: bool = True

    def __post_init__(self):
        integers = (self.inning, self.outs, self.bases, self.home_score,
                    self.away_score, self.balls, self.strikes)
        if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in integers):
            raise ValueError("Game-state fields must be integers")
        if not (self.inning >= 1 and 0 <= self.outs <= 2 and 0 <= self.bases <= 7
                and min(self.home_score, self.away_score) >= 0
                and 0 <= self.balls <= 3 and 0 <= self.strikes <= 2):
            raise ValueError("Invalid active game state")
        if not isinstance(self.top, bool) or not isinstance(self.automatic_runner, bool):
            raise ValueError("Half inning and extra-inning rule must be explicit booleans")


class WinExpectancy:
    def __init__(self, distributions: np.ndarray, metadata: dict):
        # Side (away/home), outs, bases, balls, strikes, remaining runs.
        self.distributions = np.asarray(distributions, dtype=float)
        if (self.distributions.ndim != 6 or self.distributions.shape[:5] != (2, 3, 8, 4, 3)
                or not np.isfinite(self.distributions).all()
                or (self.distributions < 0).any()
                or not np.allclose(self.distributions.sum(axis=-1), 1)):
            raise ValueError("Invalid remaining-runs artifact")
        self.metadata = metadata

    @classmethod
    def fit(cls, states: pd.DataFrame, *, use_count: bool = True, prior_strength: float = 100):
        if states.empty or prior_strength <= 0:
            raise ValueError("Training states and positive shrinkage are required")
        # Home 9th+ innings stop on a walkoff, so their observed run totals are censored.
        frame = states.loc[states.top | (states.inning < 9)].copy()
        # Regulation states provide an uncensored distribution for any occupied bases,
        # including the automatic runner. Extras have different selection/strategy.
        frame = frame.loc[frame.inning <= 9]
        keys = ["game_pk", "inning", "top", "outs", "bases"]
        if use_count:
            keys += ["balls", "strikes"]
        frame = frame.drop_duplicates(keys)
        max_runs = int(frame.remaining_runs.max())
        shape = (2, 3, 8, 4, 3, max_runs + 1)
        dist = np.empty(shape)
        base_frame = states.loc[(states.top | (states.inning < 9)) & (states.inning <= 9)]
        base_frame = base_frame.drop_duplicates(["game_pk", "inning", "top", "outs", "bases"])

        def histogram(rows):
            return np.bincount(rows.remaining_runs.to_numpy(dtype=int), minlength=max_runs + 1).astype(float)

        for outs in range(3):
            for bases in range(8):
                parent = base_frame.loc[(base_frame.outs == outs) & (base_frame.bases == bases)]
                pooled = histogram(parent)
                if pooled.sum() == 0:
                    raise ValueError(f"No training coverage for outs={outs}, bases={bases}")
                pooled /= pooled.sum()
                for side in range(2):
                    local = histogram(parent.loc[parent.top == (side == 0)])
                    base = (local + prior_strength * pooled) / (local.sum() + prior_strength)
                    subset = frame.loc[(frame.outs == outs) & (frame.bases == bases)
                                       & (frame.top == (side == 0))]
                    for balls in range(4):
                        for strikes in range(3):
                            counts = histogram(subset.loc[(subset.balls == balls) & (subset.strikes == strikes)])
                            dist[side, outs, bases, balls, strikes] = (
                                (counts + prior_strength * base) / (counts.sum() + prior_strength)
                                if use_count else base
                            )
        dist = _baseball_ordering(dist)
        return cls(dist, {
            "version": "remaining-runs-v1", "use_count": use_count,
            "prior_strength": prior_strength, "training_games": int(states.game_pk.nunique()),
            "training_min_date": str(states.game_date.min()),
            "training_max_date": str(states.game_date.max()),
            "source": "MLB Statcast states; MLB Stats API final scores and inning totals",
            "scope": "league-average; no player or market adjustments",
            "ordering": "stochastic dominance for balls, strikes, outs and advancing/adding runners",
        })

    @lru_cache(maxsize=64)
    def _future(self, side: int, innings: int, automatic_runner: bool = False):
        single = self.distributions[side, 0, 2 if automatic_runner else 0, 0, 0]
        result = np.array([1.])
        for _ in range(innings):
            result = np.convolve(result, single)
        return result

    @lru_cache(maxsize=2)
    def _extra_home_win(self, automatic_runner: bool):
        home = self._future(1, 1, automatic_runner)
        away = self._future(0, 1, automatic_runner)
        margins = np.convolve(home, away[::-1])
        tied = margins[len(away) - 1]
        return float(margins[len(away):].sum() / (1 - tied))

    @lru_cache(maxsize=8192)
    def _margin(self, inning: int, top: bool, outs: int, bases: int, balls: int, strikes: int,
                automatic_runner: bool):
        remaining = self.distributions[0 if top else 1, outs, bases, balls, strikes]
        future = max(9 - inning, 0)
        if top:
            home = self._future(1, future + 1, automatic_runner and inning >= 10)
            away = np.convolve(remaining, self._future(0, future))
        else:
            home = np.convolve(remaining, self._future(1, future))
            away = self._future(0, future)
        return np.convolve(home, away[::-1]), len(away) - 1

    def predict(self, state: GameState) -> float:
        if state.inning >= 9 and not state.top and state.home_score > state.away_score:
            return 1.
        margins, zero = self._margin(min(state.inning, 10), state.top, state.outs,
                                     state.bases, state.balls, state.strikes, state.automatic_runner)
        threshold = zero - (state.home_score - state.away_score)
        win = float(margins[max(threshold + 1, 0):].sum())
        tie = float(margins[threshold]) if 0 <= threshold < len(margins) else 0.
        # Finite empirical run support must not advertise certainty in an active game.
        return float(np.clip(win + tie * self._extra_home_win(state.automatic_runner), 1e-6, 1 - 1e-6))

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        columns = list(GameState.__dataclass_fields__)
        unique = frame[columns].drop_duplicates()
        unique = unique.assign(probability=[self.predict(GameState(**row)) for row in unique.to_dict("records")])
        return frame[columns].merge(unique, on=columns, how="left", validate="many_to_one").probability.to_numpy()

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"metadata": self.metadata, "distributions": self.distributions.tolist()}))

    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text())
        return cls(data["distributions"], data["metadata"])

    def describe(self, state: GameState) -> dict:
        home = self.predict(state)
        return {"state": asdict(state), "home_win_probability": home,
                "away_win_probability": 1 - home, "model": self.metadata}


def _baseball_ordering(distributions):
    """Pool conflicting CDFs until extra balls/runners cannot hurt the offense.

    The base-state edges represent adding or advancing a runner without an out.
    Projecting CDFs preserves an entire run distribution, rather than patching
    the reported probability for a particular inning or score.
    """
    cdf = distributions.cumsum(axis=-1)
    edges = ((0, 1), (1, 2), (2, 4), (1, 3), (2, 3), (3, 5), (4, 5), (5, 6), (6, 7))
    for _ in range(1000):
        before = cdf.copy()

        def pool(worse, better):
            adjustment = np.maximum(better - worse, 0) / 2
            worse += adjustment
            better -= adjustment

        for worse, better in edges:
            pool(cdf[:, :, worse], cdf[:, :, better])
        for balls in range(3):
            pool(cdf[:, :, :, balls], cdf[:, :, :, balls + 1])
        for strikes in range(2):
            pool(cdf[:, :, :, :, strikes + 1], cdf[:, :, :, :, strikes])
        for outs in range(2):
            pool(cdf[:, outs + 1], cdf[:, outs])
        if np.max(np.abs(cdf - before)) < 1e-12:
            break
    else:
        raise RuntimeError("Run-distribution ordering did not converge")
    result = np.maximum(np.diff(cdf, prepend=0, axis=-1), 0)
    return result / result.sum(axis=-1, keepdims=True)
