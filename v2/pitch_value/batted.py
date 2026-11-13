"""Balls in play valued by how they were hit, so fielding, park and bounce drop out.

A contact model gives each batted ball the league's chance of an out, single, double, triple or home run from
its exit velocity and launch angle in that season (the same contact carried differently from year to year:
league-wide, 2021's batted balls returned 0.007 runs more than their speed and angle predict over six seasons,
2025's 0.007 less) and with those bases and outs (a held runner, double-play depth and a drawn-in infield
open holes: without them, balls hit with runners on come out 0.008 runs under what they returned). Each outcome is worth what it was worth on average from the same
bases and outs (an out on the ground is kept apart from one in the air: one turns double plays, the other
scores a runner from third). The pitch is charged the probability-weighted value instead of what happened.

A ball with no tracked exit velocity or launch angle (about 0.4%) keeps its actual value.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from v2.pitch_value import expectancy

CLASSES = ("out", "single", "double", "triple", "home_run")
# Launch angle in degrees that separates ground balls, line drives and fly balls.
TRAJECTORY_EDGES = (10.0, 25.0)
N_TRAJECTORIES = len(TRAJECTORY_EDGES) + 1
N_BASE_OUT = 24
# Leaves this large keep a five-feature model on 700,000 batted balls from fitting single plays.
MIN_LEAF = 200


@dataclass(frozen=True)
class Contact:
    model: HistGradientBoostingClassifier
    seasons: tuple[int, ...]
    # Mean runs scored and mean next-state expectancy, by (outcome row, bases and outs).
    runs: np.ndarray
    after: np.ndarray


def tracked(pitches: pd.DataFrame) -> pd.Series:
    return (pitches["description"].eq("hit_into_play") & pitches["launch_speed"].notna()
            & pitches["launch_angle"].notna())


def _features(balls: pd.DataFrame, state: np.ndarray, seasons: tuple[int, ...]) -> np.ndarray:
    season = balls["season"].map({s: i for i, s in enumerate(seasons)})
    if season.isna().any():
        raise ValueError(f"contact model was not fitted on season(s) {sorted(balls.loc[season.isna(), 'season'].unique())}")
    return np.column_stack([balls["launch_speed"].to_numpy(), balls["launch_angle"].to_numpy(), season.to_numpy(),
                            state % 8, state // 8 % 3])


def _outcome(events: pd.Series) -> np.ndarray:
    """Index into CLASSES; everything that is not a hit is an out, errors and fielder's choices included."""
    return events.map({name: i for i, name in enumerate(CLASSES)}).fillna(0).to_numpy(dtype=int)


def _trajectory(launch_angle: pd.Series) -> np.ndarray:
    return np.digitize(launch_angle.to_numpy(), TRAJECTORY_EDGES)


def _row(outcome: np.ndarray, trajectory: np.ndarray) -> np.ndarray:
    """Outs take rows 0..2 by trajectory; hits take one row each after them."""
    return np.where(outcome == 0, trajectory, N_TRAJECTORIES - 1 + outcome)


def fit(pitches: pd.DataFrame, table: pd.DataFrame) -> Contact:
    balls = pitches[tracked(pitches)]
    outcome = _outcome(balls["events"])
    seasons = tuple(sorted(int(s) for s in balls["season"].unique()))
    model = HistGradientBoostingClassifier(min_samples_leaf=MIN_LEAF, categorical_features=[2, 3, 4], random_state=0)
    model.fit(_features(balls, balls["state"].to_numpy(), seasons), outcome)

    lookup = expectancy.lookup(table)
    n_rows = N_TRAJECTORIES + len(CLASSES) - 1
    cell = _row(outcome, _trajectory(balls["launch_angle"])) * N_BASE_OUT + balls["state"].to_numpy() % N_BASE_OUT
    n = np.bincount(cell, minlength=n_rows * N_BASE_OUT)
    if (n == 0).any():
        raise ValueError(f"{(n == 0).sum()} outcome-by-base-out cells have no batted ball")
    shape = (n_rows, N_BASE_OUT)
    runs = (np.bincount(cell, weights=balls["runs"].to_numpy(), minlength=n.size) / n).reshape(shape)
    after = (np.bincount(cell, weights=lookup[balls["next_state"].to_numpy()], minlength=n.size) / n).reshape(shape)
    return Contact(model=model, seasons=seasons, runs=runs, after=after)


def expected_after(balls: pd.DataFrame, state: np.ndarray, contact: Contact) -> np.ndarray:
    """Expected runs scored plus next-state expectancy for tracked balls in play, had each been hit in ``state``."""
    probs = contact.model.predict_proba(_features(balls, state, contact.seasons))
    base_out = state % N_BASE_OUT
    trajectory = _trajectory(balls["launch_angle"])
    worth = contact.runs + contact.after
    expected = np.zeros(len(balls))
    for outcome in range(len(CLASSES)):
        row = _row(np.full(len(balls), outcome), trajectory)
        expected += probs[:, outcome] * worth[row, base_out]
    return expected


def expected_values(pitches: pd.DataFrame, table: pd.DataFrame, value: pd.Series, contact: Contact) -> pd.Series:
    """``value`` with every tracked ball in play replaced by its expected value."""
    hit = tracked(pitches).to_numpy()
    state = pitches["state"].to_numpy()[hit]
    out = value.to_numpy(copy=True)
    out[hit] = expected_after(pitches[hit], state, contact) - expectancy.lookup(table)[state]
    return pd.Series(out, index=value.index, name="value")
