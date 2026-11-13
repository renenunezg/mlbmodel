"""Run expectancy over the full game state: count x bases x outs, 288 cells.

The table is the solution of the pitch-to-pitch chain: the expectancy of a state is the mean runs scored on
a pitch thrown in it plus the mean expectancy of the state that pitch leads to. That is the same quantity as
the mean runs scored from the state to the end of the half-inning, measured with far less noise in the rare
cells (3-0 with a runner on third and nobody out is thrown about fifteen times a season), and it makes the
pitch values of a half-inning sum exactly to its runs minus the expectancy it started with. The raw mean and
the sample are kept beside it.

Only innings one through eight are read: they are always played to three outs, and extra innings start with
a runner on second.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from v2.pitch_value.pitches import N_STATES, TERMINAL, state_parts

LAST_FULL_INNING = 8


def fit(pitches: pd.DataFrame) -> pd.DataFrame:
    sample = pitches[pitches["inning"] <= LAST_FULL_INNING]
    state = sample["state"].to_numpy()
    after = sample["next_state"].to_numpy()
    n = np.bincount(state, minlength=N_STATES)
    if (n == 0).any():
        raise ValueError(f"{(n == 0).sum()} of the {N_STATES} states never occur in the sample")
    live = after != TERMINAL
    transitions = np.zeros((N_STATES, N_STATES))
    np.add.at(transitions, (state[live], after[live]), 1.0)
    transitions /= n[:, None]
    runs = np.bincount(state, weights=sample["runs"].to_numpy(), minlength=N_STATES) / n
    expectancy = np.linalg.solve(np.eye(N_STATES) - transitions, runs)

    balls, strikes, outs, bases = state_parts(np.arange(N_STATES))
    return pd.DataFrame({
        "state": np.arange(N_STATES), "balls": balls, "strikes": strikes, "outs": outs, "bases": bases,
        "runs": expectancy,
        "raw_runs": np.bincount(state, weights=sample["runs_to_end"].to_numpy(), minlength=N_STATES) / n,
        "pitches": n,
    })


def lookup(table: pd.DataFrame) -> np.ndarray:
    """Expectancy by state index, with the terminal state (index -1) worth zero at the end."""
    return np.append(table.sort_values("state")["runs"].to_numpy(), 0.0)

