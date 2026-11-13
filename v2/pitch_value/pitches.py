"""Regular-season pitches in play order, each with the game state before and after it.

The state is (balls, strikes, outs, bases): 4 x 3 x 3 x 8 = 288 cells. The state after a pitch is the state
the next pitch of the same half-inning is thrown in, so whatever happens between two pitches (a steal, a
pickoff, a balk) lands on the earlier one and the values of a half-inning sum to its runs minus the
expectancy it started with. The last pitch of a half-inning leads to the terminal state, worth zero.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backend.data.game_types import regular_season_pitches

STATCAST_DIR = Path(__file__).resolve().parents[2] / "cache"
N_STATES = 288
TERMINAL = -1
HALF_INNING = ["game_pk", "inning", "inning_topbot"]
ORDER = ["game_pk", "at_bat_number", "pitch_number"]
COLUMNS = [
    "game_type", "game_date", "game_pk", "at_bat_number", "pitch_number", "inning", "inning_topbot", "home_team", "away_team",
    "pitcher", "batter", "player_name", "stand", "p_throws", "pitch_type", "events", "description",
    "balls", "strikes", "outs_when_up", "on_1b", "on_2b", "on_3b", "bat_score", "post_bat_score",
    "launch_speed", "launch_angle", "estimated_woba_using_speedangle", "delta_run_exp",
]


def state_index(balls, strikes, outs, bases) -> np.ndarray:
    return ((np.asarray(balls) * 3 + np.asarray(strikes)) * 3 + np.asarray(outs)) * 8 + np.asarray(bases)


def state_parts(state) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(balls, strikes, outs, bases) of each state index; bases is a bit mask, 1 first, 2 second, 4 third."""
    state = np.asarray(state)
    return state // 72, state // 24 % 3, state // 8 % 3, state % 8


def load_season(season: int, extra: tuple[str, ...] = ()) -> pd.DataFrame:
    """One season's pitches; ``extra`` names further Statcast columns to carry through unchanged."""
    path = STATCAST_DIR / f"statcast_{season}.parquet"
    frame = regular_season_pitches(pd.read_parquet(path, columns=[*COLUMNS, *extra]), season)
    frame = frame.drop_duplicates(ORDER).sort_values(ORDER, kind="stable").reset_index(drop=True)
    valid = (
        frame["balls"].between(0, 3) & frame["strikes"].between(0, 2) & frame["outs_when_up"].between(0, 2)
        & frame["bat_score"].notna() & frame["post_bat_score"].notna()
    ).fillna(False)
    if (~valid).mean() > 0.001:
        raise ValueError(f"statcast_{season}: {(~valid).sum()} pitches carry an impossible count, outs or score")
    frame = frame[valid].reset_index(drop=True)

    bases = frame["on_1b"].notna() * 1 + frame["on_2b"].notna() * 2 + frame["on_3b"].notna() * 4
    out = pd.DataFrame({
        "season": season,
        "game_pk": frame["game_pk"].astype("int64"),
        "game_date": frame["game_date"],
        "inning": frame["inning"].astype("int16"),
        "top": frame["inning_topbot"].eq("Top"),
        "home_team": frame["home_team"],
        "away_team": frame["away_team"],
        "team": frame["home_team"].where(frame["inning_topbot"].eq("Top"), frame["away_team"]),
        "pitcher": frame["pitcher"].astype("int64"),
        "batter": frame["batter"].astype("int64"),
        "pitcher_name": frame["player_name"],
        "stand": frame["stand"],
        "p_throws": frame["p_throws"],
        "pitch_type": frame["pitch_type"],
        "events": frame["events"],
        "description": frame["description"],
        "launch_speed": frame["launch_speed"].astype("float64"),
        "launch_angle": frame["launch_angle"].astype("float64"),
        "xwoba": frame["estimated_woba_using_speedangle"].astype("float64"),
        "savant_delta": frame["delta_run_exp"].astype("float64"),
        "state": state_index(frame["balls"].astype(int), frame["strikes"].astype(int),
                             frame["outs_when_up"].astype(int), bases.astype(int)),
    })
    score = frame["bat_score"].astype("int64")
    half = [frame[c] for c in HALF_INNING]
    final = frame["post_bat_score"].astype("int64").groupby(half).transform("max")
    next_score = score.groupby(half).shift(-1)
    next_state = out["state"].groupby(half).shift(-1)
    out["next_state"] = next_state.fillna(TERMINAL).astype("int64")
    out["runs"] = (next_score.fillna(final) - score).astype("int64")
    out["runs_to_end"] = (final - score).astype("int64")
    if (out["runs"] < 0).any() or (out["runs_to_end"] < 0).any():
        raise ValueError(f"statcast_{season}: the batting team's score falls within a half-inning")
    for column in extra:
        out[column] = frame[column].to_numpy()
    return out


def load(seasons: list[int], extra: tuple[str, ...] = ()) -> pd.DataFrame:
    return pd.concat([load_season(s, extra) for s in seasons], ignore_index=True)
