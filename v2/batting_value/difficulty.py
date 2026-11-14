"""Pitch difficulty: what a pitch is worth to the league before the hitter acts.

From where the pitch is, what it is, the count, the handedness matchup and the season (the called zone and
the league's contact rates drift from year to year), league models give

- the chance the league swings at it;
- taken: the chance of a ball, a called strike or a hit batter;
- swung at: the chance of a whiff, a foul or a ball in play, and the expected value of that ball in play.

Each result is worth its context-neutral value in that count, so the value of taking, of swinging and of the
league's mix of the two follow, and bases and outs never enter.

Every prediction for a hitter comes from models fitted without any of his pitches (five folds of hitters), so
a hitter is never part of his own league average and the calibration tables are out of sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from v2.pitch_value.batted import N_BASE_OUT

N_FOLDS = 5
N_COUNTS = 12
TAKE_CLASSES = ("ball", "called_strike", "hit_by_pitch")
SWING_CLASSES = ("whiff", "foul", "in_play")
FEATURES = ("inside", "height", "plate_z", "release_speed", "break_in", "pfx_z", "pitch_type", "balls", "strikes",
            "bats_left", "same_side", "season")
CATEGORICAL = [FEATURES.index("pitch_type")]
# Leaves this large keep a location surface from fitting single at-bats; boosting stops on a held-out tenth.
# The penalty keeps the boosting steps finite where a leaf is all one result (a pitch down the middle is a
# strike, a pitch off the plate never hits the batter): without it the take models diverge within a few trees.
MODEL = {"max_iter": 1000, "min_samples_leaf": 200, "l2_regularization": 10.0, "categorical_features": CATEGORICAL,
         "random_state": 0}


def features(pitches: pd.DataFrame) -> np.ndarray:
    """One row per pitch. Horizontal location and break are mirrored for left-handed hitters, so positive is
    toward the hitter; height is also given on the hitter's own strike zone (0 bottom, 1 top)."""
    left = pitches["stand"].eq("L").to_numpy()
    mirror = np.where(left, 1.0, -1.0)
    types = {name: i for i, name in enumerate(sorted(pitches["pitch_type"].dropna().unique()))}
    count = pitches["state"].to_numpy() // N_BASE_OUT
    return np.column_stack([
        pitches["plate_x"].to_numpy(float) * mirror,
        ((pitches["plate_z"] - pitches["sz_bot"]) / (pitches["sz_top"] - pitches["sz_bot"])).to_numpy(float),
        pitches["plate_z"].to_numpy(float),
        pitches["release_speed"].to_numpy(float),
        pitches["pfx_x"].to_numpy(float) * mirror,
        pitches["pfx_z"].to_numpy(float),
        pitches["pitch_type"].map(types).to_numpy(float),
        count // 3, count % 3,
        left.astype(float),
        pitches["stand"].eq(pitches["p_throws"]).to_numpy(float),
        pitches["season"].to_numpy(float),
    ])


def folds(batter: pd.Series) -> np.ndarray:
    """Fold of each pitch; all of a hitter's pitches, in every season, share one."""
    ids = np.sort(batter.unique())
    order = np.random.default_rng(0).permutation(len(ids))
    return pd.Series(order % N_FOLDS, index=ids).reindex(batter).to_numpy()


def cross_fit(make_model, x: np.ndarray, y: np.ndarray, train: np.ndarray, fold: np.ndarray, width: int) -> np.ndarray:
    """Predictions for every row from a model fitted on the ``train`` rows of the other folds."""
    out = np.empty((len(x), width))
    for k in range(N_FOLDS):
        held = fold == k
        model = make_model().fit(x[train & ~held], y[train & ~held])
        out[held] = model.predict_proba(x[held]) if width > 1 else model.predict(x[held])[:, None]
    return out


def _class_index(result: pd.Series, classes: tuple[str, ...]) -> np.ndarray:
    return result.map({name: i for i, name in enumerate(classes)}).fillna(-1).to_numpy(dtype=int)


def fit(pitches: pd.DataFrame, neutral: pd.Series) -> pd.DataFrame:
    """League chances and values for every pitch a hitter saw (other rows are left missing).

    Columns: p_swing; p_ball, p_called_strike, p_hit_by_pitch (if taken); p_whiff, p_foul, p_in_play (if swung
    at); in_play (expected value of the ball in play); take, swing, league (runs).
    """
    seen = pitches["seen"].to_numpy()
    swing = pitches["swing"].to_numpy()
    in_play = pitches["result"].eq("in_play").to_numpy()
    x, fold = features(pitches), folds(pitches["batter"])
    classifier = lambda: HistGradientBoostingClassifier(**MODEL)
    regressor = lambda: HistGradientBoostingRegressor(**MODEL)

    out = pd.DataFrame(index=pitches.index)
    out["p_swing"] = cross_fit(classifier, x, swing.astype(int), seen, fold, 2)[:, 1]
    # A take is a hit batter or not, then a called strike or a ball.
    hit = pitches["result"].eq("hit_by_pitch").to_numpy()
    called = pitches["result"].eq("called_strike").to_numpy()
    out["p_hit_by_pitch"] = cross_fit(classifier, x, hit.astype(int), seen & ~swing, fold, 2)[:, 1]
    strike = cross_fit(classifier, x, called.astype(int), seen & ~swing & ~hit, fold, 2)[:, 1]
    out["p_called_strike"] = (1.0 - out["p_hit_by_pitch"]) * strike
    out["p_ball"] = (1.0 - out["p_hit_by_pitch"]) * (1.0 - strike)
    swung = cross_fit(classifier, x, _class_index(pitches["result"], SWING_CLASSES), seen & swing, fold, len(SWING_CLASSES))
    for i, name in enumerate(SWING_CLASSES):
        out[f"p_{name}"] = swung[:, i]
    out["in_play"] = cross_fit(regressor, x, neutral.to_numpy(), seen & in_play, fold, 1)[:, 0]

    # What each result other than a ball in play is worth in each count: the league mean of its neutral value.
    count = pitches["state"].to_numpy() // N_BASE_OUT
    worth = (pd.DataFrame({"count": count, "result": pitches["result"], "value": neutral})[seen]
             .groupby(["count", "result"])["value"].mean().unstack())
    if worth[[*TAKE_CLASSES, "whiff", "foul"]].reindex(range(N_COUNTS)).isna().any().any():
        raise ValueError("a count has no pitches for one of the take or swing results")
    by_count = lambda name: worth[name].to_numpy()[count]
    out["take"] = sum(out[f"p_{name}"] * by_count(name) for name in TAKE_CLASSES)
    out["swing"] = out["p_whiff"] * by_count("whiff") + out["p_foul"] * by_count("foul") + out["p_in_play"] * out["in_play"]
    out["league"] = out["p_swing"] * out["swing"] + (1.0 - out["p_swing"]) * out["take"]
    out[~seen] = np.nan
    return out


def calibration(pitches: pd.DataFrame, league: pd.DataFrame, neutral: pd.Series, by: str) -> pd.DataFrame:
    """Predicted against actual rates, and ball-in-play values, within each group of ``by`` (a column of
    ``pitches``, or 'count')."""
    seen, swing, result = pitches["seen"], pitches["swing"], pitches["result"]
    count = pitches["state"] // N_BASE_OUT
    key = (count // 3).astype(str) + "-" + (count % 3).astype(str) if by == "count" else pitches[by]
    hit = seen & result.eq("in_play")
    parts = {
        "swing": (seen, swing, league["p_swing"]),
        "called_strike": (seen & ~swing, result.eq("called_strike"), league["p_called_strike"]),
        "whiff": (seen & swing, result.eq("whiff"), league["p_whiff"]),
        "foul": (seen & swing, result.eq("foul"), league["p_foul"]),
        "in_play": (seen & swing, result.eq("in_play"), league["p_in_play"]),
        "in_play_value": (hit, neutral, league["in_play"]),
    }
    columns = {}
    for name, (where, actual, predicted) in parts.items():
        group = key[where]
        columns[f"{name}_n"] = group.groupby(group).size()
        columns[f"{name}_actual"] = actual[where].astype(float).groupby(group).mean()
        columns[f"{name}_predicted"] = predicted[where].groupby(group).mean()
    return pd.DataFrame(columns).rename_axis(by).reset_index()


def worst_rate_gap(table: pd.DataFrame) -> float:
    """The largest gap between a predicted and an actual rate in a calibration table."""
    rates = [c[: -len("_actual")] for c in table.columns if c.endswith("_actual") and c != "in_play_value_actual"]
    return float(max((table[f"{r}_actual"] - table[f"{r}_predicted"]).abs().max() for r in rates))
