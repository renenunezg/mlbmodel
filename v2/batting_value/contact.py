"""The hitter's contact model: do spray angle and sprint speed belong beside exit velocity and launch angle?

The pitcher's contact model reads exit velocity and launch angle only, which is right for a pitcher. A hitter
also controls where he hits the ball (a pulled fly ball carries the short part of the park) and how fast he
runs (infield hits, extra bases). Each addition is kept only if a season's expected value on contact then says
more about the hitter's next-season production on contact, on the development pairs.

Every variant, the plain one included, is fitted without the hitter it values (five folds of hitters): sprint
speed is one number per hitter-season, and a model that saw his own balls in play could use it to recognise
him and hand him back his own luck.

Usage:
    python -m v2.batting_value.contact

Writes cache/batting_value/contact_test.json.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from backend.log import setup_logging
from v2.batting_value import pitches as batting_pitches
from v2.batting_value import values
from v2.batting_value.difficulty import N_FOLDS, fold_map, fold_of, folds
from v2.players import savant
from v2.pitch_value import batted, expectancy
from v2.pitch_value.batted import CLASSES, MIN_LEAF, N_BASE_OUT, _outcome, _row, _trajectory
from v2.pitch_value.pitches import STATCAST_DIR

log = logging.getLogger(__name__)

OUT_DIR = STATCAST_DIR / "batting_value"
SEASONS = (2021, 2022, 2023, 2024, 2025, 2026)
# Home plate in Statcast's hit-coordinate picture.
HOME_PLATE = (125.42, 198.27)
VARIANTS = {"plain": (), "spray": ("spray",), "sprint": ("sprint",), "spray_sprint": ("spray", "sprint")}
# The additions the test kept; the pipeline's ball-in-play values use the hitter model only when this is not empty.
# 2026-11-13, 596 development pairs with 200+ tracked balls in play: sprint speed +.021 [+.009, +.036] over the
# plain model, spray angle -.033 [-.051, -.015] (where the ball went is mostly the hit or the out itself).
KEPT: tuple[str, ...] = ("sprint",)
# First seasons of the pairs the choice is made on, as in validate.py.
DEVELOPMENT = (2021, 2022, 2023)
# Tracked balls in play required in both seasons of a pair; the middle one decides.
BALLS_CUTS = (100, 200, 300)
DECIDING_CUT = 200
BOOTSTRAP_DRAWS = 2000
# A contact model that stops this early has diverged, not converged.
MIN_TREES = 30


def spray(pitches: pd.DataFrame) -> np.ndarray:
    """Degrees from straightaway centre, positive toward the hitter's pull side."""
    angle = np.degrees(np.arctan2(pitches["hc_x"].to_numpy(float) - HOME_PLATE[0], HOME_PLATE[1] - pitches["hc_y"].to_numpy(float)))
    return np.where(pitches["stand"].eq("L"), angle, -angle)


def sprint(pitches: pd.DataFrame) -> np.ndarray:
    """The hitter's sprint speed that season in feet per second; missing under ten competitive runs."""
    boards = pd.concat([savant.load_board("sprint_speed", int(s)) for s in pitches["season"].unique()])
    speed = boards.set_index(["season", "player_id"])["sprint_speed"].astype(float)
    return speed.reindex(pd.MultiIndex.from_frame(pitches[["season", "batter"]])).to_numpy()


EXTRAS = {"spray": spray, "sprint": sprint}


@dataclass(frozen=True)
class HitterContact:
    """The hitter contact models, one per fold of hitters."""
    extras: tuple[str, ...]
    seasons: tuple[int, ...]   # seasons fitted on; a later season is read as the last of them
    folds: pd.Series           # hitter -> fold, for every hitter with a tracked ball in play in the fitting frame
    models: list


def _features(balls: pd.DataFrame, at: np.ndarray, seasons: tuple[int, ...], more: list[np.ndarray]) -> np.ndarray:
    season = np.minimum(np.searchsorted(seasons, balls["season"].to_numpy()), len(seasons) - 1)
    return np.column_stack([balls["launch_speed"].to_numpy(), balls["launch_angle"].to_numpy(), season,
                            at % 8, at // 8 % 3, *more])


def fit(pitches: pd.DataFrame, extras: tuple[str, ...], through: int) -> HitterContact:
    """Contact models that also read ``extras``, fitted on the tracked balls in play up to season ``through``,
    each without one fold of hitters."""
    balls = pitches[batted.tracked(pitches)]
    sample = balls["season"].le(through).to_numpy()
    seasons = tuple(sorted(int(s) for s in balls.loc[sample, "season"].unique()))
    x = _features(balls, balls["state"].to_numpy(), seasons, [EXTRAS[name](balls) for name in extras])
    outcome, fold = _outcome(balls["events"]), folds(balls["batter"])
    models = []
    for k in range(N_FOLDS):
        train = sample & (fold != k)
        model = HistGradientBoostingClassifier(min_samples_leaf=MIN_LEAF, categorical_features=[2, 3, 4], random_state=0)
        model.fit(x[train], outcome[train])
        if model.n_iter_ < MIN_TREES:
            raise ValueError(f"contact model with {extras or 'no additions'} stopped after {model.n_iter_} rounds of boosting")
        models.append(model)
    return HitterContact(extras=extras, seasons=seasons, folds=fold_map(balls["batter"]), models=models)


def neutral_in_play(pitches: pd.DataFrame, table: pd.DataFrame, contact: batted.Contact, hitter: HitterContact) -> pd.Series:
    """Context-neutral expected value of every tracked ball in play, from a contact model that also reads the
    hitter's extras and never saw him. Built as ``v2.pitch_value.neutral`` builds it: the expected value in
    each of the 24 base-out states, weighted by how often the league's balls in play come in each."""
    balls = pitches[batted.tracked(pitches)]
    state = balls["state"].to_numpy()
    more = [EXTRAS[name](balls) for name in hitter.extras]
    trajectory, fold = _trajectory(balls["launch_angle"]), fold_of(balls["batter"], hitter.folds)
    share = np.bincount(state % N_BASE_OUT, minlength=N_BASE_OUT) / len(balls)
    worth, lookup = contact.runs + contact.after, expectancy.lookup(table)
    count = state // N_BASE_OUT
    value = np.zeros(len(balls))
    for k, model in enumerate(hitter.models):
        held = fold == k
        for base_out in np.flatnonzero(share):
            swapped = count[held] * N_BASE_OUT + base_out
            probs = model.predict_proba(_features(balls[held], swapped, hitter.seasons, [m[held] for m in more]))
            expected = sum(probs[:, o] * worth[_row(np.full(held.sum(), o), trajectory[held]), base_out] for o in range(len(CLASSES)))
            value[held] += share[base_out] * (expected - lookup[swapped])
    return pd.Series(value, index=balls.index, name="value")


def hitter_seasons(pitches: pd.DataFrame, in_play: dict[str, pd.Series]) -> pd.DataFrame:
    """Per hitter-season: tracked balls in play, wOBA on contact, and each variant's expected value per ball."""
    balls = pitches.loc[batted.tracked(pitches), ["season", "batter", "woba_value", "woba_denom"]].join(pd.DataFrame(in_play))
    out = balls.groupby(["season", "batter"]).agg(balls=("season", "size"), woba_value=("woba_value", "sum"),
                                                  woba_denom=("woba_denom", "sum"), **{name: (name, "mean") for name in in_play})
    out["wobacon"] = out.pop("woba_value") / out.pop("woba_denom")
    return out.reset_index()


def pairs(seasons: pd.DataFrame) -> pd.DataFrame:
    nxt = seasons.assign(season=seasons["season"] - 1)
    return seasons.merge(nxt, on=["season", "batter"], suffixes=("", "_next"))


def carryover(frame: pd.DataFrame, names: list[str], rng: np.random.Generator) -> dict:
    """Each variant's correlation with next-season wOBA on contact and with itself a season later, and the
    gain of each over the plain model, with intervals that resample hitters."""
    hitters = frame["batter"].unique()
    rows = {b: np.flatnonzero(frame["batter"].to_numpy() == b) for b in hitters}
    r = lambda f, a, b: float(np.corrcoef(f[a], f[b])[0, 1])
    point = {n: {"next_wobacon": r(frame, n, "wobacon_next"), "repeat": r(frame, n, f"{n}_next")} for n in names}
    draws = {n: [] for n in names}
    for _ in range(BOOTSTRAP_DRAWS):
        sample = frame.iloc[np.concatenate([rows[b] for b in rng.choice(hitters, len(hitters))])]
        for n in names:
            draws[n].append(r(sample, n, "wobacon_next"))
    draws = {n: np.array(d) for n, d in draws.items()}
    interval = lambda d: [float(v) for v in np.percentile(d, [2.5, 97.5])]
    return {"pairs": len(frame), "hitters": len(hitters), "r": point,
            "gain_over_plain": {n: {"r": point[n]["next_wobacon"] - point["plain"]["next_wobacon"],
                                    "interval": interval(draws[n] - draws["plain"])} for n in names if n != "plain"}}


def main() -> None:
    setup_logging()
    frame = batting_pitches.load(list(SEASONS))
    built = values.build(frame)
    through = max(SEASONS)
    in_play = {name: neutral_in_play(frame, built.table, built.contact, fit(frame, extras, through))
               for name, extras in VARIANTS.items()}
    for name, value in in_play.items():
        log.info(f"{name}: {value.mean():+.4f} runs per tracked ball in play")
    table = pairs(hitter_seasons(frame, in_play))
    table = table[table["season"].isin(DEVELOPMENT)]
    rng = np.random.default_rng(0)
    report = {}
    for cut in BALLS_CUTS:
        block = report[str(cut)] = carryover(table[(table["balls"] >= cut) & (table["balls_next"] >= cut)], list(VARIANTS), rng)
        log.info(f"{cut}+ tracked balls in play both seasons: {block['pairs']} pairs, {block['hitters']} hitters")
        for name in VARIANTS:
            log.info(f"  {name:>13}  next wOBAcon r {block['r'][name]['next_wobacon']:+.3f}  repeat r {block['r'][name]['repeat']:+.3f}")
        for name, gain in block["gain_over_plain"].items():
            log.info(f"  {name} over plain: {gain['r']:+.3f}  [{gain['interval'][0]:+.3f}, {gain['interval'][1]:+.3f}]")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "contact_test.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
