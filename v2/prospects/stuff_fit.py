"""Fit the stuff and command models on MLB pitches, then the level offsets and the carry-over check.

Stage ``models``: per-family stuff models and the command pair on MLB 2021-2025, every pitcher with
``pitcher % 5 == 0`` held out (early stopping on a pitcher-grouped tenth of the rest), the grade
scales, the reliability floors, and once the held-out checks are recorded, the 2026 MLB season as
the models' own test year. Stage ``levels``: Triple-A and Florida State League offsets from pitchers
measured at two levels in a season, and the Triple-A to MLB carry-over check that decides which
pitch families are released for publishing.

Usage:
    python -m v2.prospects.stuff_fit --stage models
    python -m v2.prospects.stuff_fit --stage levels
"""
from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import xgboost as xgb

from backend.log import setup_logging
from v2.prospects import pitches, stuff
from v2.prospects.model import MODELS_DIR

log = logging.getLogger(__name__)

STUFF_PATH = MODELS_DIR / "stuff_model.json"
FIT_SEASONS = [2021, 2022, 2023, 2024, 2025]
TEST_SEASON = 2026
HELD_OUT = 5  # pitcher % HELD_OUT == 0 is held out of every fit
PARAMS = {"objective": "reg:squarederror", "max_depth": 4, "eta": 0.2, "min_child_weight": 200, "subsample": 0.8,
          "colsample_bytree": 0.9, "tree_method": "hist", "verbosity": 0}
MAX_ROUNDS = 2000
EARLY_STOP = 50
FLOOR_GRID = (25, 50, 75, 100, 150, 200, 300)
COMMAND_FLOOR_GRID = (100, 200, 300, 500)
SPLIT_HALF_R = 0.7
BIP_COLS = ["ls", "la", "balls", "strikes"]
RUN_TEST_SEASON = True  # 2026 was scored for the raw-target and depth-6 versions (2026-10-23); later choices (expected target, depth 4, no arm angle) were made on held-out pitchers
COMMAND_ETA = 0.5  # fewer, coarser rounds keep the command model inside the stored-model budget
# Depth 4 (from 6) and the count table (from a count booster) cut the stored row from 0.74 MB; chosen on held-out pitchers.
# Fixed 2026-10-23 before the levels stage first ran.
CARRY_MIN_R = 0.5
CARRY_MAX_BIAS = 5.0  # grade points
CARRY_LATER_SEASON = False  # the first MLB season at or after the Triple-A season, same season included
FSL_MIN_PAIRS = 30
# Added 2026-10-23 (second pass): spring training as a level. A Triple-A-specific command model was tried the same
# day and dropped: its carry-over r was .473 against .474 for the MLB model, for 0.16 MB more stored model.
SPRING_MIN_PAIRS = 30
BOOTSTRAP = 1000


def _train(frame: pd.DataFrame, cols: list[str], target: str = "rv_x", eta: float | None = None) -> xgb.Booster:
    val = (frame["pitcher"] // HELD_OUT) % 10 == 0
    d, v = frame[~val], frame[val]
    params = PARAMS if eta is None else {**PARAMS, "eta": eta}
    return xgb.train(params, stuff._matrix(d, cols, d[target]), MAX_ROUNDS, evals=[(stuff._matrix(v, cols, v[target]), "val")],
                     early_stopping_rounds=EARLY_STOP, verbose_eval=False)


def expected_target(mlb: pd.DataFrame) -> pd.Series:
    """Run value with each tracked ball in play replaced by its expected value from exit velocity, launch angle and count.

    Which balls in play fall for hits is mostly noise to a pitch's shape; the expectation keeps the
    contact quality the shape allowed and drops the defense and luck.
    """
    fit = mlb[(mlb["pitcher"] % HELD_OUT != 0) & mlb["bip"] & mlb["ls"].notna() & mlb["la"].notna() & mlb["rv"].notna()]
    b = _trim(_train(fit, BIP_COLS, target="rv"))
    out = mlb["rv"].copy()
    m = mlb["bip"] & mlb["ls"].notna() & mlb["la"].notna()
    out[m] = b.predict(stuff._matrix(mlb[m], BIP_COLS))
    return out


def _trim(booster: xgb.Booster) -> xgb.Booster:
    return booster[: booster.best_iteration + 1]


def _rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def fit_models(mlb: pd.DataFrame) -> stuff.StuffModel:
    model = stuff.StuffModel(fitted_at=datetime.now(UTC).isoformat(), seasons=FIT_SEASONS)
    fit = mlb[(mlb["pitcher"] % HELD_OUT != 0) & mlb["rv_x"].notna()]
    for fam in pitches.FAMILIES:
        b = _trim(_train(fit[fit["family"].eq(fam)], pitches.SHAPE))
        model.set_booster(fam, b)
        log.info(f"stuff[{fam}]: {b.num_boosted_rounds()} rounds")
    located = fit[fit["px"].notna() & fit["pz"].notna()]
    b = _trim(_train(located, pitches.LOCATION, eta=COMMAND_ETA))
    model.set_booster("command", b)
    log.info(f"stuff[command]: {b.num_boosted_rounds()} rounds")
    model.count_table = {k: float(v) for k, v in located.groupby(stuff.count_key(located))["rv_x"].mean().items()}
    return model


def grade_scales(model: stuff.StuffModel, values: pd.DataFrame) -> None:
    t = stuff.by_type(values, ["pitcher", "season"])
    t = t[t["n"] >= stuff.GRADE_MIN_PITCHES]
    model.scale = {k: [float(g["stuff_rv100"].mean()), float(g["stuff_rv100"].std())] for k, g in t.groupby("pitch_type") if len(g) >= 20}
    c = stuff.by_pitcher_command(values, ["pitcher", "season"])
    c = c[c["command_n"] >= stuff.COMMAND_MIN_PITCHES]
    model.command_scale = [float(c["command_rv100"].mean()), float(c["command_rv100"].std())]


def _split_half(values: pd.DataFrame, keys: list[str], column: str, n: int) -> float:
    """Correlation across pitchers of the two halves' mean, each pitcher contributing his first n pitches alternately split."""
    v = values[values[column].notna()]
    v = v[v.groupby(keys)[column].transform("size") >= n]
    if v.empty:
        return float("nan")
    v = v.assign(order=v.groupby(keys).cumcount())
    v = v[v["order"] < n]
    halves = v.groupby([*keys, v["order"] % 2])[column].mean().unstack()
    return float(halves[0].corr(halves[1])) if len(halves) >= 30 else float("nan")


def reliability_floors(model: stuff.StuffModel, values: pd.DataFrame) -> dict:
    report = {}
    for t in model.scale:
        sub = values[values["pitch_type"].eq(t)]
        rs = {n: _split_half(sub, ["pitcher", "season"], "stuff_rv", n) for n in FLOOR_GRID}
        model.floor[t] = next((n for n in FLOOR_GRID if rs[n] >= SPLIT_HALF_R), FLOOR_GRID[-1])
        report[t] = rs
    rs = {n: _split_half(values, ["pitcher", "season"], "command_rv", n) for n in COMMAND_FLOOR_GRID}
    model.floor["command"] = next((n for n in COMMAND_FLOOR_GRID if rs[n] >= SPLIT_HALF_R), COMMAND_FLOOR_GRID[-1])
    report["command"] = rs
    return report


def check(values: pd.DataFrame, fit_means: dict[str, float]) -> dict:
    """Per family: pitch RMSE against the pitch-type mean, and pitcher-type predicted against realized rv100."""
    out = {}
    for fam in pitches.FAMILIES:
        v = values[values["family"].eq(fam) & values["rv"].notna()]
        base = v["pitch_type"].map(fit_means)
        t = v.groupby(["pitcher", "season", "pitch_type"]).agg(n=("rv", "size"), pred=("stuff_rv", "mean"), actual=("rv", "mean"))
        t = t[t["n"] >= stuff.GRADE_MIN_PITCHES]
        out[fam] = {"pitches": int(len(v)), "rmse_model": _rmse(v["rv"], v["stuff_rv"]), "rmse_type_mean": _rmse(v["rv"], base),
                    "pitcher_types": int(len(t)), "pitcher_r": float(t["pred"].corr(t["actual"]))}
    c = values[values["command_rv"].notna() & values["rv"].notna()]
    g = c.groupby(["pitcher", "season"]).agg(n=("rv", "size"), pred=("command_rv", "mean"), actual=("rv", "mean"))
    g = g[g["n"] >= stuff.COMMAND_MIN_PITCHES]
    out["command"] = {"pitchers": int(len(g)), "pitcher_r": float(g["pred"].corr(g["actual"]))}
    return out


def stage_models() -> stuff.StuffModel:
    mlb = pd.concat([pitches.load(s, "mlb") for s in FIT_SEASONS], ignore_index=True)
    log.info(f"stuff_fit: {len(mlb):,} MLB pitches {FIT_SEASONS}")
    mlb["rv_x"] = expected_target(mlb)
    model = fit_models(mlb)
    values = stuff.pitch_values(model, mlb)
    grade_scales(model, values)
    model.evaluation["reliability"] = reliability_floors(model, values)
    fit_means = mlb[mlb["pitcher"] % HELD_OUT != 0].groupby("pitch_type")["rv"].mean().to_dict()
    model.evaluation["held_out"] = check(values[values["pitcher"] % HELD_OUT == 0], fit_means)
    for fam, m in model.evaluation["held_out"].items():
        log.info(f"held-out[{fam}]: {m}")
    if not RUN_TEST_SEASON:
        return model
    test = stuff.pitch_values(model, pitches.load(TEST_SEASON, "mlb"))
    model.evaluation["test_2026"] = check(test, fit_means)
    for fam, m in model.evaluation["test_2026"].items():
        log.info(f"2026[{fam}]: {m}")
    log.info(f"floors: {model.floor}")
    return model


def season_grades(model: stuff.StuffModel, level: str, seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per pitcher, season and pitch type at one level: readings, pitch count and stuff grade; and per pitcher-season command."""
    frames = [pitches.load(s, level) for s in seasons if pitches.path(s, level).exists()]
    values = stuff.pitch_values(model, pd.concat(frames, ignore_index=True))
    t = stuff.by_type(values, ["pitcher", "season"])
    t = t[t["pitch_type"].isin(model.scale)]
    t["family"] = t["pitch_type"].map(pitches.FAMILY)
    t["stuff_grade"] = stuff.stuff_grade(t["stuff_rv100"], t["pitch_type"], model.scale)
    c = stuff.by_pitcher_command(values, ["pitcher", "season"])
    c["command_grade"] = stuff.command_grade(c["command_rv100"], model.command_scale)
    return t, c


def _pairs(low: pd.DataFrame, high: pd.DataFrame, model: stuff.StuffModel, same_season: bool) -> pd.DataFrame:
    """Pitcher-type rows measured at both levels at or above the type's floor: same season, or the first higher-level season at or after."""
    lo = low[low["n"] >= low["pitch_type"].map(model.floor)]
    hi = high[high["n"] >= high["pitch_type"].map(model.floor)]
    if same_season:
        return lo.merge(hi, on=["pitcher", "season", "pitch_type", "family"], suffixes=("_lo", "_hi"))
    both = lo.merge(hi, on=["pitcher", "pitch_type", "family"], suffixes=("_lo", "_hi"))
    both = both[both["season_hi"] >= both["season_lo"]]
    return both.sort_values("season_hi").groupby(["pitcher", "season_lo", "pitch_type"]).head(1)


def _command_pairs(low: pd.DataFrame, high: pd.DataFrame, floor: int, same_season: bool) -> pd.DataFrame:
    lo, hi = low[low["command_n"] >= floor], high[high["command_n"] >= floor]
    if same_season:
        return lo.merge(hi, on=["pitcher", "season"], suffixes=("_lo", "_hi"))
    both = lo.merge(hi, on="pitcher", suffixes=("_lo", "_hi"))
    both = both[both["season_hi"] >= both["season_lo"]]
    return both.sort_values("season_hi").groupby(["pitcher", "season_lo"]).head(1)


def _weighted_offset(pairs: pd.DataFrame, column: str, weight: pd.Series, rng: np.random.Generator) -> dict:
    """Sample-weighted mean of (higher minus lower level), with a pitcher-clustered bootstrap interval."""
    diff = pairs[f"{column}_hi"] - pairs[f"{column}_lo"]
    ok = diff.notna()
    diff, w, who = diff[ok], weight[ok], pairs.loc[ok, "pitcher"]
    if diff.empty:
        return {"value": float("nan"), "ci": [float("nan"), float("nan")], "pitchers": 0}
    groups = pd.Series(np.arange(len(diff)), index=diff.index).groupby(who.to_numpy()).apply(list).to_list()
    d, ww = diff.to_numpy(), w.to_numpy()
    boot = []
    for _ in range(BOOTSTRAP):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        boot.append(np.average(d[idx], weights=ww[idx]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"value": float(np.average(d, weights=ww)), "ci": [float(lo), float(hi)], "pitchers": int(who.nunique())}


def level_offsets(model: stuff.StuffModel, low: tuple, high: tuple) -> dict:
    """Higher minus lower level per reading, per family stuff grade and for the command grade, from same-season pairs."""
    rng = np.random.default_rng(0)
    pairs = _pairs(low[0], high[0], model, same_season=True)
    w = np.minimum(pairs["n_lo"], pairs["n_hi"])
    out = {"pairs": int(len(pairs))}
    for r in stuff.READINGS:
        out[r] = _weighted_offset(pairs, r, w, rng)
    for fam in pitches.FAMILIES:
        f = pairs[pairs["family"].eq(fam)]
        out[f"stuff_grade:{fam}"] = _weighted_offset(f, "stuff_grade", np.minimum(f["n_lo"], f["n_hi"]), rng)
    c = _command_pairs(low[1], high[1], model.floor["command"], same_season=True)
    out["command_grade"] = _weighted_offset(c, "command_grade", np.minimum(c["command_n_lo"], c["command_n_hi"]), rng)
    return out


def shifted(frame: pd.DataFrame, offsets: dict, command: bool = False) -> pd.DataFrame:
    """Readings and grades moved to the MLB scale by a level's offsets."""
    out = frame.copy()
    if command:
        out["command_grade"] = (out["command_grade"] + offsets["command_grade"]["value"]).clip(20, 80)
        return out
    for r in stuff.READINGS:
        out[r] = out[r] + offsets[r]["value"]
    shift = out["family"].map(lambda f: offsets[f"stuff_grade:{f}"]["value"])
    out["stuff_grade"] = (out["stuff_grade"] + shift).clip(20, 80)
    return out


def carry_over(model: stuff.StuffModel, low: tuple, mlb: tuple, offsets: dict, seasons: range) -> dict:
    """A level's grades (offset-adjusted) against the same pitchers' first MLB season at or after, per family and for command."""
    out = {}
    lo = shifted(low[0][low[0]["season"].isin(seasons)], offsets)
    pairs = _pairs(lo, mlb[0], model, same_season=False)
    pairs = pairs[pairs["season_hi"] > pairs["season_lo"]] if CARRY_LATER_SEASON else pairs
    for fam in pitches.FAMILIES:
        f = pairs[pairs["family"].eq(fam)]
        r, bias = float(f["stuff_grade_lo"].corr(f["stuff_grade_hi"])), float((f["stuff_grade_hi"] - f["stuff_grade_lo"]).mean())
        out[fam] = {"r": r, "bias": bias, "n": int(len(f)), "passed": bool(r >= CARRY_MIN_R and abs(bias) <= CARRY_MAX_BIAS)}
    if np.isfinite(offsets["command_grade"]["value"]):
        lc = shifted(low[1][low[1]["season"].isin(seasons)], offsets, command=True)
        c = _command_pairs(lc, mlb[1], model.floor["command"], same_season=False)
        c = c[c["season_hi"] > c["season_lo"]] if CARRY_LATER_SEASON else c
        r, bias = float(c["command_grade_lo"].corr(c["command_grade_hi"])), float((c["command_grade_hi"] - c["command_grade_lo"]).mean())
        out["command"] = {"r": r, "bias": bias, "n": int(len(c)), "passed": bool(r >= CARRY_MIN_R and abs(bias) <= CARRY_MAX_BIAS)}
    return out


def stage_levels(model: stuff.StuffModel) -> stuff.StuffModel:
    mlb = season_grades(model, "mlb", FIT_SEASONS + [TEST_SEASON])
    aaa = season_grades(model, "aaa", [2022, 2023, 2024, 2025, 2026])
    fsl = season_grades(model, "fsl", [2021, 2022, 2023, 2024, 2025, 2026])
    spring = season_grades(model, "spring", [2021, 2022, 2023, 2024, 2025, 2026])
    model.offsets = {"aaa": level_offsets(model, aaa, mlb)}
    to_aaa = level_offsets(model, fsl, aaa)
    model.offsets["fsl_adjusted"] = bool(to_aaa["pairs"] >= FSL_MIN_PAIRS)
    # FSL to MLB is chained through Triple-A; intervals are reported per link.
    model.offsets["fsl"] = {k: ({"value": v["value"] + model.offsets["aaa"][k]["value"], "ci_fsl_to_aaa": v["ci"], "pitchers": v["pitchers"]}
                                if isinstance(v, dict) else v) for k, v in to_aaa.items()}
    model.offsets["spring"] = level_offsets(model, spring, mlb)
    model.offsets["spring_adjusted"] = bool(model.offsets["spring"]["pairs"] >= SPRING_MIN_PAIRS)
    # Triple-A's check releases Triple-A and FSL pitches and spring's releases spring; FSL's own check is reported only.
    model.carry = {"aaa": carry_over(model, aaa, mlb, model.offsets["aaa"], range(2022, 2026)),
                   "spring": carry_over(model, spring, mlb, model.offsets["spring"], range(2021, 2026)),
                   "fsl": carry_over(model, fsl, mlb, model.offsets["fsl"], range(2021, 2026))}
    for level in ("aaa", "fsl", "spring"):
        log.info(f"offsets[{level}]: " + ", ".join(f"{k} {v['value']:+.2f}" if isinstance(v, dict) else f"{k} {v}"
                                                 for k, v in model.offsets[level].items()))
    log.info(f"adjusted: fsl {model.offsets['fsl_adjusted']} ({to_aaa['pairs']}), spring {model.offsets['spring_adjusted']} "
             f"({model.offsets['spring']['pairs']})")
    for level, checks in model.carry.items():
        for k, v in checks.items():
            log.info(f"carry[{level}][{k}]: {v}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["models", "levels"], required=True)
    args = parser.parse_args()
    setup_logging()
    if args.stage == "models":
        model = stage_models()
    else:
        model = stage_levels(stuff.StuffModel.load(STUFF_PATH))
    model.save(STUFF_PATH)
    log.info(f"stuff model saved to {STUFF_PATH} ({STUFF_PATH.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
