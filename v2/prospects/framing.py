"""Catcher framing runs at the tracked minor league levels (Triple-A, Florida State League).

Only taken pitches the umpire called (``called_strike`` or ``ball``) count, with ``fielder_2`` as the
catcher. A called-strike probability is fitted per level and season from location (``plate_x``, the
height within the batter's zone and the zone's own top and bottom), batter side and count, cross-fitted
by game so no pitch is scored by a model that saw it. Each call is worth (called strike minus that
probability) times the run value of a strike over a ball in its count: the MLB run expectancy table
(``cache/pitch_value/re288.csv``) averaged over bases and outs with MLB weights, the way pitch run value
prices a pitch. Positive runs are runs the catcher saved.

Triple-A called every pitch by the automated zone on some days in 2023 and 2024, and the Florida State
League in some 2021 games. Those games carry no framing, so a game whose calls disagree with the
rulebook zone on fewer than ``ABS_MAX_MISS`` of its called pitches is dropped. The challenge-system
games that remain still reward framing, though less than a human-only zone.

Usage:
    python -m v2.prospects.framing            # build the table and print the carry-over checks
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import xgboost as xgb

from backend.log import setup_logging
from v2.players import savant
from v2.prospects.feeds import CACHE_DIR
from v2.prospects.stuff_fit import CARRY_MIN_PAIRS

log = logging.getLogger(__name__)

STATCAST_DIR = CACHE_DIR.parent
LEVEL_SEASONS = {"aaa": range(2022, 2027), "fsl": range(2021, 2027)}
MLB_SEASONS = range(2015, 2027)
TABLE_PATH = CACHE_DIR / "framing.parquet"
RE288_PATH = STATCAST_DIR / "pitch_value" / "re288.csv"
CALLED = ("called_strike", "ball")
COLUMNS = ["game_pk", "description", "plate_x", "plate_z", "sz_top", "sz_bot", "stand", "balls", "strikes", "fielder_2"]
FEATURES = ["plate_x_bat", "z_rel", "plate_z", "sz_top", "sz_bot", "stand_r", "balls", "strikes"]
HALF_PLATE_FT = 17 / 24
BALL_RADIUS_FT = 0.121
ABS_MAX_MISS = 0.01  # human games run 3 to 10% off the rulebook zone; automated-zone games read 0 to 1%
MIN_GAME_CALLED = 30
FOLDS = 5
PARAMS = {"objective": "binary:logistic", "max_depth": 6, "eta": 0.1, "min_child_weight": 50, "subsample": 0.8, "verbosity": 0}
ROUNDS = 200
MIN_CALLED = 500  # a catcher season's called pitches before it enters a carry-over pair


def count_values() -> pd.Series:
    """Run value of a ball over a strike in each count, ``balls-strikes`` keys, from the defense's side (positive: a strike saves runs).

    A strike in a two-strike count is a strikeout, a ball in a three-ball count a walk; both are
    averaged over the MLB bases-and-outs mix of the count they are thrown in.
    """
    re = pd.read_csv(RE288_PATH)
    runs = re.set_index(["balls", "strikes", "outs", "bases"])["runs"]

    def after(b: int, s: int, outs: int, bases: int, call: str) -> float:
        if call == "strike":
            if s < 2:
                return runs[(b, s + 1, outs, bases)]
            return 0.0 if outs == 2 else runs[(0, 0, outs + 1, bases)]
        if b < 3:
            return runs[(b + 1, s, outs, bases)]
        # A walk forces runners only as far as the bases behind them are full.
        first, second, third = bases & 1, bases >> 1 & 1, bases >> 2 & 1
        scored = int(first and second and third)
        new = 1 | (2 if first or second else 0) | (4 if (first and second) or third else 0)
        return scored + runs[(0, 0, outs, new)]

    out = {}
    for (b, s), cell in re.groupby(["balls", "strikes"]):
        w = cell["pitches"].to_numpy(float)
        diff = [after(b, s, o, base, "ball") - after(b, s, o, base, "strike") for o, base in zip(cell["outs"], cell["bases"])]
        out[f"{b}-{s}"] = float(np.average(diff, weights=w))
    return pd.Series(out)


def called_pitches(level: str, season: int) -> pd.DataFrame:
    """Called pitches with a catcher and full location, automated-zone games removed."""
    d = pd.read_parquet(STATCAST_DIR / f"statcast_{level}_{season}.parquet", columns=COLUMNS)
    d = d[d["description"].isin(CALLED)].dropna(subset=["plate_x", "plate_z", "sz_top", "sz_bot", "fielder_2"]).copy()
    d["strike"] = d["description"].eq("called_strike").astype(float)
    in_zone = ((d["plate_x"].abs() <= HALF_PLATE_FT + BALL_RADIUS_FT) & (d["plate_z"] <= d["sz_top"] + BALL_RADIUS_FT)
               & (d["plate_z"] >= d["sz_bot"] - BALL_RADIUS_FT))
    games = (d["strike"].astype(bool) != in_zone).groupby(d["game_pk"]).agg(["mean", "size"])
    keep = games.index[(games["mean"] >= ABS_MAX_MISS) & (games["size"] >= MIN_GAME_CALLED)]
    log.info(f"framing[{level} {season}]: {len(games) - len(keep):,} of {len(games):,} games dropped (automated zone or too few calls)")
    d = d[d["game_pk"].isin(keep)].copy()
    d["stand_r"] = d["stand"].eq("R").astype(float)
    d["plate_x_bat"] = np.where(d["stand"].eq("R"), d["plate_x"], -d["plate_x"])  # positive away from the batter
    d["z_rel"] = (d["plate_z"] - d["sz_bot"]) / (d["sz_top"] - d["sz_bot"]).where(lambda s: s > 0)
    d["fielder_2"] = d["fielder_2"].astype("int64")
    d["level"], d["season"] = level, season
    return d


def expected_strike(d: pd.DataFrame) -> np.ndarray:
    """Called-strike probability per pitch, each fold scored by a model fitted on the other games."""
    fold = d["game_pk"].to_numpy() % FOLDS
    out = np.empty(len(d))
    X = d[FEATURES].astype(float)
    for k in range(FOLDS):
        tr, te = fold != k, fold == k
        b = xgb.train(PARAMS, xgb.DMatrix(X[tr], label=d["strike"].to_numpy()[tr]), ROUNDS)
        out[te] = b.predict(xgb.DMatrix(X[te]))
    return out


def level_season(level: str, season: int, values: pd.Series) -> pd.DataFrame:
    d = called_pitches(level, season)
    d["p_strike"] = expected_strike(d)
    d["rv"] = (d["strike"] - d["p_strike"]) * (d["balls"].astype(str) + "-" + d["strikes"].astype(str)).map(values).to_numpy(float)
    g = d.groupby("fielder_2")
    out = pd.DataFrame({"called": g.size(), "strikes_added": g["strike"].sum() - g["p_strike"].sum(), "runs": g["rv"].sum()})
    out.index.name = "player_id"
    out = out.reset_index()
    out["level"], out["season"] = level, season
    log.info(f"framing[{level} {season}]: {len(d):,} called pitches, {len(out)} catchers, log loss "
             f"{-np.mean(d['strike'] * np.log(d['p_strike']) + (1 - d['strike']) * np.log(1 - d['p_strike'])):.4f}")
    return out


def build() -> pd.DataFrame:
    """One row per catcher, level and season: called pitches, strikes added and framing runs, plus runs per 100 called pitches."""
    values = count_values()
    table = pd.concat([level_season(level, s, values) for level, seasons in LEVEL_SEASONS.items() for s in seasons], ignore_index=True)
    table["runs_per_100"] = 100 * table["runs"] / table["called"]
    return table


def mlb_framing() -> pd.DataFrame:
    """The corrected Savant MLB framing boards: framing runs and called pitches per catcher and season."""
    frames = []
    for s in MLB_SEASONS:
        b = savant.load_board("catcher_framing", s)
        frames.append(b[["player_id", "season", "pitches", "rv_tot"]])
    out = pd.concat(frames, ignore_index=True).rename(columns={"pitches": "called", "rv_tot": "runs"})
    out["runs_per_100"] = 100 * out["runs"] / out["called"]
    return out


def _corr(a: pd.Series, b: pd.Series, w: pd.Series | None = None) -> float:
    if len(a) < 3:
        return float("nan")
    if w is None:
        return float(np.corrcoef(a, b)[0, 1])
    w = w / w.sum()
    ma, mb = (w * a).sum(), (w * b).sum()
    return float((w * (a - ma) * (b - mb)).sum() / np.sqrt((w * (a - ma) ** 2).sum() * (w * (b - mb) ** 2).sum()))


def _check(pairs: pd.DataFrame, lo: str = "runs_per_100_lo", hi: str = "runs_per_100_hi") -> dict:
    n = len(pairs)
    return {"pairs": int(n), "catchers": int(pairs["player_id"].nunique()) if n else 0,
            "r": _corr(pairs[lo], pairs[hi]),
            "r_weighted": _corr(pairs[lo], pairs[hi], np.minimum(pairs["called_lo"], pairs["called_hi"])) if n >= 3 else float("nan"),
            "established": bool(n >= CARRY_MIN_PAIRS)}


def carry_over(table: pd.DataFrame, mlb: pd.DataFrame, min_called: int = MIN_CALLED) -> dict:
    """Year-to-year repeatability within each level, and each level against the same catchers' MLB framing.

    MLB pairs read the minor league season against the MLB season of the same year (the catcher split his
    season) and the next one. The Savant boards list qualified catchers only, so these pairs are the
    catchers who went on to play regularly.
    """
    t = table[table["called"] >= min_called]
    out = {"min_called": min_called}
    for level in LEVEL_SEASONS:
        lv = t[t["level"] == level]
        nxt = lv.assign(season=lv["season"] - 1)
        pairs = lv.merge(nxt, on=["player_id", "season", "level"], suffixes=("_lo", "_hi"))
        out[f"{level}_year_to_year"] = _check(pairs)
        for lag, name in ((0, "same"), (1, "next")):
            m = mlb.assign(season=mlb["season"] - lag)
            pairs = lv.merge(m, on=["player_id", "season"], suffixes=("_lo", "_hi"))
            out[f"{level}_to_mlb_{name}"] = _check(pairs)
        m = mlb.rename(columns={"season": "season_hi"})
        p = lv.merge(m, on="player_id", suffixes=("_lo", "_hi"))
        p = p[(p["season_hi"] >= p["season"]) & (p["season_hi"] <= p["season"] + 1)]
        out[f"{level}_to_mlb_either"] = _check(p)
    # Florida State League catchers who reached Triple-A: the chain the FSL readings would have to carry through.
    a = t[t["level"] == "aaa"][["player_id", "season", "called", "runs_per_100"]]
    f = t[t["level"] == "fsl"][["player_id", "season", "called", "runs_per_100"]]
    p = f.merge(a.rename(columns={"season": "season_hi"}), on="player_id", suffixes=("_lo", "_hi"))
    p = p[(p["season_hi"] >= p["season"]) & (p["season_hi"] <= p["season"] + 2)]
    out["fsl_to_aaa"] = _check(p)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-called", type=int, default=MIN_CALLED)
    args = parser.parse_args()
    setup_logging()
    table = build()
    table.to_parquet(TABLE_PATH, index=False)
    log.info(f"framing: {len(table):,} catcher seasons written to {TABLE_PATH}")
    for name, check in carry_over(table, mlb_framing(), args.min_called).items():
        log.info(f"framing carry {name}: {check}")


if __name__ == "__main__":
    main()
