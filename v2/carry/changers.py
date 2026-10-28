"""Hitters whose swing changed between consecutive seasons, and what changed with it.

These are natural changes, not experiments, so the coefficients describe what tended to follow a change rather than what
a change would cause. The lab uses them with their standard errors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from v2.carry.swing import at_belt_high

MIN_SWINGS = 300
MIN_PA = 250
COMPETITIVE_SHARE = 0.9
OUTCOMES = ["ba", "obp", "slg", "ops", "hr600", "so600", "bb600", "carry"]
TILT_GROUPS = [-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf]
TILT_LABELS = ["flattened 1.5+", "flattened a bit", "about the same", "steepened a bit", "steepened 1.5+"]


def swing_profiles(pitches: pd.DataFrame, curves: dict[str, list[float]]) -> pd.DataFrame:
    """Mean belt-high tilt and attack angle over every swing with a tracked path, and mean bat speed over the fastest 90% of
    every swing with a tracked speed (Savant's competitive-swing rule, which drops checked and defensive swings), per
    hitter-season. The build passes no bunts."""
    tracked = pitches[pitches.swing_path_tilt.notna() & pitches.attack_angle.notna() & pitches.plate_z.notna()]
    belt = at_belt_high(tracked, curves)
    out = pd.concat([tracked[["batter", "season"]], belt[["tilt", "attack"]]], axis=1).groupby(["batter", "season"]).agg(
        tracked=("tilt", "size"), tilt=("tilt", "mean"), attack=("attack", "mean"))
    speeds = pitches.loc[pitches.bat_speed.notna(), ["batter", "season", "bat_speed"]]
    floor = speeds.groupby(["batter", "season"]).bat_speed.transform(lambda s: s.quantile(1 - COMPETITIVE_SHARE))
    out["bat_speed"] = speeds[speeds.bat_speed >= floor].groupby(["batter", "season"]).bat_speed.mean()
    return out.reset_index()


def rates(counts: pd.DataFrame) -> pd.DataFrame:
    c = counts
    out = c[["batter", "season", "pa", "swings", "whiffs"]].copy()
    out["ba"] = c.h / c.ab
    out["obp"] = (c.h + c.bb + c.hbp) / (c.ab + c.bb + c.hbp + c.sf)
    out["slg"] = c.tb / c.ab
    out["ops"] = out.obp + out.slg
    for col in ("hr", "so", "bb"):
        out[f"{col}600"] = c[col] / c.pa * 600
    out["whiff"] = c.whiffs / c.swings
    return out


def pairs(profiles: pd.DataFrame, counts: pd.DataFrame, carry: pd.DataFrame) -> pd.DataFrame:
    """carry: per hitter-season mean carry. Returns one row per consecutive-season pair with changes."""
    frame = profiles.merge(rates(counts), on=["batter", "season"]).merge(carry, on=["batter", "season"], how="left")
    frame = frame[(frame.tracked >= MIN_SWINGS) & (frame.pa >= MIN_PA)]
    nxt = frame.assign(season=frame.season - 1)
    p = frame.merge(nxt, on=["batter", "season"], suffixes=("_0", "_1"))
    for col in ["tilt", "attack", "bat_speed"] + OUTCOMES:
        p[f"d_{col}"] = p[f"{col}_1"] - p[f"{col}_0"]
    return p


def coefficients(p: pd.DataFrame) -> dict:
    """Per outcome: change per degree of tilt, per degree of attack angle at average whiff rate, and the extra per degree
    of attack angle per standard deviation of whiff rate; bat speed change held as a control. `cov` is the coefficients'
    covariance in that order, clustered by hitter because a hitter contributes one pair per consecutive-season span."""
    whiff_mean, whiff_sd = float(p.whiff_0.mean()), float(p.whiff_0.std())
    z = (p.whiff_0 - whiff_mean) / whiff_sd
    names = ["d_tilt", "d_attack", "d_attack_x_whiff", "d_bat_speed"]
    design = np.column_stack([np.ones(len(p)), p.d_tilt, p.d_attack, p.d_attack * z, p.d_bat_speed])
    out = {"n": int(len(p)), "n_hitters": int(p.batter.nunique()), "whiff_mean": round(whiff_mean, 4),
           "whiff_sd": round(whiff_sd, 4), "outcomes": {}}
    for col in OUTCOMES:
        y = p[f"d_{col}"].to_numpy()
        ok = np.isfinite(y)
        X, y = design[ok], y[ok]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        cov = _clustered_cov(X, y - X @ beta, p.batter.to_numpy()[ok])[1:, 1:]
        out["outcomes"][col] = {**{name: _sig(b) for name, b in zip(names, beta[1:])}, "cov": [[_sig(c) for c in row] for row in cov]}
    return out


def _clustered_cov(X: np.ndarray, resid: np.ndarray, cluster: np.ndarray) -> np.ndarray:
    """Cluster-robust (sandwich) covariance of OLS coefficients with the usual small-sample scaling."""
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    scores = pd.DataFrame(X * resid[:, None]).groupby(cluster).sum().to_numpy()
    g = len(scores)
    return bread @ (scores.T @ scores) @ bread * g / (g - 1) * (n - 1) / (n - k)


def _sig(x: float, digits: int = 6) -> float:
    return float(f"{float(x):.{digits}g}")


def groups(p: pd.DataFrame) -> list[dict]:
    """Mean change by tilt-change group, relative to hitters whose tilt held steady."""
    g = pd.cut(p.d_tilt, TILT_GROUPS, labels=TILT_LABELS)
    means = p.groupby(g, observed=True)[[f"d_{c}" for c in OUTCOMES]].mean()
    rel = means - means.loc["about the same"]
    counts = p.groupby(g, observed=True).size()
    return [{"group": str(k), "n": int(counts[k]), **{c: round(float(rel.loc[k, f"d_{c}"]), 4) for c in OUTCOMES}} for k in rel.index]
