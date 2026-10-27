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
OUTCOMES = ["ba", "obp", "slg", "ops", "hr600", "so600", "bb600", "carry"]
TILT_GROUPS = [-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf]
TILT_LABELS = ["flattened 1.5+", "flattened a bit", "about the same", "steepened a bit", "steepened 1.5+"]


def swing_profiles(pitches: pd.DataFrame, curves: dict[str, list[float]]) -> pd.DataFrame:
    """Mean belt-high tilt and attack angle and mean bat speed over every tracked swing of each hitter-season."""
    tracked = pitches[pitches.swing_path_tilt.notna() & pitches.attack_angle.notna() & pitches.plate_z.notna()]
    belt = at_belt_high(tracked, curves)
    frame = pd.concat([tracked[["batter", "season", "bat_speed"]], belt[["tilt", "attack"]]], axis=1)
    out = frame.groupby(["batter", "season"]).agg(tracked=("tilt", "size"), tilt=("tilt", "mean"), attack=("attack", "mean"),
                                                  bat_speed=("bat_speed", "mean"))
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
    of attack angle per standard deviation of whiff rate; bat speed change held as a control."""
    whiff_mean, whiff_sd = float(p.whiff_0.mean()), float(p.whiff_0.std())
    z = (p.whiff_0 - whiff_mean) / whiff_sd
    names = ["d_tilt", "d_attack", "d_attack_x_whiff", "d_bat_speed"]
    design = np.column_stack([np.ones(len(p)), p.d_tilt, p.d_attack, p.d_attack * z, p.d_bat_speed])
    out = {"n": int(len(p)), "whiff_mean": round(whiff_mean, 4), "whiff_sd": round(whiff_sd, 4), "outcomes": {}}
    for col in OUTCOMES:
        y = p[f"d_{col}"].to_numpy()
        ok = np.isfinite(y)
        X, y = design[ok], y[ok]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * resid.var() * len(y) / (len(y) - X.shape[1]))
        out["outcomes"][col] = {name: [round(float(b), 5), round(float(s), 5)] for name, b, s in zip(names, beta[1:], se[1:])}
    return out


def groups(p: pd.DataFrame) -> list[dict]:
    """Mean change by tilt-change group, relative to hitters whose tilt held steady."""
    g = pd.cut(p.d_tilt, TILT_GROUPS, labels=TILT_LABELS)
    means = p.groupby(g, observed=True)[[f"d_{c}" for c in OUTCOMES]].mean()
    rel = means - means.loc["about the same"]
    counts = p.groupby(g, observed=True).size()
    return [{"group": str(k), "n": int(counts[k]), **{c: round(float(rel.loc[k, f"d_{c}"]), 4) for c in OUTCOMES}} for k in rel.index]
