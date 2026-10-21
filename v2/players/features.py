"""Feature assembly shared by fitting and scoring: traits, composites, and projections.

The same code turns a window of inputs into trait composites and a projected
runs figure in both places, so the weights are learned on exactly what they
are scored with.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from v2.data.pa_dataset import OUTCOMES
from v2.players import runs as runs_mod
from v2.players.inputs import INPUTS, prior_rates
from v2.players.model import ValueModel

Z_CAP = 3.0
MIN_WEIGHT_COVERAGE = 0.6
MIN_INPUT_AVAILABILITY = 0.5
HITTER_POOL, PITCHER_POOL = "hitter", "pitcher"
PITCHER_GROUPS = {"SP", "RP"}
CATCHER_INPUTS = {"arm.cs_above_average", "arm.pop_time", "reliability.blocking_runs", "reliability.framing_runs"}
# Seasonal age in the projected season enters the grade as (age - 28) and its square; unknown counts as 28.
AGE_CENTER = 28.0
AGE_FEATURES = ("age", "age2")


@dataclass(frozen=True)
class TraitSpec:
    name: str
    pool: str
    panel: str
    label: str


TRAITS: dict[str, TraitSpec] = {t.name: t for t in [
    TraitSpec("contact", HITTER_POOL, "batting", "Contact"),
    TraitSpec("power", HITTER_POOL, "batting", "Power"),
    TraitSpec("discipline", HITTER_POOL, "batting", "Discipline"),
    TraitSpec("bbq", HITTER_POOL, "batting", "Batted-ball quality"),
    TraitSpec("platoon", HITTER_POOL, "batting", "Platoon"),
    TraitSpec("range", HITTER_POOL, "fielding", "Range"),
    TraitSpec("arm", HITTER_POOL, "fielding", "Arm"),
    TraitSpec("reliability", HITTER_POOL, "fielding", "Reliability"),
    TraitSpec("baserunning", HITTER_POOL, "fielding", "Baserunning"),
    TraitSpec("speed", HITTER_POOL, "fielding", "Sprint speed"),
    TraitSpec("stuff", PITCHER_POOL, "arsenal", "Stuff"),
    TraitSpec("command", PITCHER_POOL, "arsenal", "Command"),
    TraitSpec("batmiss", PITCHER_POOL, "results", "Bat-missing"),
    TraitSpec("contactmgmt", PITCHER_POOL, "results", "Contact management"),
    TraitSpec("workload", PITCHER_POOL, "durability", "Workload"),
]}
INPUT_KIND = {name: ("physical" if name in ("speed.sprint_speed", "power.bat_speed", "stuff.velo", "arm.strength") else "measured") for name in INPUTS}
# The sample unit that gates a grade: weighted plate appearances or batters faced.
MIN_GRADE_SAMPLE = 100.0


def pool_of(group: pd.Series) -> pd.Series:
    return np.where(group.isin(PITCHER_GROUPS), PITCHER_POOL, HITTER_POOL)


def wide_inputs(window: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(values, samples) wide by input, indexed by player_id."""
    values = window.pivot_table(index="player_id", columns="input", values="value", aggfunc="first")
    samples = window.pivot_table(index="player_id", columns="input", values="sample", aggfunc="first")
    return values, samples


def oriented_z(values: pd.DataFrame, pool: pd.Series, season_key: pd.Series | None = None) -> pd.DataFrame:
    """Capped z-scores within pool (and season when given), signed so higher is better."""
    keys = [pool] if season_key is None else [pool, season_key]
    z = values.copy()
    for col in z.columns:
        spec = INPUTS.get(col)
        sign = 1.0 if spec is None or spec.higher_is_better else -1.0
        grouped = z[col].groupby(keys)
        mean, sd = grouped.transform("mean"), grouped.transform("std")
        z[col] = (sign * (z[col] - mean) / sd.where(sd > 0)).clip(-Z_CAP, Z_CAP)
    return z


def input_percentiles(values: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    pct = values.copy()
    for col in pct.columns:
        spec = INPUTS.get(col)
        ranked = pct[col].groupby(group).rank(pct=True)
        pct[col] = (ranked if spec is None or spec.higher_is_better else 1 - ranked + 1 / pct[col].groupby(group).transform("count")) * 100
    return pct


def trait_composites(z: pd.DataFrame, pool: pd.Series, model: ValueModel) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weighted composites per trait and the share of weight each player's inputs covered.

    An input enters a pool's composite only where at least half the pool has it; a
    player missing an available input counts as average (z = 0). A composite is NaN
    when his inputs cover under MIN_WEIGHT_COVERAGE of the trait's weight.
    """
    weights = model.trait_weight_frame()
    comps = pd.DataFrame(index=z.index)
    coverage = pd.DataFrame(index=z.index)
    for trait, spec in TRAITS.items():
        tw = weights[(weights["pool"] == spec.pool) & (weights["trait"] == trait) & (weights["weight"] > 0)]
        members = pool == spec.pool
        if tw.empty or not members.any():
            comps[trait], coverage[trait] = np.nan, np.nan
            continue
        cols = [c for c in tw["input"] if c in z.columns]
        if not cols:
            comps[trait], coverage[trait] = np.nan, np.nan
            continue
        sub = z.loc[members, cols]
        available = sub.notna().mean() >= MIN_INPUT_AVAILABILITY
        cols = [c for c in cols if available[c]]
        if not cols:
            comps[trait], coverage[trait] = np.nan, np.nan
            continue
        w = tw.set_index("input").loc[cols, "weight"].astype(float)
        w = w / w.abs().sum()
        present = sub[cols].notna().astype(float)
        cov = present @ w.abs()
        composite = sub[cols].fillna(0.0) @ w
        comps.loc[members, trait] = composite.where(cov >= MIN_WEIGHT_COVERAGE)
        coverage.loc[members, trait] = cov
    return comps, coverage


def role_rows(rates: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    """Keep each player's row for the role his position group implies (a two-way player has both)."""
    expected = np.where(rates["player_id"].map(group).isin(PITCHER_GROUPS), "pit", "bat")
    return rates[rates["role"].values == expected]


def shrunk_runs(rates: pd.DataFrame, model: ValueModel, roles: pd.Series, leverage: pd.Series | None = None) -> pd.Series:
    """Runs per 600 PA (hitters) or runs prevented per 180/65 IP (pitchers) from shrunk rates.

    ``leverage`` (per player, league mean 1) scales relievers' runs.
    """
    lw = model.linear_weights
    rates = role_rows(rates, roles)
    bat = rates[rates["role"] == "bat"].set_index("player_id")
    pit = rates[rates["role"] == "pit"].set_index("player_id")
    out = pd.Series(dtype="float64")
    if not bat.empty:
        out = pd.concat([out, runs_mod.batting_runs(bat, lw)])
    if not pit.empty:
        role = roles.reindex(pit.index).fillna("RP")
        out = pd.concat([out, runs_mod.pitching_runs(pit, lw, role, model.pa_per_ip, leverage)])
    return out


def pitching_runs_for(rates: pd.DataFrame, model: ValueModel, role: pd.Series, leverage: pd.Series | None = None) -> pd.Series:
    """Runs prevented from a frame of pitching rate rows regardless of position group (two-way players)."""
    pit = rates[rates["role"] == "pit"].set_index("player_id")
    if pit.empty:
        return pd.Series(dtype="float64")
    return runs_mod.pitching_runs(pit, model.linear_weights, role.reindex(pit.index).fillna("SP"), model.pa_per_ip, leverage)


def grade_features(composites: pd.DataFrame, shrunk: pd.Series, pool: pd.Series, age: pd.Series) -> pd.DataFrame:
    """The design matrix for the group grade: trait composites, the shrunk-runs feature, and age."""
    x = composites.copy()
    x["shrunk_runs"] = shrunk.reindex(x.index)
    centred = (pd.to_numeric(age, errors="coerce").astype("float64").reindex(x.index) - AGE_CENTER).fillna(0.0)
    x["age"], x["age2"] = centred, centred ** 2
    x["pool"] = pool
    return x


def project_runs(features: pd.DataFrame, group: pd.Series, model: ValueModel) -> pd.Series:
    """Projected next-season runs above average per full season, by the fitted grade weights.

    Pools for the grade fit are hitter, SP, and RP; a missing composite counts as average.
    """
    gw = model.grade_weight_frame()
    out = pd.Series(np.nan, index=features.index)
    fit_pool = pd.Series(np.where(group.isin(PITCHER_GROUPS), group, HITTER_POOL), index=features.index)
    for pool_name, rows in gw.groupby("pool"):
        members = fit_pool == pool_name
        if not members.any():
            continue
        coef = rows.set_index("feature")["coefficient"]
        intercept = float(coef.get("intercept", 0.0))
        total = pd.Series(intercept, index=features.index[members])
        for feature, c in coef.items():
            if feature == "intercept" or feature not in features.columns:
                continue
            total = total + float(c) * features.loc[members, feature].fillna(0.0)
        out.loc[members] = total
    return out


def to_grade_scale(values: pd.Series, group: pd.Series, reference: pd.Series | None = None) -> pd.Series:
    """20-80 within group: mean 50, sd 10 among graded players, clipped.

    ``reference`` supplies the players whose mean and spread define the scale
    (the fully graded ones), so provisional grades map onto the same numbers
    without entering the scale.
    """
    g = (values if reference is None else reference).groupby(group)
    mean, sd = g.transform("mean"), g.transform("std")
    if reference is not None:
        mean, sd = mean.groupby(group).first().reindex(group).set_axis(group.index), sd.groupby(group).first().reindex(group).set_axis(group.index)
    scaled = 50 + 10 * (values - mean) / sd.where(sd > 0)
    # A group of one, or one with no spread, grades at the mean.
    scaled = scaled.where(scaled.notna() | values.isna(), 50.0)
    return scaled.clip(20, 80)


def reliability_band(sample: pd.Series, k: pd.Series | float) -> pd.Series:
    """Half-width of the grade band: 10 × sqrt(1 − r), r = n / (n + k); k = 0 means no band."""
    k = pd.Series(k, index=sample.index) if np.isscalar(k) else pd.Series(k, index=sample.index).astype(float)
    r = (sample / (sample + k)).where(k > 0, 1.0)
    return 10.0 * np.sqrt((1 - r).clip(lower=0))


def dirichlet_draws(counts: pd.DataFrame, group_of: pd.Series, strength: float, k: int, seed: int = 7,
                    prior_counts: pd.DataFrame | None = None) -> np.ndarray:
    """K draws of outcome rates per player from the shrunk Dirichlet posterior, shape (rows, k, 8)."""
    cols = list(OUTCOMES)
    frame = counts[["player_id", "role", *cols]].copy()
    frame["group"] = frame["player_id"].map(group_of)
    frame["group"] = frame["group"].fillna(pd.Series(np.where(frame["role"].eq("pit"), "RP", "DH"), index=frame.index))
    alpha = frame[cols].values + strength * prior_rates(frame, strength, prior_counts)
    rng = np.random.default_rng(seed)
    draws = np.empty((len(frame), k, len(cols)))
    for i in range(len(frame)):
        draws[i] = rng.dirichlet(np.maximum(alpha[i], 1e-6), size=k)
    return draws
