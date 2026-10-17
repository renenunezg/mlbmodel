"""Dated, role-specific PA/BF forecasts for each future season.

Each horizon is fitted directly from observed history. Expected workloads are
never fed back as observations, and absent target players remain zero outcomes.
Five years of history preserve pre-injury workloads; dated reserve membership
separates a retained player from someone who has left MLB.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from v2.players import aging, bio

FIRST_HISTORY_SEASON = 2005
# Earlier dated reserve lists have materially thinner historical coverage.
FIRST_ORIGIN = 2010
MAX_HORIZON = 15
LAGS = tuple(f"lag{i}" for i in range(1, 6))
FEATURES = (*LAGS, "peak", "older", "older_volume", "young_volume", "rostered", "rostered_peak", "dormant", "quality", "quality_volume")
PENALTIES = np.logspace(-3, 3, 25)


def load_lines(seasons: list[int]) -> pd.DataFrame:
    """Complete league populations, not career lines selected from today's roster."""
    parts = []
    for group, role, column in (("hitting", "bat", "plateAppearances"), ("pitching", "pit", "battersFaced")):
        f = bio.league_season_lines(seasons, group)
        components = aging.hitting_components(f) if role == "bat" else aging.pitching_components(f)
        component = "bat_runs_600" if role == "bat" else "pitch_runs_180"
        components["weighted"] = components[component] * components["volume"]
        totals = components.groupby("season")[["weighted", "volume"]].sum()
        centre = totals["weighted"] / totals["volume"]
        components["quality"] = components[component] - components["season"].map(centre)
        f = f.merge(components[["player_id", "season", "quality"]], on=["player_id", "season"], how="left", validate="one_to_one")
        f["quality"] = f["quality"].fillna(0.0)
        f = f.assign(role=role, volume=pd.to_numeric(f[column]).astype(float))
        # Normalize the scheduled 60-game season as an input, never a target.
        f.loc[f["season"].eq(2020), "volume"] *= 162 / 60
        parts.append(f[["player_id", "role", "season", "age", "volume", "quality"]])
    return pd.concat(parts, ignore_index=True)


def history(lines: pd.DataFrame, first_season: int, roster_ids: set[int]) -> pd.DataFrame:
    """Five completed seasons and membership known at the forecast date."""
    if not set(range(first_season - 5, first_season)) <= set(lines["season"]):
        raise ValueError("Missing completed workload history")
    prior = lines[lines["season"].between(first_season - 5, first_season - 1)]
    age = prior.sort_values("season").drop_duplicates(["player_id", "role"], keep="last")
    age = age.assign(age=age["age"] + first_season - age["season"])[["player_id", "role", "age"]]
    volume = prior.set_index(["player_id", "role", "season"])["volume"]
    for lag in range(1, 6):
        age = age.join(volume.xs(first_season-lag, level="season").rename(f"lag{lag}"), on=["player_id", "role"])
    age[list(LAGS)] = age[list(LAGS)].fillna(0.0)
    recent = prior[prior["season"] >= first_season-3].copy()
    recent["runs"] = recent["quality"] * recent["volume"]
    totals = recent.groupby(["player_id", "role"])[["runs", "volume"]].sum()
    # League-centred quality, with 100 average PA/BF to stabilize small samples.
    quality = (totals["runs"] / (totals["volume"] + 100) / 20).clip(-3, 3).rename("quality")
    age = age.join(quality, on=["player_id", "role"]).fillna({"quality": 0.0})
    return age.assign(rostered=age["player_id"].isin(roster_ids).astype(float))


def design(frame: pd.DataFrame, age: np.ndarray) -> np.ndarray:
    lag = frame[list(LAGS)].to_numpy(float) / 600.0
    older = np.maximum(np.asarray(age, dtype=float) - 30.0, 0.0) / 10.0
    young = np.maximum(27.0 - np.asarray(age, dtype=float), 0.0) / 10.0
    peak, mean = lag.max(axis=1), lag.mean(axis=1)
    rostered = frame["rostered"].to_numpy(float)
    quality = frame["quality"].to_numpy(float)
    return np.column_stack([lag, peak, older, older*mean, young*mean, rostered,
                            rostered*peak, (lag[:, :2].sum(axis=1) == 0).astype(float), quality, quality*mean])


def training_frame(lines: pd.DataFrame, rosters: dict[int, set[int]], last_target_season: int,
                   max_horizon: int = MAX_HORIZON) -> pd.DataFrame:
    """Every prior eligible player, including those absent from the target season.

    Eligibility matches the previous model: 100 PA/BF in any prior three years.
    Roster status is an input, never a filter based on future participation.
    """
    outcomes = lines.set_index(["player_id", "role", "season"])[["volume", "quality"]]
    frames = []
    for origin, roster_ids in sorted(rosters.items()):
        if origin < FIRST_ORIGIN or origin > last_target_season:
            continue
        h = history(lines, origin, roster_ids)
        h = h[h[list(LAGS[:3])].max(axis=1) >= 100].dropna(subset=["age"])
        for horizon in range(1, min(max_horizon, last_target_season-origin+1) + 1):
            target = origin + horizon - 1
            if target == 2020:
                continue
            actual = outcomes.xs(target, level="season").rename(columns={"volume": "actual", "quality": "target_quality"})
            f = h.join(actual, on=["player_id", "role"]).fillna({"actual": 0.0})
            frames.append(f.assign(origin=origin, target_season=target, horizon=horizon))
    return pd.concat(frames, ignore_index=True)


def _ridge(rows: pd.DataFrame, penalty: float) -> np.ndarray:
    x = np.column_stack([np.ones(len(rows)), design(rows, rows["age"].to_numpy(float))])
    regularizer = np.eye(x.shape[1]) * np.sqrt(penalty)
    regularizer[0, 0] = 0.0
    lower, upper = np.full(x.shape[1], -np.inf), np.full(x.shape[1], np.inf)
    # Older age cannot mechanically increase volume; reserve membership cannot
    # reduce it for otherwise identical histories. No salary enters this fit.
    for feature in ("older", "older_volume"):
        upper[1 + FEATURES.index(feature)] = 0.0
    for feature in ("rostered", "rostered_peak", "quality", "quality_volume"):
        lower[1 + FEATURES.index(feature)] = 0.0
    result = lsq_linear(np.vstack([x, regularizer]), np.r_[rows["actual"], np.zeros(x.shape[1])],
                        bounds=(lower, upper))
    if not result.success:
        raise RuntimeError("Workload fit did not converge")
    return result.x


def fit(frame: pd.DataFrame, last_target_season: int) -> dict:
    """Choose shrinkage with forward one-year folds, then fit direct horizons.

    Long-horizon fits share the role's one-year shrinkage. They never use outcomes
    after the forecast origin to select parameters or fit a historical replay.
    """
    training = frame[(frame["target_season"] <= last_target_season) & (frame["origin"] >= FIRST_ORIGIN)]
    models, penalties = {}, {}
    for role in ("bat", "pit"):
        one = training[training["role"].eq(role) & training["horizon"].eq(1)]
        folds = sorted(one["origin"].unique())[3:][-3:]
        if not folds:
            raise ValueError("Insufficient chronological workload training seasons")
        errors = []
        for penalty in PENALTIES:
            sse, n = 0.0, 0
            for origin in folds:
                train, held = one[one["target_season"] < origin], one[one["origin"] == origin]
                beta = _ridge(train, float(penalty))
                pred = np.maximum(0, beta[0] + design(held, held["age"].values) @ beta[1:])
                sse += float(np.sum((pred-held["actual"])**2))
                n += len(held)
            errors.append(sse/n)
        penalty = float(PENALTIES[np.argmin(errors)])
        penalties[role] = {"penalty": penalty, "forward_origins": [int(x) for x in folds]}
        for horizon in sorted(training["horizon"].unique()):
            rows = training[training["role"].eq(role) & training["horizon"].eq(horizon)]
            if len(rows) < 100:
                continue
            models.setdefault(str(int(horizon)), {})[role] = {
                "coefficients": _ridge(rows, penalty).tolist(), "n": len(rows),
                "origins": int(rows["origin"].nunique()), "max_volume": float(rows["actual"].max())}
    return {"method": "direct_horizon_v1", "last_target_season": last_target_season,
            "features": list(FEATURES), "models": models, "regularization": penalties,
            "roster_date": "October 1 before forecast origin", "short_season": "2020 normalized as input; excluded as target"}


def predict(frame: pd.DataFrame, ages: pd.Series, fitted: dict, first_season: int, last_season: int) -> pd.DataFrame:
    if first_season <= fitted["last_target_season"]:
        raise ValueError("Workload forecast overlaps its training targets")
    if fitted.get("method") != "direct_horizon_v1" or fitted["features"] != list(FEATURES):
        raise ValueError("Refit the direct-horizon workload model before scoring")
    age = frame["player_id"].map(ages).to_numpy(float)
    if not np.isfinite(age).all():
        raise ValueError("Workload forecasts require player ages")
    if not frame["role"].isin(("bat", "pit")).all():
        raise ValueError("Unsupported workload role")
    x = design(frame, age)
    rows = []
    for year in range(first_season, last_season + 1):
        horizon = str(year-first_season+1)
        if horizon not in fitted["models"]:
            raise ValueError(f"Unsupported workload horizon {horizon}; no silent extrapolation")
        pa = np.empty(len(frame))
        for role in ("bat", "pit"):
            member = frame["role"].eq(role).to_numpy()
            model = fitted["models"][horizon][role]
            beta = np.asarray(model["coefficients"])
            pa[member] = np.clip(beta[0] + x[member] @ beta[1:], 0, model["max_volume"])
        pa[frame[list(LAGS)].sum(axis=1).eq(0).to_numpy()] = 0.0
        rows.append(frame[["player_id", "role"]].assign(season=year, pa=pa))
    return pd.concat(rows, ignore_index=True)
