"""Age curves by the delta method, fitted once per offseason.

Each curve is cumulative change in a component from age 27, built from
consecutive-season pairs of the same player weighted by the harmonic mean of
their playing time, then smoothed. Only seasons at or before the fit's last
development season enter, so the curves never see a holdout.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from v2.players import bio

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "players"
ANCHOR_AGE = 27
AGE_RANGE = (20, 42)
FIRST_LINE_SEASON = 2008
SMOOTH_WINDOW = 3
MIN_PAIRS_PER_AGE = 20
# Linear weights for the stat-line components (Statcast-era averages); the
# curve's shape, not its level, is what matters here.
LINE_WEIGHTS = {"K": -0.23, "BB": 0.23, "HBP": 0.36, "1B": 0.44, "2B": 0.72, "3B": 0.97, "HR": 1.52, "OUT": -0.25}
FIP_WEIGHTS = {"HR": 13.0, "BB": 3.0, "K": -2.0}
PA_PER_IP = 4.37
STARTER_INNINGS = 180.0
COMPONENTS = ("bat_runs_600", "k_rate", "bb_rate", "iso", "hr_rate", "pitch_runs_180", "pitch_k_rate", "pitch_bb_rate",
              "oaa_season", "sprint_speed", "velo")


def _total_rows(lines: pd.DataFrame, volume: str) -> pd.DataFrame:
    """A traded player has a row per team and a total; keep the row with the most volume."""
    lines = lines.copy()
    lines[volume] = pd.to_numeric(lines[volume], errors="coerce").fillna(0)
    return lines.sort_values(volume, ascending=False).drop_duplicates(["player_id", "season"])


def hitting_components(lines: pd.DataFrame) -> pd.DataFrame:
    """Per player-season: age, pa, and the hitting components."""
    h = _total_rows(lines, "plateAppearances")
    num = {c: pd.to_numeric(h[c], errors="coerce").fillna(0) for c in
           ("plateAppearances", "strikeOuts", "baseOnBalls", "hitByPitch", "hits", "doubles", "triples", "homeRuns", "atBats", "totalBases")}
    pa = num["plateAppearances"]
    singles = num["hits"] - num["doubles"] - num["triples"] - num["homeRuns"]
    outs = pa - num["strikeOuts"] - num["baseOnBalls"] - num["hitByPitch"] - num["hits"]
    runs = (LINE_WEIGHTS["K"] * num["strikeOuts"] + LINE_WEIGHTS["BB"] * num["baseOnBalls"] + LINE_WEIGHTS["HBP"] * num["hitByPitch"]
            + LINE_WEIGHTS["1B"] * singles + LINE_WEIGHTS["2B"] * num["doubles"] + LINE_WEIGHTS["3B"] * num["triples"]
            + LINE_WEIGHTS["HR"] * num["homeRuns"] + LINE_WEIGHTS["OUT"] * outs)
    out = pd.DataFrame({
        "player_id": h["player_id"].values, "season": h["season"].values, "age": pd.to_numeric(h["age"], errors="coerce").values,
        "volume": pa.values,
        "bat_runs_600": (runs / pa.where(pa > 0) * 600).values,
        "k_rate": (num["strikeOuts"] / pa.where(pa > 0)).values,
        "bb_rate": (num["baseOnBalls"] / pa.where(pa > 0)).values,
        "iso": ((num["totalBases"] - num["hits"]) / num["atBats"].where(num["atBats"] > 0)).values,
        "hr_rate": (num["homeRuns"] / pa.where(pa > 0)).values,
    })
    return out[out["volume"] > 0]


def pitching_components(lines: pd.DataFrame) -> pd.DataFrame:
    p = _total_rows(lines, "battersFaced")
    num = {c: pd.to_numeric(p[c], errors="coerce").fillna(0) for c in ("battersFaced", "strikeOuts", "baseOnBalls", "homeRuns")}
    bf = num["battersFaced"]
    fip_runs = (FIP_WEIGHTS["HR"] * num["homeRuns"] + FIP_WEIGHTS["BB"] * num["baseOnBalls"] + FIP_WEIGHTS["K"] * num["strikeOuts"])
    out = pd.DataFrame({
        "player_id": p["player_id"].values, "season": p["season"].values, "age": pd.to_numeric(p["age"], errors="coerce").values,
        "volume": bf.values,
        # Runs prevented per 180 innings: FIP-style runs allowed per batter, negated.
        "pitch_runs_180": (-(fip_runs / bf.where(bf > 0)) * PA_PER_IP * STARTER_INNINGS / 9.0).values,
        "pitch_k_rate": (num["strikeOuts"] / bf.where(bf > 0)).values,
        "pitch_bb_rate": (num["baseOnBalls"] / bf.where(bf > 0)).values,
    })
    return out[out["volume"] > 0]


def delta_curve(frame: pd.DataFrame, component: str) -> dict[int, float]:
    """Cumulative change in ``component`` by age, anchored at zero at ANCHOR_AGE.

    ``frame`` has player_id, season, age, volume, and the component column.
    Consecutive seasons of a player are paired; each pair's change is weighted by the
    harmonic mean of the two volumes. Ages with too few pairs borrow the neighbours' mean.
    """
    f = frame.dropna(subset=["age", component]).sort_values(["player_id", "season"])
    f["age"] = f["age"].astype(int)
    nxt = f.groupby("player_id").shift(-1)
    pair = f[(nxt["season"] == f["season"] + 1)].copy()
    pair["delta"] = nxt.loc[pair.index, component] - pair[component]
    v1, v2 = pair["volume"], nxt.loc[pair.index, "volume"]
    pair["w"] = 2 * v1 * v2 / (v1 + v2).where((v1 + v2) > 0)
    pair = pair.dropna(subset=["delta", "w"])
    ages = range(AGE_RANGE[0], AGE_RANGE[1])
    if pair.empty:
        return {a: 0.0 for a in range(AGE_RANGE[0], AGE_RANGE[1] + 1)}
    by_age = pair.groupby("age").apply(lambda g: pd.Series({"delta": np.average(g["delta"], weights=g["w"]), "n": len(g)}), include_groups=False)
    deltas = pd.Series({a: by_age["delta"].get(a, np.nan) if by_age["n"].get(a, 0) >= MIN_PAIRS_PER_AGE else np.nan for a in ages})
    deltas = deltas.interpolate(limit_direction="both").fillna(0.0)
    smoothed = deltas.rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()
    # delta at age a is the change from a to a+1; cumulate from the anchor.
    curve: dict[int, float] = {ANCHOR_AGE: 0.0}
    for a in range(ANCHOR_AGE, AGE_RANGE[1]):
        curve[a + 1] = curve[a] + float(smoothed[a])
    for a in range(ANCHOR_AGE - 1, AGE_RANGE[0] - 1, -1):
        curve[a] = curve[a + 1] - float(smoothed[a])
    return dict(sorted(curve.items()))


def fit_curves(
    lines_hitting: pd.DataFrame, lines_pitching: pd.DataFrame, statcast_components: pd.DataFrame, max_season: int,
) -> dict[str, dict[int, float]]:
    """All component curves from seasons at or before ``max_season``.

    ``statcast_components`` has player_id, season, age, volume and any of oaa_season,
    sprint_speed, velo (the Statcast-era components, 2016 onward).
    """
    hit = hitting_components(lines_hitting[lines_hitting["season"].between(FIRST_LINE_SEASON, max_season)])
    pit = pitching_components(lines_pitching[lines_pitching["season"].between(FIRST_LINE_SEASON, max_season)])
    sc = statcast_components[statcast_components["season"] <= max_season] if not statcast_components.empty else statcast_components
    curves: dict[str, dict[int, float]] = {}
    for comp in ("bat_runs_600", "k_rate", "bb_rate", "iso", "hr_rate"):
        curves[comp] = delta_curve(hit, comp)
    for comp in ("pitch_runs_180", "pitch_k_rate", "pitch_bb_rate"):
        curves[comp] = delta_curve(pit, comp)
    for comp in ("oaa_season", "sprint_speed", "velo"):
        if comp in sc.columns:
            curves[comp] = delta_curve(sc.dropna(subset=[comp]), comp)
    return curves


def apply(value: float, component: str, age_from: int, age_to: int, curves: dict[str, dict[int, float]]) -> float:
    """Shift ``value`` from one age to another along the component's curve (clamped to the fitted ages)."""
    curve = curves[component]
    lo, hi = min(curve), max(curve)
    a, b = int(np.clip(age_from, lo, hi)), int(np.clip(age_to, lo, hi))
    return float(value + curve[b] - curve[a])


def statcast_components_from_cache(seasons: list[int], ages: pd.DataFrame) -> pd.DataFrame:
    """OAA, sprint speed, and fastball velocity per player-season from the inputs cache, with ages.

    ``ages`` has player_id, season, age (from the season lines).
    """
    from v2.players.inputs import season_inputs

    frames = []
    for season in seasons:
        try:
            f = season_inputs(season)
        except FileNotFoundError:
            continue
        wide = f[f["input"].isin(["range.oaa_per_600", "speed.sprint_speed", "stuff.velo"])]
        pivot = wide.pivot_table(index="player_id", columns="input", values="value").rename(columns={
            "range.oaa_per_600": "oaa_season", "speed.sprint_speed": "sprint_speed", "stuff.velo": "velo",
        })
        sample = wide.pivot_table(index="player_id", columns="input", values="sample").max(axis=1).rename("volume")
        pivot = pivot.join(sample).reset_index()
        pivot["season"] = season
        frames.append(pivot)
    if not frames:
        return pd.DataFrame(columns=["player_id", "season", "age", "volume"])
    out = pd.concat(frames, ignore_index=True).merge(ages, on=["player_id", "season"], how="left")
    return out


def fit_curves_from_cache(max_season: int) -> dict[str, dict[int, float]]:
    """Complete historical populations, including players retired before Statcast."""
    years = list(range(FIRST_LINE_SEASON, max_season + 1))
    hitting = bio.league_season_lines(years, "hitting")
    pitching = bio.league_season_lines(years, "pitching")
    ages = pd.concat([hitting[["player_id", "season", "age"]], pitching[["player_id", "season", "age"]]]).drop_duplicates(["player_id", "season"])
    seasons = sorted(int(p.stem.split("_")[1]) for p in CACHE_DIR.glob("inputs_*.parquet")
                     if int(p.stem.split("_")[1]) <= max_season)
    sc = statcast_components_from_cache(seasons, ages)
    return fit_curves(hitting, pitching, sc, max_season)


def _projection_design(frame: pd.DataFrame, ages: np.ndarray) -> np.ndarray:
    age = (np.asarray(ages, dtype=float) - 27) / 10
    quality = frame["quality"].to_numpy(float)
    return np.column_stack([np.ones(len(frame)), quality, age, quality*age])


def _projection_fit(rows: pd.DataFrame, penalty: float) -> np.ndarray:
    x = _projection_design(rows, rows["age"].to_numpy())
    weights = rows["actual"].to_numpy(float) / rows["actual"].mean()
    regularizer = np.eye(x.shape[1]) * penalty
    regularizer[0, 0] = 0
    return np.linalg.solve(x.T @ (weights[:, None]*x) + regularizer,
                           x.T @ (weights*rows["target_quality"].to_numpy(float)))


def fit_projection(panel: pd.DataFrame, last_target_season: int) -> dict:
    """Expected future rate weighted by the volume at which it is produced.

    E[volume * rate] / E[volume] is the rate that belongs beside an expected
    workload. Annual survivor deltas cannot be repeatedly subtracted from an
    already regressed skill forecast. Fit each horizon directly, then express
    its change relative to the same model's first-year rate prediction.
    """
    training = panel[(panel["target_season"] <= last_target_season) & panel["actual"].gt(0)
                     & panel["target_quality"].notna()]
    models, regularization = {}, {}
    for role in ("bat", "pit"):
        one = training[training["role"].eq(role) & training["horizon"].eq(1)]
        folds = sorted(one["origin"].unique())[3:][-3:]
        if not folds:
            raise ValueError("Insufficient chronological production-rate history")
        penalties = np.logspace(-3, 3, 25)
        errors = []
        for penalty in penalties:
            sse, volume = 0.0, 0.0
            for origin in folds:
                train, held = one[one["target_season"] < origin], one[one["origin"] == origin]
                beta = _projection_fit(train, float(penalty))
                error = _projection_design(held, held["age"].to_numpy()) @ beta - held["target_quality"].to_numpy()
                sse += float(np.sum(held["actual"].to_numpy() * error**2))
                volume += float(held["actual"].sum())
            errors.append(sse/volume)
        penalty = float(penalties[np.argmin(errors)])
        regularization[role] = {"penalty": penalty, "forward_origins": [int(x) for x in folds]}
        for horizon, rows in training[training["role"].eq(role)].groupby("horizon"):
            if len(rows) < 8:
                continue
            models.setdefault(str(int(horizon)), {})[role] = {
                "coefficients": _projection_fit(rows, penalty).tolist(), "n": len(rows),
                "origins": int(rows["origin"].nunique())}
    return {"method": "direct_volume_weighted_rate_v1", "last_target_season": last_target_season,
            "models": models, "regularization": regularization,
            "units": {"bat": "runs per 600 PA", "pit": "runs per 180 IP"}}


def add_projection_shifts(forecast: pd.DataFrame, history: pd.DataFrame, ages: pd.Series,
                          fitted: dict, first_season: int) -> pd.DataFrame:
    """Add rate changes without treating a future forecast as observed performance."""
    if not fitted or fitted.get("method") != "direct_volume_weighted_rate_v1":
        raise ValueError("Refit and validate the production aging model before scoring")
    if fitted["last_target_season"] >= first_season:
        raise ValueError("Production aging forecast overlaps training targets")
    out = forecast.merge(history[["player_id", "role", "quality"]], on=["player_id", "role"], how="left", validate="many_to_one")
    out["run_shift"] = np.nan
    out["run_rate"] = np.nan
    for (season, role), rows in out.groupby(["season", "role"]):
        horizon = str(int(season-first_season+1))
        if horizon not in fitted["models"] or role not in fitted["models"][horizon]:
            raise ValueError(f"Unsupported production aging horizon {horizon} for {role}")
        x = _projection_design(rows, rows["player_id"].map(ages).to_numpy())
        current = np.asarray(fitted["models"]["1"][role]["coefficients"])
        future = np.asarray(fitted["models"][horizon][role]["coefficients"])
        out.loc[rows.index, "run_shift"] = x @ (future-current)
        out.loc[rows.index, "run_rate"] = x @ future
    if not np.isfinite(out["run_shift"]).all():
        raise ValueError("Missing production aging projection")
    return out.drop(columns="quality")
