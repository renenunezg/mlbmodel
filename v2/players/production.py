"""Forecast total FanGraphs WAR, independently of the descriptive grade.

Targets include players who leave MLB (zero WAR), rather than only survivors.
Each future year is fitted directly from information at the forecast origin.
Source WAR is a current historical revision, not an archived as-of snapshot.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.linalg import solve
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from xgboost import Booster, XGBRegressor

from v2.players import bio, workload

FIRST_SEASON = 1950
FIRST_ORIGIN = 1955
RECENT_FIRST_ORIGIN = 1990
LONG_HISTORY_HORIZON = 9
MAX_HORIZON = 15
PENALTIES = np.logspace(-2, 4, 25)
KEYS = ["player_id", "role"]
FEATURES = [*(f"war{i}" for i in range(1, 6)), *(f"volume{i}" for i in range(1, 6)),
            "age", "age_squared", "quality_age", "quality_age_squared",
            "skill1", "skill2", "skill3", "recent_skill_volume"]
SOURCE = "https://statsapi.mlb.com/api/v1/stats"
METHOD = "direct_total_fwar_v1"
SHORT_SEASONS = {1972: 155, 1981: 107, 1994: 114, 1995: 144, 2020: 60}
PITCHER_INPUTS = ("k", "bb", "hr", "starts")
PITCHER_FEATURES = [f"{name}{lag}" for name in PITCHER_INPUTS for lag in range(1, 4)]
ROSTER_FEATURES = ["roster_known", "rostered", "injured"]
SURGERY_FEATURES = ["tj_years", "tj_count"]
AAA_FEATURES = ["aaa_known", "aaa_volume", "aaa_k", "aaa_bb", "aaa_hr", "aaa_quality"]


def source_paths(last_season: int) -> list[Path]:
    return [bio.CACHE_DIR / f"fangraphs_war_mlb_{year}.json"
            for year in range(FIRST_SEASON, last_season+1)]


def parse_war(payload: dict, season: int) -> pd.DataFrame:
    """MLB distributes fWAR as sabermetrics.war; ra9War is a different metric.

    Request league player totals, not sabermetricsMultiTeam stints. Reject
    pagination, duplicates, missing outcomes and wrong seasons before caching.
    """
    parts = []
    blocks = payload.get("stats", [])
    if sorted(b.get("group", {}).get("displayName", "") for b in blocks) != ["hitting", "pitching"]:
        raise ValueError(f"Incomplete fWAR groups for {season}")
    for block in blocks:
        group = block["group"]["displayName"]
        splits = block["splits"]
        if block.get("type", {}).get("displayName") != "sabermetrics" or not splits or len(splits) != block["totalSplits"]:
            raise ValueError(f"Incomplete fWAR {group} season {season}")
        if any(int(r["season"]) != season or r.get("gameType", "R") != "R"
               or r["sport"]["id"] != 1 for r in splits):
            raise ValueError(f"Wrong fWAR season or competition for {season}")
        frame = pd.DataFrame([{"player_id": r["player"]["id"], "season": season,
                               "war": r["stat"].get("war")} for r in splits])
        if frame.player_id.duplicated().any() or not np.isfinite(frame.to_numpy(float)).all():
            raise ValueError(f"Duplicate identity or missing fWAR outcome for {season}")
        parts.append(frame.assign(role="bat" if group == "hitting" else "pit"))
    return pd.concat(parts, ignore_index=True)


def load_war(last_season: int | None = None, *, refresh: bool = False) -> pd.DataFrame:
    """Cache the original MLB fWAR response for each season, without bWAR fallback."""
    last_season = last_season or date.today().year
    parts = []
    for season, path in zip(range(FIRST_SEASON, last_season+1), source_paths(last_season)):
        if path.exists() and not (refresh and season == last_season):
            payload = json.loads(path.read_text())
        else:
            response = requests.get(SOURCE, params={"stats": "sabermetrics", "group": "hitting,pitching",
                "season": season, "sportIds": 1, "limit": 10000, "playerPool": "ALL", "gameType": "R"}, timeout=60)
            response.raise_for_status()
            payload = response.json()
            parse_war(payload, season)
            path.parent.mkdir(parents=True, exist_ok=True)
            pending = path.with_suffix(".tmp")
            pending.write_text(json.dumps(payload))
            pending.replace(path)
        parts.append(parse_war(payload, season))
    return pd.concat(parts, ignore_index=True)


def load_lines(last_season: int) -> pd.DataFrame:
    war = load_war(last_season)
    seasons = list(range(FIRST_SEASON, last_season+1))
    parts = []
    for group, role, column, unit in (("hitting", "bat", "plateAppearances", 600),
                                      ("pitching", "pit", "outs", 540)):
        stats = bio.league_season_lines(seasons, group)
        f = stats[["player_id", "season", "age", column]].assign(role=role)
        f = f.merge(war[war.role.eq(role)], on=KEYS+["season"], how="outer", validate="one_to_one", indicator=True)
        missing = f["_merge"].ne("both")
        eligible = f.groupby(KEYS)[column].transform("max") >= unit/6
        if (missing & (eligible | f["_merge"].eq("right_only"))).any():
            raise ValueError(f"fWAR and official MLB {group} populations differ for eligible players")
        # A player never reaching eligibility cannot contribute a history or
        # target. Do not invent WAR for his source-missing cameo appearances.
        f = f[~missing].copy()
        f["volume"] = f[column] / unit
        rates = stats[["player_id", "season"]].copy()
        for name, stat in (("k", "strikeOuts"), ("bb", "baseOnBalls"), ("hr", "homeRuns")):
            if role == "pit":
                league = stats.groupby("season")[[stat, "battersFaced"]].sum()
                centre = league[stat] / league["battersFaced"]
                rates[name] = 10 * (stats[stat] - stats.battersFaced * stats.season.map(centre)) / (stats.battersFaced + 100)
            else:
                rates[name] = 0.0
        rates["starts"] = stats.gamesStarted / stats.gamesPlayed.clip(lower=1) if role == "pit" else 0.0
        f = f.merge(rates, on=["player_id", "season"], validate="one_to_one")
        parts.append(f[KEYS+["season", "age", "volume", "war", *PITCHER_INPUTS]])
    f = pd.concat(parts, ignore_index=True)
    skill = workload.load_lines(seasons)[KEYS+["season", "quality"]]
    f = f.merge(skill, on=KEYS+["season"], how="left", validate="one_to_one")
    if not np.isfinite(f[["age", "volume", "war", "quality"]].to_numpy()).all():
        raise ValueError("Missing official MLB performance inputs for fWAR history")
    f["skill"] = f.pop("quality") / 20
    return f


def history(lines: pd.DataFrame, origin: int, *, include_roster: bool = False,
            include_surgery: bool = False, include_aaa: bool = False) -> pd.DataFrame:
    if not set(range(origin-5, origin)) <= set(lines["season"]):
        raise ValueError("WAR forecast requires five complete historical seasons")
    prior = lines[lines["season"].between(origin-5, origin-1)].copy()
    for year, games in SHORT_SEASONS.items():
        prior.loc[prior["season"].eq(year), ["war", "volume"]] *= 162 / games
    last = prior.sort_values("season").drop_duplicates(KEYS, keep="last")
    h = last[KEYS].copy()
    h["age"] = last["age"] + origin - last["season"]
    for lag in range(1, 6):
        columns = ["war", "volume", "skill", *([c for c in PITCHER_INPUTS if c in prior] if lag <= 3 else [])]
        row = prior[prior["season"].eq(origin-lag)][KEYS+columns]
        h = h.merge(row.rename(columns={c: f"{c}{lag}" for c in columns}),
                    on=KEYS, how="left", validate="one_to_one")
    h = h.fillna(0)
    if include_roster:
        h[ROSTER_FEATURES] = 0.0
        if origin >= 2010:
            roster = bio.october_roster(origin-1)
            injured = set(roster.loc[roster.status.str.startswith("D"), "player_id"])
            h["roster_known"] = 1.0
            h["rostered"] = h.player_id.isin(roster.player_id).astype(float)
            h["injured"] = h.player_id.isin(injured).astype(float)
    if include_surgery:
        surgery = bio.surgery_dates()
        cutoff = pd.Timestamp(origin-1, 11, 1)
        past = surgery[surgery.date.lt(cutoff)].groupby("player_id").date.agg(["max", "count"])
        past["tj_years"] = ((cutoff-past["max"]).dt.days/365.25).clip(upper=5)
        h["tj_years"] = h.player_id.map(past.tj_years).fillna(5.0)
        h["tj_count"] = h.player_id.map(past["count"]).fillna(0.0)
    if include_aaa:
        aaa = aaa_hitting(origin-1)
        h = h.merge(aaa, on=KEYS, how="left", validate="one_to_one")
        h["aaa_known"] = float(origin >= 2007 and origin != 2021)
        h[AAA_FEATURES] = h[AAA_FEATURES].fillna(0.0)
    return h[h[[f"volume{i}" for i in range(1, 6)]].max(axis=1) >= 1/6].reset_index(drop=True)


def panel(lines: pd.DataFrame, last_target: int) -> pd.DataFrame:
    rows = []
    for origin in range(FIRST_ORIGIN, last_target+1):
        h = history(lines, origin, include_roster=True, include_surgery=True, include_aaa=True)
        for horizon in range(1, min(MAX_HORIZON, last_target-origin+1)+1):
            target = origin+horizon-1
            if target in SHORT_SEASONS:
                continue
            actual = lines[lines["season"].eq(target)][KEYS+["war", "volume"]].rename(columns={"volume": "actual_volume"})
            row = h.merge(actual, on=KEYS, how="left", validate="one_to_one").fillna({"war": 0, "actual_volume": 0})
            rows.append(row.assign(origin=origin, target=target, horizon=horizon))
    return pd.concat(rows, ignore_index=True)


def design(h: pd.DataFrame) -> np.ndarray:
    w = h[[f"war{i}" for i in range(1, 6)]].to_numpy(float)
    v = h[[f"volume{i}" for i in range(1, 6)]].to_numpy(float)
    age = (h["age"].to_numpy(float)-29)/10
    quality = w[:, :3].mean(axis=1)
    return np.column_stack([np.ones(len(h)), w, v, age, age**2, quality*age, quality*age**2,
                            h[["skill1", "skill2", "skill3"]].to_numpy(float), h["skill1"].to_numpy()*v[:, 0]])


def _fit(x: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    regularizer = np.eye(x.shape[1])*penalty
    regularizer[0, 0] = 0
    return solve(x.T@x+regularizer, x.T@y, assume_a="pos")


def _hitter_fit(rows: pd.DataFrame, penalty: float, *, include_aaa: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Retain exits as zero outcomes while estimating active-season production."""
    x = hitter_design(rows, include_aaa=include_aaa)
    active = rows.actual_volume.to_numpy() > 0
    availability = LogisticRegression(C=1/penalty, max_iter=1000, solver="lbfgs").fit(x[:, 1:], active)
    beta = _fit(x[active], rows.war.to_numpy()[active], penalty)
    return beta, np.r_[availability.intercept_, availability.coef_[0]]


def hitter_design(rows: pd.DataFrame, *, include_aaa: bool = False) -> np.ndarray:
    names = ROSTER_FEATURES + (AAA_FEATURES if include_aaa else [])
    x = np.column_stack([design(rows), rows[names].to_numpy(float)])
    if not np.isfinite(x).all():
        raise ValueError("Missing hitter roster inputs")
    return x


def aaa_hitting_rows(season: int) -> pd.DataFrame:
    """One row per Triple-A hitter-season with the official MLB stat columns, cached per season.

    The cancelled 2020 season and years before source coverage raise.
    """
    if season < 2006 or season == 2020:
        raise ValueError(f"No Triple-A coverage for {season}")
    path = bio.CACHE_DIR / f"aaa_season_{season}.json"
    if path.exists():
        payload = json.loads(path.read_text())
    else:
        response = requests.get(SOURCE, params={"stats": "season", "group": "hitting,pitching", "sportIds": 11,
            "season": season, "limit": 10000, "playerPool": "ALL", "gameType": "R"}, timeout=60)
        response.raise_for_status()
        payload = response.json()
    blocks = [b for b in payload.get("stats", []) if b.get("group", {}).get("displayName") == "hitting"]
    if len(blocks) != 1 or blocks[0].get("type", {}).get("displayName") != "season":
        raise ValueError("Missing Triple-A hitting season")
    b = blocks[0]
    splits = b["splits"]
    if len(splits) != b["totalSplits"] or len(splits) < 300 or any(int(s["season"]) != season or s["sport"]["id"] != 11 for s in splits):
        raise ValueError("Incomplete or wrong Triple-A season")
    r = pd.DataFrame([{"player_id": s["player"]["id"], **s["stat"]} for s in splits])
    if r.player_id.duplicated().any():
        raise ValueError("Duplicate Triple-A player totals")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(".tmp")
        pending.write_text(json.dumps(payload))
        pending.replace(path)
    return r


def aaa_hitting(season: int) -> pd.DataFrame:
    """Last-season Triple-A performance, used only for existing MLB histories.

    Rates are centered on that Triple-A season and stabilized with 100 PA.
    These are performance inputs, not invented minor-league WAR. The cancelled
    2020 season and years before source coverage are explicitly unknown.
    """
    if season < 2006 or season == 2020:
        return pd.DataFrame(columns=KEYS+AAA_FEATURES[1:])
    r = aaa_hitting_rows(season)
    pa = r.plateAppearances
    out = r[["player_id"]].assign(role="bat", aaa_volume=pa/600)
    for name, stat in (("k", "strikeOuts"), ("bb", "baseOnBalls"), ("hr", "homeRuns")):
        out["aaa_"+name] = 10*(r[stat]-pa*r[stat].sum()/pa.sum())/(pa+100)
    points = .7*r.baseOnBalls+.9*r.hits+.35*r.doubles+.65*r.triples+1.1*r.homeRuns
    out["aaa_quality"] = 10*(points-pa*points.sum()/pa.sum())/(pa+100)
    if not np.isfinite(out[AAA_FEATURES[1:]].to_numpy(float)).all():
        raise ValueError("Missing Triple-A production inputs")
    return out


def fit_aaa(frame: pd.DataFrame, last_target: int) -> dict:
    """An additional batting model only where prior-year Triple-A data exists."""
    training = frame[frame.target.le(last_target) & frame.role.eq("bat")
                     & frame.origin.ge(RECENT_FIRST_ORIGIN) & frame.horizon.lt(LONG_HISTORY_HORIZON)]
    models = {}
    for horizon, rows in training.groupby("horizon"):
        folds = sorted(rows.origin.unique())[-3:]
        errors = np.zeros(len(PENALTIES))
        for origin in folds:
            train, held = rows[rows.target.lt(origin)], rows[rows.origin.eq(origin)]
            if train.origin.nunique() < 5:
                continue
            x = hitter_design(held, include_aaa=True)
            for i, penalty in enumerate(PENALTIES):
                beta, availability = _hitter_fit(train, penalty, include_aaa=True)
                prediction = np.maximum(0, (x@beta)*expit(x@availability))
                errors[i] += float(np.sum((prediction-held.war.to_numpy())**2))
        if not errors.any() or not rows.aaa_volume.gt(0).any():
            continue
        penalty = float(PENALTIES[np.argmin(errors)])
        beta, availability = _hitter_fit(rows, penalty, include_aaa=True)
        models[str(int(horizon))] = {"coefficients": beta.tolist(), "availability_coefficients": availability.tolist(),
                                     "penalty": penalty, "n": len(rows)}
    return {"features": AAA_FEATURES, "models": models,
            "scope": "Only hitters with observed Triple-A PA in the season preceding the forecast; MLB eligibility unchanged",
            "source_hashes": {f"aaa_season_{year}.json": hashlib.sha256((bio.CACHE_DIR/f"aaa_season_{year}.json").read_bytes()).hexdigest()
                              for year in range(2006, last_target) if year != 2020}}


def pitcher_design(rows: pd.DataFrame, *, include_surgery: bool = False) -> np.ndarray:
    names = PITCHER_FEATURES + (SURGERY_FEATURES if include_surgery else [])
    x = np.column_stack([design(rows)[:, 1:], rows[names].to_numpy(float)])
    if not np.isfinite(x).all():
        raise ValueError("Missing pitcher production inputs")
    return x


def _pitcher_estimator(depth: int, trees: int) -> XGBRegressor:
    return XGBRegressor(n_estimators=trees, max_depth=depth, learning_rate=.05,
                        min_child_weight=50, n_jobs=1, objective="reg:squarederror", tree_method="hist")


def _pitcher_parameters(rows: pd.DataFrame, *, include_surgery: bool = False) -> dict:
    """Select complexity only on forward folds that precede the forecast origin."""
    folds = sorted(rows.origin.unique())[-3:]
    errors = {depth: np.zeros(50) for depth in (2, 3, 4)}
    for origin in folds:
        train, held = rows[rows.target < origin], rows[rows.origin.eq(origin)]
        if train.origin.nunique() < 5:
            raise ValueError("Insufficient forward pitcher training history")
        x = pitcher_design(train, include_surgery=include_surgery)
        xt = pitcher_design(held, include_surgery=include_surgery)
        for depth in errors:
            model = _pitcher_estimator(depth, 500).fit(x, train.war)
            for i, count in enumerate(range(10, 501, 10)):
                pred = np.maximum(0, model.predict(xt, iteration_range=(0, count)))
                errors[depth][i] += float(np.sum((pred-held.war.to_numpy())**2))
    depth, index = min(((d, int(np.argmin(e))) for d, e in errors.items()), key=lambda item: errors[item[0]][item[1]])
    return {"depth": depth, "trees": (index+1)*10, "forward_origins": [int(o) for o in folds]}


def fit(frame: pd.DataFrame, last_target: int) -> dict:
    training = frame[frame["target"] <= last_target]
    models = {}
    recent_pitchers = training[training.role.eq("pit") & training.horizon.eq(1) & training.origin.ge(RECENT_FIRST_ORIGIN)]
    pitcher_parameters = _pitcher_parameters(recent_pitchers)
    surgery_parameters = _pitcher_parameters(recent_pitchers, include_surgery=True)
    for role in ("bat", "pit"):
        for horizon in range(1, MAX_HORIZON+1):
            rows = training[training["role"].eq(role) & training["horizon"].eq(horizon)]
            # The validated near-term population stays fixed. Decade-long
            # forecasts need older completed careers for honest forward folds.
            if horizon < LONG_HISTORY_HORIZON:
                rows = rows[rows.origin.ge(RECENT_FIRST_ORIGIN)]
            if role == "pit":
                if rows.origin.nunique() < 5:
                    continue
                # Known surgery timing improved near-term injury-return forecasts.
                # The long-horizon experiment did not improve, so keep its baseline.
                with_surgery = horizon == 1
                parameters = surgery_parameters if with_surgery else pitcher_parameters
                model = _pitcher_estimator(parameters["depth"], parameters["trees"]).fit(
                    pitcher_design(rows, include_surgery=with_surgery), rows.war)
                models.setdefault(str(horizon), {})[role] = {
                    "booster_json": model.get_booster().save_raw(raw_format="json").decode(),
                    "includes_surgery": with_surgery,
                    "n": len(rows), "origins": int(rows.origin.nunique())}
                continue
            # Last three fully observed origins; fitting for each fold stops BEFORE
            # its origin, not before its later outcome year.
            folds = sorted(rows["origin"].unique())[-3:]
            errors = np.zeros(len(PENALTIES))
            for origin in folds:
                train = rows[rows["target"] < origin]
                held = rows[rows["origin"] == origin]
                if train["origin"].nunique() < 5:
                    continue
                xt = hitter_design(held)
                for i, penalty in enumerate(PENALTIES):
                    beta, availability = _hitter_fit(train, penalty)
                    prediction = np.maximum(0, (xt @ beta) * expit(xt @ availability))
                    errors[i] += float(np.sum((prediction-held["war"])**2))
            if not errors.any() or rows["origin"].nunique() < 5:
                continue
            penalty = float(PENALTIES[np.argmin(errors)])
            beta, availability = _hitter_fit(rows, penalty)
            models.setdefault(str(horizon), {})[role] = {
                "coefficients": beta.tolist(), "availability_coefficients": availability.tolist(), "penalty": penalty, "n": len(rows),
                "origins": int(rows["origin"].nunique()), "forward_origins": [int(x) for x in folds]}
    return {"method": METHOD, "target": "fWAR", "last_target_season": last_target,
            "aaa": fit_aaa(training, last_target),
            "pitcher_features": PITCHER_FEATURES, "pitcher_parameters": pitcher_parameters,
            "hitter_roster_features": ROSTER_FEATURES,
            "surgery_features": SURGERY_FEATURES, "surgery_parameters": surgery_parameters,
            "surgery_source": bio.SURGERY_SOURCE,
            "surgery_scope": "Only operations before November 1 at the forecast origin; no return dates; year-one pitching only",
            "roster_date": "October 1 before the forecast year; earlier than 2010 explicitly unknown",
            "history_first_season": FIRST_SEASON,
            "recent_first_origin": RECENT_FIRST_ORIGIN, "long_history_horizon": LONG_HISTORY_HORIZON,
            "features": FEATURES, "models": models, "source": SOURCE, "source_stat": "sabermetrics.war",
            "source_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in [*source_paths(last_target), bio.CACHE_DIR / "tommy_john_surgeries.csv",
                                  *[bio.CACHE_DIR / f"roster_oct1_{year}.parquet" for year in range(2009, last_target)]]},
            "population": "100 PA or 30 IP in any prior five years; absent future players are zero targets",
            "short_seasons": "1972, 1981, 1994, 1995, 2020 scaled to 162 as inputs and excluded as targets; earlier standard 154-game seasons retained as observed",
            "source_revision": "Current historical WAR revisions, not archived real-time WAR"}


def predict(h: pd.DataFrame, fitted: dict, first: int, last: int) -> pd.DataFrame:
    if fitted.get("method") != METHOD or fitted.get("features") != FEATURES:
        raise ValueError("Missing or incompatible WAR production model")
    if first <= fitted["last_target_season"]:
        raise ValueError("WAR forecast overlaps training targets")
    if last < first or not np.isfinite(design(h)).all():
        raise ValueError("Invalid WAR horizon or history")
    rows = []
    for year in range(first, last+1):
        horizon = str(year-first+1)
        for role, members in h.groupby("role"):
            model = fitted["models"].get(horizon, {}).get(role)
            if model is None:
                raise ValueError(f"Unsupported WAR horizon {horizon} for {role}")
            if "booster_json" in model:
                if fitted.get("pitcher_features") != PITCHER_FEATURES:
                    raise ValueError("Incompatible pitcher production features")
                booster = Booster(params={"nthread": 1})
                booster.load_model(bytearray(model["booster_json"], "utf-8"))
                with_surgery = model.get("includes_surgery", False)
                if with_surgery and fitted.get("surgery_features") != SURGERY_FEATURES:
                    raise ValueError("Incompatible surgery production features")
                raw = booster.inplace_predict(pitcher_design(members, include_surgery=with_surgery))
            else:
                x = design(members)
                if role == "bat" and "hitter_roster_features" in fitted:
                    if fitted["hitter_roster_features"] != ROSTER_FEATURES:
                        raise ValueError("Incompatible hitter roster features")
                    x = hitter_design(members)
                raw = x @ np.asarray(model["coefficients"])
                if "availability_coefficients" in model:
                    raw *= expit(x @ np.asarray(model["availability_coefficients"]))
                if role == "bat" and "aaa" in fitted:
                    aaa = fitted["aaa"]
                    if aaa.get("features") != AAA_FEATURES:
                        raise ValueError("Incompatible Triple-A production features")
                    observed = members.aaa_volume.to_numpy() > 0
                    extra = aaa["models"].get(horizon)
                    if extra is not None and observed.any():
                        xx = hitter_design(members[observed], include_aaa=True)
                        raw[observed] = (xx@np.asarray(extra["coefficients"]))*expit(xx@np.asarray(extra["availability_coefficients"]))
            rows.append(members[KEYS].assign(season=year, expected_war=raw))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=KEYS+["season", "expected_war"])


def add_forecasts(playing_time: pd.DataFrame, fitted: dict, first: int, last: int) -> pd.DataFrame:
    h = history(load_lines(first-1), first, include_roster="hitter_roster_features" in fitted,
                include_surgery="surgery_features" in fitted, include_aaa="aaa" in fitted)
    forecast = predict(h, fitted, first, last)
    forecast = forecast.merge(h[KEYS].assign(pricing_role=pricing_roles(h)), on=KEYS, validate="many_to_one")
    # Missing history remains unknown, never a replacement-level dollar estimate.
    return playing_time.merge(forecast, on=KEYS+["season"], how="left", validate="one_to_one")


def pricing_roles(h: pd.DataFrame) -> pd.Series:
    """Latest observed pitching role, retaining it through a missed season."""
    starts = pd.Series(np.nan, index=h.index)
    for lag in range(1, 4):
        starts = starts.fillna(h[f"starts{lag}"].where(h[f"volume{lag}"] > 0))
    return pd.Series(np.where(h.role.eq("bat"), "bat", np.where(starts.ge(.5), "SP", "RP")), index=h.index)


def actual_war(season: int) -> pd.Series:
    f = load_war(season)
    return f[f["season"].eq(season)].groupby("player_id")["war"].sum()
