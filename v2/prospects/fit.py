"""Fit the prospect model on matured cohorts, evaluate it once on the holdout, and store it.

Per pool (hitters, pitchers), gradient-boosted trees predict from each snapshot's features:
reaching the majors, becoming a regular, peak WAR, control-years WAR with its quartiles, and years
to arrival. Rounds are chosen by early stopping on the last two development cohorts, then every
booster is refit on all development cohorts with that round count. A KATOH-style logistic and
linear baseline (age relative to level, strikeout, walk and power rates, level) is fitted on the
same rows. The holdout cohorts are scored once; a pool passes the publish gate when it beats the
baseline on both log loss for ``regular`` and Spearman for ``peak_war``.

Usage:
    MLBMODEL_DB_WRITES=1 python -m v2.prospects.fit --last-dev-season 2017 --holdout 2018 2019 --write-db
    python -m v2.prospects.fit --last-dev-season 2017 --holdout 2018 2019
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from backend.log import setup_logging
from v2.prospects import dataset, features, identity, lists, targets
from v2.prospects.feeds import CACHE_DIR
from v2.prospects.model import POOLS, QUANTILES, ProspectModel, model_path, publish_model

log = logging.getLogger(__name__)

FIRST_DEV_SEASON = 2005
VALIDATION_COHORTS = 2
EVALUATION_PATH = CACHE_DIR / "evaluation.json"
PARAMS = {"max_depth": 4, "eta": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 25, "lambda": 2.0}
MAX_ROUNDS = 3000
TWEEDIE_POWER = 1.5
HOLDOUT_TOUCHES = 2  # 2026-10-22: a second pass after the development-set experiments (new call-code and trend features, Tweedie)
EARLY_STOP = 100
MONOTONE = {
    "bat": {"age_rel": -1, "k_rate_rel": -1, "iso_rel": 1, "bb_rate_rel": 1, "top_level_rank": 1},
    "pit": {"age_rel": -1, "k_rate_rel": 1, "bb_rate_rel": -1, "top_level_rank": 1},
}
BASELINE_FEATURES = {
    "bat": ["age_rel", "k_rate", "bb_rate", "iso", "top_level_rank"],
    "pit": ["age_rel", "k_rate", "bb_rate", "hr_rate", "top_level_rank", "start_share"],
}
GRADE_BINS = [(20, 35), (35, 45), (45, 50), (50, 55), (55, 60), (60, 81)]
TOP_SHARE = 0.1


def _feature_names(pool: str) -> list[str]:
    return features.hitter_feature_names() if pool == "bat" else features.pitcher_feature_names()


def _dmatrix(frame: pd.DataFrame, names: list[str], y=None) -> xgb.DMatrix:
    X = frame.reindex(columns=names).astype(float)
    return xgb.DMatrix(X, label=y, feature_names=names, missing=np.nan)


def _params(pool: str, names: list[str], objective: str, **extra) -> dict:
    mono = "(" + ",".join(str(MONOTONE[pool].get(n, 0)) for n in names) + ")"
    return {**PARAMS, "objective": objective, "monotone_constraints": mono, "verbosity": 0, **extra}


def _fit_target(pool: str, names: list[str], dev: pd.DataFrame, val: pd.DataFrame, target: str) -> tuple[xgb.Booster, int]:
    """Early-stopped rounds on the validation cohorts, then a refit on dev plus validation rows."""
    if target in ("mlb", "regular"):
        objective, column, kw = "binary:logistic", target, {"eval_metric": "logloss"}
    elif target == "control_q":
        objective, column, kw = "reg:quantileerror", "control_war", {"quantile_alpha": list(QUANTILES)}
    elif target == "eta_years":
        objective, column, kw = "reg:squarederror", "eta_years", {}
    else:
        # Peak and control WAR are non-negative with a point mass at zero; Tweedie ordered the development
        # cohorts better than squared error (2014 to 2017 Spearman .376 against .343 for hitters, .334 against .309 for pitchers).
        objective, column, kw = "reg:tweedie", target, {"tweedie_variance_power": TWEEDIE_POWER}
    d = dev[dev[column].notna()]
    v = val[val[column].notna()]
    params = _params(pool, names, objective, **kw)
    y_d = d[column].astype(float).to_numpy()
    y_v = v[column].astype(float).to_numpy()
    if objective == "reg:tweedie":
        # A negative control WAR is worth nothing in a trade; Tweedie needs non-negative labels.
        y_d, y_v = np.clip(y_d, 0.0, None), np.clip(y_v, 0.0, None)
        d = d.assign(**{column: y_d})
        v = v.assign(**{column: y_v})
    booster = xgb.train(params, _dmatrix(d, names, y_d), MAX_ROUNDS, evals=[(_dmatrix(v, names, y_v), "val")],
                        early_stopping_rounds=EARLY_STOP, verbose_eval=False)
    rounds = max(int(booster.best_iteration) + 1, 50)
    full = pd.concat([d, v])
    refit = xgb.train(params, _dmatrix(full, names, full[column].astype(float).to_numpy()), rounds)
    return refit, rounds


def fit_baseline(pool: str, rows: pd.DataFrame) -> dict:
    cols = BASELINE_FEATURES[pool]
    X = rows[cols].astype(float)
    fill = X.median()
    X = X.fillna(fill)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(Xs, rows["regular"].astype(int))
    reg = LinearRegression().fit(Xs, rows["peak_war"].astype(float))
    return {"features": cols, "fill": fill.to_dict(), "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "logit_coef": clf.coef_[0].tolist(), "logit_intercept": float(clf.intercept_[0]),
            "linear_coef": reg.coef_.tolist(), "linear_intercept": float(reg.intercept_)}


def predict_baseline(base: dict, rows: pd.DataFrame) -> pd.DataFrame:
    X = rows[base["features"]].astype(float).fillna(pd.Series(base["fill"]))
    Xs = (X.to_numpy() - np.array(base["mean"])) / np.array(base["scale"])
    logit = Xs @ np.array(base["logit_coef"]) + base["logit_intercept"]
    return pd.DataFrame({"p_regular": 1 / (1 + np.exp(-logit)),
                         "peak_war": np.clip(Xs @ np.array(base["linear_coef"]) + base["linear_intercept"], 0, None)}, index=rows.index)


def _spearman(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    return float(spearmanr(a[m], b[m]).statistic) if m.sum() > 2 else float("nan")


def evaluate(rows: pd.DataFrame, pred: pd.DataFrame, base: pd.DataFrame) -> dict:
    """Holdout metrics for the model and the baseline on the same rows."""
    y_reg = rows["regular"].astype(int)
    y_mlb = rows["mlb"].astype(int)
    out = {"n": int(len(rows)), "regular_rate": float(y_reg.mean()), "mlb_rate": float(y_mlb.mean())}
    for name, p in (("model", pred), ("baseline", base)):
        m = {"logloss_regular": float(log_loss(y_reg, np.clip(p["p_regular"], 1e-6, 1 - 1e-6))),
             "auc_regular": float(roc_auc_score(y_reg, p["p_regular"])) if y_reg.nunique() > 1 else float("nan"),
             "spearman_peak": _spearman(p["peak_war"], rows["peak_war"])}
        top = p["peak_war"] >= p["peak_war"].quantile(1 - TOP_SHARE)
        m["top_decile_regular_rate"] = float(y_reg[top].mean()) if top.any() else float("nan")
        if "p_mlb" in p:
            m["logloss_mlb"] = float(log_loss(y_mlb, np.clip(p["p_mlb"], 1e-6, 1 - 1e-6)))
            m["auc_mlb"] = float(roc_auc_score(y_mlb, p["p_mlb"])) if y_mlb.nunique() > 1 else float("nan")
        if "control_war" in p:
            c = rows["control_war"].notna()
            m["spearman_control"] = _spearman(p["control_war"][c], rows["control_war"][c])
            m["control_n"] = int(c.sum())
        if "eta_years" in p:
            r = rows["eta_years"].notna()
            m["eta_mae"] = float((p["eta_years"][r] - rows["eta_years"][r]).abs().mean()) if r.any() else float("nan")
        out[name] = m
    return out


def calibration(grade: pd.Series, rows: pd.DataFrame) -> list[dict]:
    """Realized outcomes by grade bin, to read against Edwards' FV base rates."""
    table = []
    for lo, hi in GRADE_BINS:
        m = (grade >= lo) & (grade < hi)
        if m.sum() == 0:
            continue
        c = rows.loc[m, "control_war"]
        table.append({"bin": f"{lo}-{hi - 1}", "n": int(m.sum()), "regular_rate": float(rows.loc[m, "regular"].mean()),
                      "mlb_rate": float(rows.loc[m, "mlb"].mean()), "mean_peak_war": float(rows.loc[m, "peak_war"].mean()),
                      "mean_control_war": float(c.mean()) if c.notna().any() else None,
                      "bust_rate": float((c.fillna(0) < 0.5).mean()) if c.notna().any() else None})
    return table


def risk_score(pred: pd.DataFrame) -> pd.Series:
    """Risk is the chance of never reaching the majors; the control-WAR quartiles sit at zero for most prospects, so they cannot band it."""
    return 1.0 - pred["p_mlb"]


def list_comparison(rows: pd.DataFrame, pred: pd.DataFrame, ranked: pd.DataFrame) -> dict:
    """On the holdout players MLB Pipeline ranked: how the model's order and the list's order each track the realized peak."""
    if ranked.empty:
        return {"n": 0}
    r = rows[["player_id", "season", "peak_war", "regular"]].copy()
    r["model_peak"] = pred["peak_war"].to_numpy()
    r = r.merge(ranked.dropna(subset=["player_id"])[["player_id", "season", "rank"]].astype({"player_id": "int64"}),
                on=["player_id", "season"], how="inner")
    if len(r) < 6:
        return {"n": int(len(r))}
    return {"n": int(len(r)), "spearman_model": _spearman(r["model_peak"], r["peak_war"]),
            "spearman_pipeline": _spearman(-r["rank"], r["peak_war"]), "rank_agreement": _spearman(r["model_peak"], -r["rank"]),
            "regular_rate": float(r["regular"].mean())}


def grade_reference(last_season: int) -> dict[str, dict[str, float]]:
    """Mean and spread of MLB projected WAR per pool from the published player table, or realized WAR of regulars as a fallback."""
    try:
        from sqlalchemy import text

        from backend.db import engine
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT position_group, projected_war FROM player_values WHERE projected_war IS NOT NULL")).fetchall()
        frame = pd.DataFrame(rows, columns=["group", "war"])
        frame["war"] = frame["war"].astype(float)
        frame["pool"] = np.where(frame["group"].isin(["SP", "RP"]), "pit", "bat")
        source = "player_values.projected_war"
    except Exception as exc:  # noqa: BLE001 - fitting must work without the database
        log.warning(f"grade reference: falling back to realized WAR ({exc})")
        mlb = targets.mlb_seasons(last_season)
        recent = mlb[(mlb["season"] > last_season - 3) & mlb["qualifies"]]
        from v2.players.production import load_war
        roles = load_war(last_season).groupby("player_id")["role"].agg(lambda s: s.value_counts().idxmax())
        frame = recent.assign(pool=recent["player_id"].map(roles)).rename(columns={"war": "war"})
        source = "realized_war_regulars"
    out = {}
    for pool, g in frame.groupby("pool"):
        out[pool] = {"mean": float(g["war"].mean()), "sd": float(g["war"].std()), "n": int(len(g)), "source": source}
    return out


def to_grade(peak_war: pd.Series, ref: dict[str, float]) -> pd.Series:
    return (50 + 10 * (peak_war - ref["mean"]) / ref["sd"]).clip(20, 80)


def price_from_player_model() -> tuple[dict, dict[int, float]]:
    try:
        from v2.players.model import load_latest_from_db
        m = load_latest_from_db()
        return dict(m.price), dict(m.arb_fractions)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"price: using player model fallbacks ({exc})")
        from v2.players.model import ValueModel
        m = ValueModel()
        return dict(m.price), dict(m.arb_fractions)


def fit(last_dev_season: int, holdout: list[int], last_season: int) -> ProspectModel:
    dev_seasons = list(range(FIRST_DEV_SEASON, last_dev_season + 1))
    data = dataset.build(dev_seasons + holdout, last_season)
    data = data[data["gradeable"] & data["window_complete"]].copy()
    data["eta_years"] = (data["eta"] - data["season"]).where(data["eta"].notna())
    model = ProspectModel(fitted_at=datetime.now(UTC).isoformat(), dev_seasons=dev_seasons, holdout_seasons=holdout, last_season=last_season)
    model.price, model.arb_fractions = price_from_player_model()
    model.grade_reference = grade_reference(last_season)
    evaluation = {"dev_seasons": dev_seasons, "holdout_seasons": holdout, "last_season": last_season, "holdout_touches": HOLDOUT_TOUCHES,
                  "pools": {}}
    val_seasons = dev_seasons[-VALIDATION_COHORTS:]
    bio_tables = dataset.load_tables(list(range(FIRST_DEV_SEASON - dataset.WINDOW + 1, max(holdout) + 1)))
    active = pd.concat([bio_tables[0][["player_id", "season"]], bio_tables[1][["player_id", "season"]]]).drop_duplicates()
    bio, _ = identity.people(sorted(active["player_id"].unique()))
    list_years = [y for y in lists.available_years() if y - 1 in holdout]
    ranked = lists.load(bio, active, list_years) if list_years else pd.DataFrame(columns=["player_id", "season", "rank"])
    for pool in POOLS:
        rows = data[data["role"] == pool]
        dev = rows[rows["season"].isin(dev_seasons)]
        hold = rows[rows["season"].isin(holdout)]
        names = [n for n in _feature_names(pool) if n in rows.columns]
        model.feature_names[pool] = names
        d, v = dev[~dev["season"].isin(val_seasons)], dev[dev["season"].isin(val_seasons)]
        log.info(f"fit[{pool}]: {len(dev):,} dev rows ({len(v):,} validation), {len(hold):,} holdout rows, {len(names)} features")
        for target in ("mlb", "regular", "peak_war", "control_war", "control_q", "eta_years"):
            booster, rounds = _fit_target(pool, names, d, v, target)
            model.set_booster(pool, target, booster)
            model.rounds.setdefault(pool, {})[target] = rounds
        model.baseline[pool] = fit_baseline(pool, dev)
        pred, _ = model.predict(hold, pool)
        base = predict_baseline(model.baseline[pool], hold)
        metrics = evaluate(hold, pred, base)
        grade = to_grade(pred["peak_war"], model.grade_reference[pool])
        metrics["calibration"] = calibration(grade, hold)
        metrics["pipeline"] = list_comparison(hold, pred, ranked)
        passed = (metrics["model"]["logloss_regular"] < metrics["baseline"]["logloss_regular"]
                  and metrics["model"]["spearman_peak"] > metrics["baseline"]["spearman_peak"])
        model.validated[pool] = bool(passed)
        metrics["validated"] = bool(passed)
        metrics["rounds"] = model.rounds[pool]
        model.risk_cutpoints[pool] = [float(q) for q in np.nanquantile(risk_score(pred).to_numpy(), [0.25, 0.5, 0.75])]
        evaluation["pools"][pool] = metrics
        log.info(f"fit[{pool}]: holdout logloss(regular) {metrics['model']['logloss_regular']:.4f} vs baseline "
                 f"{metrics['baseline']['logloss_regular']:.4f}; spearman(peak) {metrics['model']['spearman_peak']:.3f} vs "
                 f"{metrics['baseline']['spearman_peak']:.3f}; validated={passed}")
    base_h, base_p = dataset.baselines(bio_tables, bio)
    model.level_age = {"bat": base_h.to_dict("records"), "pit": base_p.to_dict("records")}
    model.evaluation = evaluation
    EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVALUATION_PATH.write_text(json.dumps(evaluation, indent=1, default=float))
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--last-dev-season", type=int, required=True)
    parser.add_argument("--holdout", type=int, nargs="+", required=True)
    parser.add_argument("--last-season", type=int, default=datetime.now(UTC).year)
    parser.add_argument("--write-db", action="store_true")
    args = parser.parse_args()
    setup_logging()
    model = fit(args.last_dev_season, args.holdout, args.last_season)
    path = model.save(model_path(args.last_dev_season))
    log.info(f"model saved to {path}")
    if args.write_db:
        publish_model(model)


if __name__ == "__main__":
    main()
