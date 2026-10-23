"""Fit the prospect model on matured cohorts, evaluate it once on the holdout, and store it.

Per pool (hitters, pitchers), gradient-boosted trees predict from each snapshot's features:
reaching the majors, becoming a regular, peak WAR, control-years WAR with its quartiles, and years
to arrival. Rounds are chosen by early stopping on a player-grouped slice of the development cohorts,
then every booster is refit on all development cohorts with that round count. P(regular) is Platt-scaled
on the early-stopped booster's predictions for that slice, players the trees never trained on. A
KATOH-style logistic and linear baseline (age relative to level, strikeout, walk and power rates, level) is fitted on the
same rows. Players who appear in a holdout cohort are left out of every development cohort, so a
holdout label never shares its outcome seasons with a training label of the same player. The holdout
is scored once per documented touch; a pool passes the publish gate when it beats the baseline on both
log loss for ``regular`` and Spearman for ``peak_war``.

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
from sqlalchemy import text

from backend.db import engine
from backend.log import setup_logging
from v2.players.model import load_latest_from_db
from v2.prospects import dataset, features, identity, lists
from v2.prospects.feeds import CACHE_DIR
from v2.prospects.model import POOLS, QUANTILES, ProspectModel, logit, model_path, publish_model, risk_score

log = logging.getLogger(__name__)

FIRST_DEV_SEASON = 2005
VALIDATION_SHARE = 0.15  # of development players, grouped by player, for early stopping only
EVALUATION_PATH = CACHE_DIR / "evaluation.json"
PARAMS = {"max_depth": 4, "eta": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 25, "lambda": 2.0}
MAX_ROUNDS = 3000
TWEEDIE_POWER = 1.5
HOLDOUT_TOUCHES = 4  # 2026-10-22: second pass after the development experiments (call-code and trend features, Tweedie); third after the audit fixes and the player-disjoint split; 2026-10-23: fourth after dropping partial pitch logs and Platt-scaling P(regular)
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


def _fit_target(pool: str, names: list[str], dev: pd.DataFrame, val: pd.DataFrame, target: str) -> tuple[xgb.Booster, int, np.ndarray]:
    """Early-stopped rounds on the validation rows, then a refit on dev plus validation rows; also the early-stopped validation predictions."""
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
    val_pred = booster.predict(_dmatrix(v, names), iteration_range=(0, rounds))
    full = pd.concat([d, v])
    refit = xgb.train(params, _dmatrix(full, names, full[column].astype(float).to_numpy()), rounds)
    return refit, rounds, val_pred


def fit_platt(pred: np.ndarray, y: pd.Series) -> list[float]:
    """Slope and intercept of an unregularized logistic fit of the outcome on the logit of the prediction."""
    clf = LogisticRegression(C=1e6, max_iter=1000).fit(logit(pred).reshape(-1, 1), y.astype(int))
    return [float(clf.coef_[0][0]), float(clf.intercept_[0])]


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


def spearman(a: pd.Series, b: pd.Series) -> float:
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
             "spearman_peak": spearman(p["peak_war"], rows["peak_war"])}
        top = p["peak_war"] >= p["peak_war"].quantile(1 - TOP_SHARE)
        m["top_decile_regular_rate"] = float(y_reg[top].mean()) if top.any() else float("nan")
        if "p_mlb" in p:
            m["logloss_mlb"] = float(log_loss(y_mlb, np.clip(p["p_mlb"], 1e-6, 1 - 1e-6)))
            m["auc_mlb"] = float(roc_auc_score(y_mlb, p["p_mlb"])) if y_mlb.nunique() > 1 else float("nan")
        if "control_war" in p:
            c = rows["control_war"].notna()
            m["spearman_control"] = spearman(p["control_war"][c], rows["control_war"][c])
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
    return {"n": int(len(r)), "spearman_model": spearman(r["model_peak"], r["peak_war"]),
            "spearman_pipeline": spearman(-r["rank"], r["peak_war"]), "rank_agreement": spearman(r["model_peak"], -r["rank"]),
            "regular_rate": float(r["regular"].mean())}


def grade_reference() -> dict[str, dict[str, float]]:
    """Mean and spread of the published MLB projected WAR per pool: the scale a 60 prospect is measured on."""
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT position_group, projected_war FROM player_values WHERE projected_war IS NOT NULL")).fetchall()
    frame = pd.DataFrame(rows, columns=["group", "war"])
    if frame.empty:
        raise RuntimeError("player_values has no projected WAR to scale prospect grades against")
    frame["war"] = frame["war"].astype(float)
    frame["pool"] = np.where(frame["group"].isin(["SP", "RP"]), "pit", "bat")
    return {pool: {"mean": float(g["war"].mean()), "sd": float(g["war"].std()), "n": int(len(g)), "source": "player_values.projected_war"}
            for pool, g in frame.groupby("pool")}


def price_from_player_model() -> tuple[dict, dict[int, float]]:
    """The published player model's price of a win and arbitration fractions; prospects are priced on the same scale."""
    m = load_latest_from_db()
    return dict(m.price), dict(m.arb_fractions)


def fit(last_dev_season: int, holdout: list[int], last_season: int) -> ProspectModel:
    dev_seasons = list(range(FIRST_DEV_SEASON, last_dev_season + 1))
    data = dataset.build(dev_seasons + holdout, last_season)
    data = data[data["gradeable"] & data["window_complete"]].copy()
    data["eta_years"] = (data["eta"] - data["season"]).where(data["eta"].notna())
    model = ProspectModel(fitted_at=datetime.now(UTC).isoformat(), dev_seasons=dev_seasons, holdout_seasons=holdout, last_season=last_season)
    model.price, model.arb_fractions = price_from_player_model()
    model.grade_reference = grade_reference()
    evaluation = {"dev_seasons": dev_seasons, "holdout_seasons": holdout, "last_season": last_season, "holdout_touches": HOLDOUT_TOUCHES,
                  "pools": {}}
    bio_tables = dataset.load_tables(list(range(FIRST_DEV_SEASON - dataset.WINDOW + 1, max(holdout) + 1)))
    active = pd.concat([bio_tables[0][["player_id", "season"]], bio_tables[1][["player_id", "season"]]]).drop_duplicates()
    bio, _ = identity.people(sorted(active["player_id"].unique()))
    list_years = [y for y in lists.available_years() if y - 1 in holdout]
    ranked = lists.load(bio, active, list_years) if list_years else pd.DataFrame(columns=["player_id", "season", "rank"])
    for pool in POOLS:
        rows = data[data["role"] == pool]
        hold = rows[rows["season"].isin(holdout)]
        # Player-disjoint: a holdout player's earlier snapshots would share most of his outcome window with the holdout label.
        dev = rows[rows["season"].isin(dev_seasons) & ~rows["player_id"].isin(hold["player_id"])]
        names = [n for n in _feature_names(pool) if n in rows.columns]
        model.feature_names[pool] = names
        in_val = (dev["player_id"] % 100) < VALIDATION_SHARE * 100  # a fixed player-grouped slice, every season represented
        d, v = dev[~in_val], dev[in_val]
        log.info(f"fit[{pool}]: {len(dev):,} dev rows ({len(v):,} validation) after removing holdout players, {len(hold):,} holdout rows, {len(names)} features")
        for target in ("mlb", "regular", "peak_war", "control_war", "control_q", "eta_years"):
            booster, rounds, val_pred = _fit_target(pool, names, d, v, target)
            if target == "regular":
                model.regular_platt[pool] = fit_platt(val_pred, v["regular"])
            model.set_booster(pool, target, booster)
            model.rounds.setdefault(pool, {})[target] = rounds
        model.baseline[pool] = fit_baseline(pool, dev)
        pred, _ = model.predict(hold, pool)
        base = predict_baseline(model.baseline[pool], hold)
        metrics = evaluate(hold, pred, base)
        metrics["calibration"] = calibration(model.to_grade(pred["peak_war"], pool), hold)
        metrics["pipeline"] = list_comparison(hold, pred, ranked)
        passed = (metrics["model"]["logloss_regular"] < metrics["baseline"]["logloss_regular"]
                  and metrics["model"]["spearman_peak"] > metrics["baseline"]["spearman_peak"])
        model.validated[pool] = bool(passed)
        metrics["validated"] = bool(passed)
        metrics["rounds"] = model.rounds[pool]
        metrics["regular_platt"] = model.regular_platt[pool]
        model.risk_cutpoints[pool] = [float(q) for q in np.nanquantile(risk_score(pred).to_numpy(), [0.25, 0.5, 0.75])]
        model.eta_band[pool] = max(1, int(round(metrics["model"]["eta_mae"])))
        evaluation["pools"][pool] = metrics
        log.info(f"fit[{pool}]: holdout logloss(regular) {metrics['model']['logloss_regular']:.4f} vs baseline "
                 f"{metrics['baseline']['logloss_regular']:.4f}; spearman(peak) {metrics['model']['spearman_peak']:.3f} vs "
                 f"{metrics['baseline']['spearman_peak']:.3f}; validated={passed}")
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
