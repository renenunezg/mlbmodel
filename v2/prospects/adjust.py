"""The stuff adjustment: how much tracked stuff moves a pitcher's outcomes beyond the outcome-only model.

The published prospect model (fitted on 2005-2017, never refit here) gives every pitcher snapshot a
P(mlb) and a peak WAR. For tracked pitchers (``stuff.TRACKED_FLOOR`` released pitches in the window),
two shallow boosters start from those predictions as offsets and learn from stuff summaries
(``FEATURES``): one for reaching the majors within the horizon, one for horizon WAR floored at zero.
On the board the adjustment only re-orders tracked pitchers (``apply``); untracked pitchers keep the
base prediction exactly.

The rules were fixed on 2026-10-23 before the first run (``COHORTS`` to ``ROUNDS``). Every tracked
pitcher snapshot of the cohorts is scored by a cross-fitting fold that never saw that player. The
adjustment is adopted when, for floored-WAR Spearman and reach AUC, the adjusted-minus-base gap
(cohort gaps averaged with row weights) is above zero with its 95% player-clustered bootstrap interval
above zero; pitchers are graded only if, in addition, the same Spearman gap holds among the
``ELITE_N`` tracked rows per cohort with the highest base peak WAR. MLB Pipeline's top 100 is reported
on the tracked pitchers it ranked, never gated. The 2021-2023 outcomes were seen in aggregate once by
the base-model recent-cohort test (``recent.py``); nothing about stuff was fit or chosen on them.

Usage:
    python -m v2.prospects.adjust --model v2/prospects/models/prospect_model_2017.json --stuff v2/prospects/models/stuff_model.json
"""
from __future__ import annotations

import argparse
import base64
import gzip
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from backend.log import setup_logging
from v2.prospects import dataset, features, fit, identity, injuries, lists, recent, stuff
from v2.prospects.model import ProspectModel, logit

log = logging.getLogger(__name__)

COHORTS = {2021: 3, 2022: 3, 2023: 3, 2024: 2}  # snapshot season -> outcome horizon in seasons
FOLDS = 5
ELITE_N = 200
BOOTSTRAP = 1000
PARAMS = {"max_depth": 2, "eta": 0.05, "min_child_weight": 20, "subsample": 0.8, "verbosity": 0}
ROUNDS = 100
TWEEDIE_POWER = 1.5
WAR_EPS = 0.01
# fsl_share and spring_share let the adjustment read those levels' grades on their own scale where they could not be shifted.
STUFF_FEATURES = ["fb_stuff", "best_secondary_stuff", "arsenal_stuff", "command_grade", "fb_velo", "n_types", "fsl_share", "spring_share"]
# Injured-list history (second pass, 2026-10-23): aimed at the overconfident top of the pitcher board.
FEATURES = [*STUFF_FEATURES, *injuries.FEATURES]


def stuff_features(model: stuff.StuffModel, season: int) -> pd.DataFrame:
    """Per tracked pitcher at the end of ``season``: the stuff summaries and injured-list history the adjustment reads, by player_id."""
    return with_injuries(summarize(*stuff.window_arsenal(model, season, features.WINDOW_WEIGHTS)), season)


def with_injuries(feats: pd.DataFrame, season: int, refresh_current: bool = False) -> pd.DataFrame:
    """Stuff summaries joined with window injured-list history; a tracked pitcher never on a list reads zero."""
    il = injuries.window_features(season, features.WINDOW_WEIGHTS, refresh_current).reindex(feats.index).fillna(0.0)
    return pd.concat([feats, il], axis=1)[FEATURES]


def summarize(arsenal: pd.DataFrame, pitchers_: pd.DataFrame) -> pd.DataFrame:
    """Stuff summaries from a window arsenal: primary fastball grade and velocity, best secondary, usage-weighted grade, command."""
    if pitchers_.empty:
        return pd.DataFrame(columns=STUFF_FEATURES)
    a = arsenal.copy()
    fb = a[a["family"].eq("fastball")].sort_values("usage").groupby("pitcher").tail(1).set_index("pitcher")
    sec = a[~a["family"].eq("fastball")].groupby("pitcher")["stuff_grade"].max()
    weighted = (a["stuff_grade"] * a["usage"]).groupby(a["pitcher"]).sum() / a.groupby("pitcher")["usage"].sum()
    out = pitchers_.set_index("pitcher")[["command_grade", "fsl_share", "spring_share"]].copy()
    out["fb_stuff"] = fb["stuff_grade"].reindex(out.index)
    out["fb_velo"] = fb["velo"].reindex(out.index)
    out["best_secondary_stuff"] = sec.reindex(out.index)
    out["arsenal_stuff"] = weighted.reindex(out.index)
    out["n_types"] = a.groupby("pitcher").size().reindex(out.index).fillna(0).astype(float)
    out.index.name = "player_id"
    return out[STUFF_FEATURES].astype(float)


def _encode(b: xgb.Booster) -> str:
    return base64.b64encode(gzip.compress(bytes(b.save_raw("ubj")), 9)).decode("ascii")


def _decode(s: str) -> xgb.Booster:
    b = xgb.Booster()
    b.load_model(bytearray(gzip.decompress(base64.b64decode(s))))
    return b


def _margins(base: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return logit(base["p_mlb"].to_numpy(float)), np.log(base["peak_war"].to_numpy(float) + WAR_EPS)


def _train(feats: pd.DataFrame, base: pd.DataFrame, reached: pd.Series, war: pd.Series, seed: int = 0) -> tuple[xgb.Booster, xgb.Booster]:
    m_reach, m_war = _margins(base)
    X = feats[FEATURES].astype(float)
    reach = xgb.train({**PARAMS, "objective": "binary:logistic", "seed": seed},
                      xgb.DMatrix(X, label=reached.astype(float), base_margin=m_reach, missing=np.nan), ROUNDS)
    wtrain = xgb.train({**PARAMS, "objective": "reg:tweedie", "tweedie_variance_power": TWEEDIE_POWER, "seed": seed},
                       xgb.DMatrix(X, label=war.astype(float), base_margin=m_war, missing=np.nan), ROUNDS)
    return reach, wtrain


def _shifts(reach: xgb.Booster, war: xgb.Booster, feats: pd.DataFrame, base: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Logit shift for the probabilities and log shift for the WAR quantities, per row."""
    m_reach, m_war = _margins(base)
    X = feats[FEATURES].astype(float)
    r = reach.predict(xgb.DMatrix(X, base_margin=m_reach, missing=np.nan), output_margin=True) - m_reach
    w = war.predict(xgb.DMatrix(X, base_margin=m_war, missing=np.nan), output_margin=True) - m_war
    return r, w


def apply(base: pd.DataFrame, feats: pd.DataFrame, adjustment: dict) -> pd.DataFrame:
    """Tracked rows re-ordered by the stuff adjustment, keeping the base model's values; untracked rows come back unchanged.

    The adjustment was validated on ordering only (Spearman and AUC over three-season outcomes), and its
    raw shifts also carry the change from a seven-season to a three-season horizon. So among the tracked
    rows passed in together (the graded pitchers of one board), the base model's own values of each
    quantity are reassigned in the adjusted order: P(mlb) by the adjusted reach score, P(regular) and the
    WAR quantities by the adjusted WAR score. Levels and spread stay the base model's.
    """
    out = base.astype("float64")  # booster outputs are float32
    have = feats.reindex(columns=STUFF_FEATURES).notna().any(axis=1).to_numpy()
    if not have.any():
        return out
    rows = out[have]
    r, w = _shifts(_decode(adjustment["reach"]), _decode(adjustment["war"]), feats[have], rows)
    m_reach, m_war = _margins(rows)
    order = {"p_mlb": m_reach + r}
    for col in ("p_regular", "peak_war", "control_war", "control_q25", "control_q75"):
        order[col] = m_war + w
    for col, score in order.items():
        if col in out:
            ranks = np.argsort(np.argsort(score, kind="stable"), kind="stable")
            out.loc[have, col] = np.sort(rows[col].to_numpy(float))[ranks]
    return out


def _gap(frame: pd.DataFrame, kind: str) -> float:
    """Row-weighted mean over cohorts of (adjusted minus base) for one measure."""
    gaps, weights = [], []
    for _, g in frame.groupby("season"):
        if kind == "auc":
            y = g["reached"].astype(int)
            if y.nunique() < 2:
                continue
            gaps.append(roc_auc_score(y, g["adj_p_mlb"]) - roc_auc_score(y, g["p_mlb"]))
        else:
            gaps.append(fit.spearman(g["adj_peak"], g["war"]) - fit.spearman(g["peak_war"], g["war"]))
        weights.append(len(g))
    return float(np.average(gaps, weights=weights)) if gaps else float("nan")


def _interval(frame: pd.DataFrame, kind: str, rng: np.random.Generator) -> list[float]:
    groups = frame.groupby("player_id").indices
    ids = np.array(list(groups))
    draws = []
    for _ in range(BOOTSTRAP):
        pick = rng.choice(ids, size=len(ids), replace=True)
        draws.append(_gap(frame.iloc[np.concatenate([groups[i] for i in pick])], kind))
    return [float(v) for v in np.nanpercentile(draws, [2.5, 97.5])]


def run(model: ProspectModel, stuff_model: stuff.StuffModel, last_season: int) -> dict:
    frames = []
    for season, horizon in COHORTS.items():
        rows = dataset.build([season])
        rows = rows[rows["gradeable"] & rows["role"].eq("pit")].copy()
        feats = stuff_features(stuff_model, season)
        rows = rows[rows["player_id"].isin(feats.index)]
        rows = rows.merge(recent.outcomes(rows, last_season, horizon), on=["player_id", "season"])
        pred, _ = model.predict(rows, "pit")
        rows[["p_mlb", "peak_war"]] = pred[["p_mlb", "peak_war"]].to_numpy()
        # The base snapshot carries its own feed-based fb_velo; the adjustment reads only the stuff summaries.
        frames.append(rows.drop(columns=[c for c in FEATURES if c in rows]).join(feats, on="player_id"))
        log.info(f"adjust[{season}]: {len(rows):,} tracked pitcher rows, horizon {horizon}, reach rate {rows['reached'].mean():.3f}")
    data = pd.concat(frames, ignore_index=True)
    fold = data["player_id"] % FOLDS
    data["adj_p_mlb"], data["adj_peak"] = np.nan, np.nan
    for k in range(FOLDS):
        tr, te = data[fold != k], data[fold == k]
        reach, war = _train(tr, tr, tr["reached"], tr["war"])
        r, w = _shifts(reach, war, te, te)
        data.loc[te.index, "adj_p_mlb"] = 1 / (1 + np.exp(-(logit(te["p_mlb"].to_numpy(float)) + r)))
        data.loc[te.index, "adj_peak"] = te["peak_war"].to_numpy(float) * np.exp(w)
    rng = np.random.default_rng(0)
    result = {"evaluated_at": datetime.now(UTC).isoformat(), "cohorts": {str(k): v for k, v in COHORTS.items()}, "rows": int(len(data)),
              "players": int(data["player_id"].nunique()), "disclosure": "2021-2023 outcomes were seen in aggregate by the base-model recent-cohort test"}
    for kind in ("spearman", "auc"):
        result[kind] = {"gap": _gap(data, kind), "ci": _interval(data, kind, rng)}
    elite = data.sort_values("peak_war", ascending=False).groupby("season").head(ELITE_N)
    result["elite"] = {"rows": int(len(elite)), "gap": _gap(elite, "spearman"), "ci": _interval(elite, "spearman", rng)}
    result["by_cohort"] = {int(s): {"n": int(len(g)), "spearman_base": fit.spearman(g["peak_war"], g["war"]),
                                    "spearman_adjusted": fit.spearman(g["adj_peak"], g["war"])} for s, g in data.groupby("season")}
    result["adoption_passed"] = bool(all(result[k]["gap"] > 0 and result[k]["ci"][0] > 0 for k in ("spearman", "auc")))
    result["elite_passed"] = bool(result["elite"]["gap"] > 0 and result["elite"]["ci"][0] > 0)
    result["pipeline"] = pipeline_report(data)
    reach, war = _train(data, data, data["reached"], data["war"])
    stuff_model.adjustment = {"reach": _encode(reach), "war": _encode(war), "features": FEATURES, "adoption_passed": result["adoption_passed"],
                              "elite_passed": result["elite_passed"], "evaluation": result}
    return result


def pipeline_report(data: pd.DataFrame) -> dict:
    tables = dataset.load_tables(list(range(min(COHORTS) - 1, max(COHORTS) + 2)))
    active = pd.concat([tables[0][["player_id", "season"]], tables[1][["player_id", "season"]]]).drop_duplicates()
    bio, _ = identity.people(sorted(active["player_id"].unique()))
    ranked = lists.load(bio, active, [y + 1 for y in COHORTS if y + 1 in lists.available_years()])
    ranked = ranked.dropna(subset=["player_id"])[["player_id", "season", "rank"]].astype({"player_id": "int64"})
    listed = data.merge(ranked, on=["player_id", "season"], how="inner")
    if len(listed) < 6:
        return {"n": int(len(listed))}
    return {"n": int(len(listed)), "players": int(listed["player_id"].nunique()),
            "spearman_base": fit.spearman(listed["peak_war"], listed["war"]),
            "spearman_adjusted": fit.spearman(listed["adj_peak"], listed["war"]),
            "spearman_pipeline": fit.spearman(-listed["rank"], listed["war"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--stuff", type=str, required=True)
    parser.add_argument("--last-season", type=int, default=datetime.now(UTC).year)
    args = parser.parse_args()
    setup_logging()
    stuff_model = stuff.StuffModel.load(Path(args.stuff))
    result = run(ProspectModel.load(Path(args.model)), stuff_model, args.last_season)
    log.info(f"adjust: spearman gap {result['spearman']['gap']:+.3f} {result['spearman']['ci']}, auc gap {result['auc']['gap']:+.3f} "
             f"{result['auc']['ci']}, elite gap {result['elite']['gap']:+.3f} {result['elite']['ci']}; adoption={result['adoption_passed']} "
             f"elite={result['elite_passed']}")
    log.info(f"adjust: by cohort {result['by_cohort']}; pipeline {result['pipeline']}")
    stuff_model.save(Path(args.stuff))


if __name__ == "__main__":
    main()
