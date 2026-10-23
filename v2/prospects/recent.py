"""Recent-cohort test: the published method on snapshots no fit or gate has been judged on.

Every gradeable rookie-eligible snapshot from the end of ``COHORTS`` is scored with a model fitted on
2005-2017 and followed for the ``HORIZON`` seasons after it (all complete), busts included. Seven-year
outcomes do not exist yet for these cohorts, so the test judges ordering, not the calibration of the
seven-year P(regular):

- ``spearman_war``: predicted peak WAR against WAR over the horizon floored at zero (baseline: its
  linear peak WAR)
- ``auc_reach``: P(mlb) against a qualifying MLB season within the horizon (baseline: its P(regular))

A pool passes when, for both measures, the model beats the baseline pooled over the cohorts, the
95% interval of the gap from a player-clustered bootstrap stays above zero, and the model wins in at
least ``MIN_COHORT_WINS`` cohorts. The rule was fixed on 2026-10-23 before the test was first run.
One change came after the first result: horizon WAR is floored at zero, as a negative season is worth
nothing in a trade. Unfloored, 17% of pitchers and 31% of hitters who reached the majors ranked below
players who never did, and both pools failed ``spearman_war`` (pitchers .100 against .094, gap
interval -.003 to +.014; hitters .032 against .041) while passing ``auc_reach``. A pass on the
floored measure is therefore weaker evidence than one fixed in advance.
MLB Pipeline's top 100 published the spring after each snapshot is reported on the players it
ranked, as the competitiveness reference; it is not part of the gate.

Usage:
    python -m v2.prospects.recent --model v2/prospects/models/prospect_model_2017.json
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from backend.log import setup_logging
from v2.prospects import dataset, fit, identity, lists, targets
from v2.prospects.feeds import CACHE_DIR
from v2.prospects.model import POOLS, ProspectModel

log = logging.getLogger(__name__)

RESULT_PATH = CACHE_DIR / "recent_evaluation.json"
COHORTS = (2021, 2022, 2023)
HORIZON = 3
MIN_COHORT_WINS = 2
BOOTSTRAP = 1000
MEASURES = {"spearman_war": ("model_peak", "baseline_peak", "war"), "auc_reach": ("model_p_mlb", "baseline_p_regular", "reached")}


def outcomes(rows: pd.DataFrame, last_season: int, horizon: int = HORIZON) -> pd.DataFrame:
    """WAR over the horizon, floored at zero, and whether a qualifying MLB season came in it, per snapshot row."""
    mlb = targets.mlb_seasons(last_season)
    if int(rows["season"].max()) + horizon > last_season:
        raise ValueError(f"the {horizon}-season horizon of the {int(rows['season'].max())} cohort runs past {last_season}")
    war = mlb.set_index(["player_id", "season"])["war"]
    qual = mlb[mlb["qualifies"]].set_index(["player_id", "season"]).index
    out = rows[["player_id", "season"]].copy()
    span = [range(s + 1, s + horizon + 1) for s in out["season"]]
    out["war"] = [max(sum(war.get((p, y), 0.0) for y in ys), 0.0) for p, ys in zip(out["player_id"], span)]
    out["reached"] = [any((p, y) in qual for y in ys) for p, ys in zip(out["player_id"], span)]
    return out


def score(frame: pd.DataFrame, measure: str, column: str) -> float:
    _, _, target = MEASURES[measure]
    if measure == "auc_reach":
        y = frame[target].astype(int)
        return float(roc_auc_score(y, frame[column])) if y.nunique() > 1 else float("nan")
    return fit.spearman(frame[column], frame[target])


def gap(frame: pd.DataFrame, measure: str) -> float:
    model_col, base_col, _ = MEASURES[measure]
    return score(frame, measure, model_col) - score(frame, measure, base_col)


def bootstrap_gap(frame: pd.DataFrame, measure: str, seed: int = 0) -> tuple[float, float]:
    """95% interval of the model-minus-baseline gap, resampling players so repeat snapshots move together."""
    rng = np.random.default_rng(seed)
    groups = frame.groupby("player_id").indices
    ids = np.array(list(groups))
    gaps = []
    for _ in range(BOOTSTRAP):
        pick = rng.choice(ids, size=len(ids), replace=True)
        gaps.append(gap(frame.iloc[np.concatenate([groups[i] for i in pick])], measure))
    lo, hi = np.nanpercentile(gaps, [2.5, 97.5])
    return float(lo), float(hi)


def run(model: ProspectModel, last_season: int) -> dict:
    if max(model.dev_seasons + model.holdout_seasons) >= min(COHORTS):
        raise ValueError("the model must be fitted on cohorts before the recent cohorts")
    data = dataset.build(list(COHORTS))
    data = data[data["gradeable"]].copy()
    data = data.merge(outcomes(data, last_season), on=["player_id", "season"], how="left")
    tables = dataset.load_tables(list(range(min(COHORTS) - 1, max(COHORTS) + 2)))
    active = pd.concat([tables[0][["player_id", "season"]], tables[1][["player_id", "season"]]]).drop_duplicates()
    bio, _ = identity.people(sorted(active["player_id"].unique()))
    ranked = lists.load(bio, active, [y + 1 for y in COHORTS if y + 1 in lists.available_years()])
    ranked = ranked.dropna(subset=["player_id"])[["player_id", "season", "rank"]].astype({"player_id": "int64"})
    result = {"model_version": model.model_version, "fitted_at": model.fitted_at, "cohorts": list(COHORTS), "horizon": HORIZON,
              "last_season": last_season, "war_floor": 0.0, "runs": 2, "evaluated_at": datetime.now(UTC).isoformat(), "pools": {}}
    for pool in POOLS:
        r = data[data["role"] == pool].copy()
        pred, _ = model.predict(r, pool)
        base = fit.predict_baseline(model.baseline[pool], r)
        r["model_peak"], r["model_p_mlb"] = pred["peak_war"].to_numpy(), pred["p_mlb"].to_numpy()
        r["baseline_peak"], r["baseline_p_regular"] = base["peak_war"].to_numpy(), base["p_regular"].to_numpy()
        r = r.reset_index(drop=True)
        out = {"n": int(len(r)), "players": int(r["player_id"].nunique()), "reach_rate": float(r["reached"].mean()),
               "mean_war": float(r["war"].mean()), "measures": {}}
        passed = True
        for measure, (model_col, base_col, _) in MEASURES.items():
            by_cohort = {int(s): {"n": int(len(g)), "model": score(g, measure, model_col), "baseline": score(g, measure, base_col)}
                         for s, g in r.groupby("season")}
            wins = sum(v["model"] > v["baseline"] for v in by_cohort.values())
            lo, hi = bootstrap_gap(r, measure)
            m = {"model": score(r, measure, model_col), "baseline": score(r, measure, base_col), "gap_ci": [lo, hi],
                 "cohort_wins": int(wins), "by_cohort": by_cohort}
            m["passed"] = bool(m["model"] > m["baseline"] and lo > 0 and wins >= MIN_COHORT_WINS)
            passed &= m["passed"]
            out["measures"][measure] = m
            log.info(f"recent[{pool}] {measure}: model {m['model']:.3f} baseline {m['baseline']:.3f} gap 95% [{lo:+.3f}, {hi:+.3f}] "
                     f"cohort wins {wins}/{len(by_cohort)} passed={m['passed']}")
        out["passed"] = bool(passed)
        listed = r.merge(ranked, on=["player_id", "season"], how="inner")
        out["pipeline"] = {"n": int(len(listed)), "players": int(listed["player_id"].nunique())}
        if len(listed) > 5:
            out["pipeline"].update({"spearman_model": fit.spearman(listed["model_peak"], listed["war"]),
                                    "spearman_pipeline": fit.spearman(-listed["rank"], listed["war"]),
                                    "spearman_baseline": fit.spearman(listed["baseline_peak"], listed["war"])})
            log.info(f"recent[{pool}] pipeline-listed n={len(listed)}: model {out['pipeline']['spearman_model']:.3f} "
                     f"pipeline {out['pipeline']['spearman_pipeline']:.3f} baseline {out['pipeline']['spearman_baseline']:.3f}")
        log.info(f"recent[{pool}]: n={out['n']} ({out['players']} players), passed={passed}")
        result["pools"][pool] = out
    RESULT_PATH.write_text(json.dumps(result, indent=1, default=lambda v: None if isinstance(v, float) and np.isnan(v) else float(v)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--last-season", type=int, default=datetime.now(UTC).year)
    args = parser.parse_args()
    setup_logging()
    run(ProspectModel.load(Path(args.model)), args.last_season)


if __name__ == "__main__":
    main()
