"""Snapshot of recent-form xwOBA leaderboards + posterior sigma summaries.

Writes two tables consumed by the v2 Diagnostics tab:

  - posterior_skills: top-10 / bottom-10 by trailing 10-day Statcast xwOBA.
      Per-PA xwOBA value is taken from Statcast's `estimated_woba_using_speedangle`
      for balls in play, and from `woba_value` for K / BB / HBP. The
      per-actor leaderboard is the simple sum(value) / sum(denom).
      Batters split by pitcher hand (vs_rhp / vs_lhp); pitchers split by
      role (SP / RP) as determined by `classify_roles` on the same window.

  - posterior_sigmas: hierarchical sigma magnitudes from the trained PyMC
    traces, used by the Variance Decomposition chart. These remain
    posterior-derived because they describe the model itself, not players.

Usage:
    python -m v2.pipeline.write_posterior_summaries [--refit-date YYYY-MM-DD] [--window-days 10]
"""
from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import statsapi
from sqlalchemy import text

from backend.db import engine
from v2.bayesian._common import POSTERIORS_DIR, WOBA_WEIGHTS, encode_outcomes
from v2.bayesian.pitcher_skill import classify_roles
from v2.data.pa_dataset import EVENT_TO_OUTCOME, NON_PA_EVENTS, OUTCOMES
from v2.simulator.posteriors import PosteriorMeans, load_posteriors

TOP_N = 10
SIGMA_NAMES = ["sigma_batter", "sigma_platoon", "sigma_pitcher", "sigma_park"]
CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"
WINDOW_DAYS_DEFAULT = 10

# wOBA weights in OUTCOMES order; OUT is the last (reference) outcome.
WOBA_VEC = np.array([WOBA_WEIGHTS[o] for o in OUTCOMES], dtype=np.float64)

# Empirical-Bayes prior strength, in PA-equivalent pseudocounts: window outcomes
# are shrunk toward each actor's trained posterior skill. At 100, the ~15-25 PA
# in a 10-day window move the ranking only a little, which stops the daily churn
# raw 10-day xwOBA had. Tunable via --prior-strength.
PRIOR_STRENGTH_DEFAULT = 100.0

# Minimum PA / batters faced for a player to be eligible for the leaderboard.
# These scale with the window length passed at the CLI. vs_lhp gets a lower
# threshold because batters see roughly a third as many lefties.
MIN_PA_BATTER_VS_RHP_PER_DAY = 1.5  # ~15 PA across 10 days
MIN_PA_BATTER_VS_LHP_PER_DAY = 0.7  # ~7 PA across 10 days
MIN_BF_SP_PER_DAY = 2.0             # ~20 BF across 10 days ~ one full start
MIN_BF_RP_PER_DAY = 1.0             # ~10 BF across 10 days ~ 5 outings

PARQUET_COLS = [
    "game_pk", "game_date", "batter", "pitcher",
    "p_throws", "events",
    "home_team", "away_team", "inning", "inning_topbot",
]


def _load_window_pa(window_start: date, window_end: date) -> pd.DataFrame:
    """Load PA-terminating rows from cached parquet within [start, end].

    Each PA is mapped to one of the 8 categorical OUTCOMES; `outcome_code` is
    the index into OUTCOMES. The leaderboard skill is a wOBA computed from this
    categorical distribution (not statcast's speedangle xwOBA), so it shares the
    exact parameterization the Bayesian prior was fit on.
    """
    year = window_end.year
    path = CACHE_DIR / f"statcast_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    df = pd.read_parquet(path, columns=PARQUET_COLS)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df = df[(df["game_date"] >= window_start) & (df["game_date"] <= window_end)]
    df = df[df["events"].notna() & ~df["events"].isin(NON_PA_EVENTS)]
    df = df[df["events"].isin(EVENT_TO_OUTCOME)]

    df["outcome"] = df["events"].map(EVENT_TO_OUTCOME)
    df["outcome_code"] = encode_outcomes(df["outcome"])
    df["batter"] = df["batter"].astype("int64")
    df["pitcher"] = df["pitcher"].astype("int64")
    return df.reset_index(drop=True)


def _outcome_counts(pa: pd.DataFrame, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-actor 8-vector of outcome counts. Returns (sorted ids, counts[N,8])."""
    ids = np.sort(pa[key].unique()).astype(np.int64)
    pos = np.searchsorted(ids, pa[key].to_numpy(dtype=np.int64))
    counts = np.zeros((len(ids), len(OUTCOMES)), dtype=np.float64)
    np.add.at(counts, (pos, pa["outcome_code"].to_numpy()), 1.0)
    return ids, counts


def _prior_probs(
    pm: PosteriorMeans, actor_type: str, split: str, ids: np.ndarray
) -> np.ndarray:
    """Trained-posterior outcome probabilities per actor vs a league-average
    opponent. Returns (len(ids), 8) in OUTCOMES order. Unknown ids resolve to
    the league-mean fallback row in PosteriorMeans."""
    if actor_type == "batter":
        idx = pm.encode_batter(ids)
        free = pm.intercept[None, :] + pm.batter_offset[idx]
        if split == "vs_lhp":
            free = free + pm.platoon_offset[idx]
    else:
        idx = pm.encode_pitcher(ids)
        role = 0 if split == "sp" else 1
        free = pm.intercept[None, :] + pm.pitcher_offset[idx, role]

    # OUT is the reference outcome (last in OUTCOMES), logit fixed at 0.
    logits = np.concatenate([free, np.zeros((len(ids), 1))], axis=1)
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def _eb_xwoba(prior_probs: np.ndarray, counts: np.ndarray, strength: float) -> np.ndarray:
    """Dirichlet-Multinomial conjugate update: prior pseudocounts strength*prior
    plus observed window counts, normalized and mapped to wOBA."""
    n = counts.sum(axis=1, keepdims=True)
    post = (strength * prior_probs + counts) / (strength + n)
    return post @ WOBA_VEC


def _batter_team_lookup(pa: pd.DataFrame) -> dict[int, str]:
    """Most recent team in the window per batter (offense side)."""
    pa = pa.assign(team=np.where(pa["inning_topbot"] == "Top", pa["away_team"], pa["home_team"]))
    last = pa.sort_values("game_date").drop_duplicates("batter", keep="last")
    return dict(zip(last["batter"].astype(int), last["team"].astype(str)))


def _pitcher_team_lookup(pa: pd.DataFrame) -> dict[int, str]:
    pa = pa.assign(team=np.where(pa["inning_topbot"] == "Top", pa["home_team"], pa["away_team"]))
    last = pa.sort_values("game_date").drop_duplicates("pitcher", keep="last")
    return dict(zip(last["pitcher"].astype(int), last["team"].astype(str)))


def _fetch_names(ids: list[int], chunk: int = 100) -> dict[int, str]:
    out: dict[int, str] = {}
    for i in range(0, len(ids), chunk):
        batch = ids[i : i + chunk]
        try:
            resp = statsapi.get(
                "people",
                {"personIds": ",".join(str(x) for x in batch), "fields": "people,id,fullName"},
            )
            for p in resp.get("people", []):
                out[int(p["id"])] = p.get("fullName") or ""
        except Exception as e:
            print(f"  [warn] name lookup batch {i // chunk} failed: {e}")
        time.sleep(0.1)
    return out


def _build_leaderboard_rows(
    refit_date: date,
    actor_type: str,
    split_label: str,
    ids: np.ndarray,
    skill: np.ndarray,
    names: dict[int, str],
    teams: dict[int, str],
) -> list[dict]:
    order = np.argsort(skill)
    bottom_idx = order[:TOP_N]
    top_idx = order[-TOP_N:][::-1]

    rows = []
    for rank_type, idx_list in (("top", top_idx), ("bottom", bottom_idx)):
        for rank, i in enumerate(idx_list, start=1):
            aid = int(ids[i])
            rows.append({
                "refit_date": refit_date,
                "actor_type": actor_type,
                "split_label": split_label,
                "rank_type": rank_type,
                "rank": rank,
                "actor_id": aid,
                "actor_name": names.get(aid, ""),
                "team": teams.get(aid, ""),
                "skill_score": round(float(skill[i]), 5),
            })
    return rows


def _eb_leaderboard(
    sub: pd.DataFrame, key: str, split: str,
    pm: PosteriorMeans, min_n: int, strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical-Bayes posterior xwOBA per actor over the window, restricted to
    actors with at least min_n window PA. Returns (sorted ids, skill)."""
    actor_type = "batter" if key == "batter" else "pitcher"
    if sub.empty:
        return np.empty(0, np.int64), np.empty(0, np.float64)
    ids, counts = _outcome_counts(sub, key)
    keep = counts.sum(axis=1) >= min_n
    ids, counts = ids[keep], counts[keep]
    if len(ids) == 0:
        return ids, np.empty(0, np.float64)
    prior = _prior_probs(pm, actor_type, split, ids)
    return ids, _eb_xwoba(prior, counts, strength)


def _sigma_summary(idata: az.InferenceData, var: str) -> dict[str, float]:
    arr = idata.posterior[var].values
    flat = arr.mean(axis=tuple(range(2, arr.ndim)))
    return {
        "mean": round(float(flat.mean()), 5),
        "p10": round(float(np.quantile(flat, 0.10)), 5),
        "p90": round(float(np.quantile(flat, 0.90)), 5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit-date", default=str(date.today()))
    ap.add_argument("--window-days", type=int, default=WINDOW_DAYS_DEFAULT)
    ap.add_argument("--prior-strength", type=float, default=PRIOR_STRENGTH_DEFAULT)
    args = ap.parse_args()

    refit_date = pd.Timestamp(args.refit_date).date()
    window_days = args.window_days
    strength = args.prior_strength
    window_end = refit_date
    window_start = window_end - timedelta(days=window_days - 1)
    min_pa_vs_rhp = max(1, int(round(MIN_PA_BATTER_VS_RHP_PER_DAY * window_days)))
    min_pa_vs_lhp = max(1, int(round(MIN_PA_BATTER_VS_LHP_PER_DAY * window_days)))
    min_bf_sp = max(1, int(round(MIN_BF_SP_PER_DAY * window_days)))
    min_bf_rp = max(1, int(round(MIN_BF_RP_PER_DAY * window_days)))

    print(f"[posterior_summaries] refit_date={refit_date}  window={window_start}..{window_end}")
    print(f"[posterior_summaries] min PA: vs_rhp={min_pa_vs_rhp}  vs_lhp={min_pa_vs_lhp}  sp={min_bf_sp}  rp={min_bf_rp}")

    pa = _load_window_pa(window_start, window_end)
    print(f"[posterior_summaries] {len(pa):,} PAs in window  (prior-strength={strength})")

    pm = load_posteriors()

    # --- Batter leaderboards: vs_rhp and vs_lhp ---
    batter_rows: list[dict] = []
    batter_teams = _batter_team_lookup(pa)
    for split_label, hand, min_pa in (("vs_rhp", "R", min_pa_vs_rhp), ("vs_lhp", "L", min_pa_vs_lhp)):
        sub = pa[pa["p_throws"] == hand]
        ids, skill = _eb_leaderboard(sub, "batter", split_label, pm, min_pa, strength)
        print(f"  batter {split_label}: {len(ids)} qualifying")
        if len(ids) == 0:
            continue
        batter_rows.append((split_label, ids, skill))

    # --- Pitcher leaderboards: filter to actual SP / RP via classify_roles ---
    role_series = classify_roles(pa)  # index = pitcher_id, value in {"SP","RP"}
    pitcher_teams = _pitcher_team_lookup(pa)
    pitcher_rows: list[dict] = []
    for split_label, role in (("sp", "SP"), ("rp", "RP")):
        eligible_ids = set(role_series[role_series == role].index.astype(int))
        sub = pa[pa["pitcher"].isin(eligible_ids)]
        min_bf = min_bf_sp if role == "SP" else min_bf_rp
        ids, skill = _eb_leaderboard(sub, "pitcher", split_label, pm, min_bf, strength)
        print(f"  pitcher {split_label}: {len(ids)} qualifying (of {len(eligible_ids)} {role}-classified)")
        if len(ids) == 0:
            continue
        pitcher_rows.append((split_label, ids, skill))

    # --- Resolve names in one batch ---
    all_ids: set[int] = set()
    for split_label, ids, skill in batter_rows + pitcher_rows:
        order = np.argsort(skill)
        all_ids.update(int(ids[i]) for i in order[:TOP_N])
        all_ids.update(int(ids[i]) for i in order[-TOP_N:])
    print(f"[posterior_summaries] fetching {len(all_ids)} names from MLB Stats API")
    names = _fetch_names(sorted(all_ids))

    skill_rows: list[dict] = []
    for split_label, ids, skill in batter_rows:
        skill_rows += _build_leaderboard_rows(refit_date, "batter", split_label, ids, skill, names, batter_teams)
    for split_label, ids, skill in pitcher_rows:
        skill_rows += _build_leaderboard_rows(refit_date, "pitcher", split_label, ids, skill, names, pitcher_teams)

    # --- Sigma summaries straight from posteriors (Variance Decomposition chart) ---
    bat = az.from_netcdf(POSTERIORS_DIR / "batter_skill.nc")
    pit = az.from_netcdf(POSTERIORS_DIR / "pitcher_skill.nc")
    park = az.from_netcdf(POSTERIORS_DIR / "park_effects.nc")

    sigma_rows = [
        {"refit_date": refit_date, "sigma_name": "sigma_batter", **_sigma_summary(bat, "sigma_batter")},
        {"refit_date": refit_date, "sigma_name": "sigma_platoon", **_sigma_summary(bat, "sigma_platoon")},
        {"refit_date": refit_date, "sigma_name": "sigma_pitcher", **_sigma_summary(pit, "sigma_pitcher")},
    ]
    park_log_std = park.posterior["park_log"].std(("venue",)).mean(("chain", "draw")).values
    sigma_rows.append({
        "refit_date": refit_date,
        "sigma_name": "sigma_park",
        "mean": round(float(park_log_std), 5),
        "p10": None,
        "p90": None,
    })

    print(f"[posterior_summaries] writing {len(skill_rows)} skill rows + {len(sigma_rows)} sigma rows")
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM posterior_skills WHERE refit_date = :d"), {"d": refit_date})
        if skill_rows:
            pd.DataFrame(skill_rows).to_sql("posterior_skills", con=conn, if_exists="append", index=False)
        conn.execute(text("DELETE FROM posterior_sigmas WHERE refit_date = :d"), {"d": refit_date})
        if sigma_rows:
            pd.DataFrame(sigma_rows).to_sql("posterior_sigmas", con=conn, if_exists="append", index=False)

    print("[posterior_summaries] done")


if __name__ == "__main__":
    main()
