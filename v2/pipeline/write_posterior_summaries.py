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
from v2.bayesian._common import POSTERIORS_DIR
from v2.bayesian.pitcher_skill import classify_roles
from v2.data.pa_dataset import EVENT_TO_OUTCOME, NON_PA_EVENTS

TOP_N = 10
SIGMA_NAMES = ["sigma_batter", "sigma_platoon", "sigma_pitcher", "sigma_park"]
CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"
WINDOW_DAYS_DEFAULT = 10

# Minimum PA / batters faced for a player to be eligible for the leaderboard.
# These scale with the window length passed at the CLI. vs_lhp gets a lower
# threshold because batters see roughly a third as many lefties.
MIN_PA_BATTER_VS_RHP_PER_DAY = 1.5  # ~15 PA across 10 days
MIN_PA_BATTER_VS_LHP_PER_DAY = 0.7  # ~7 PA across 10 days
MIN_BF_SP_PER_DAY = 2.0             # ~20 BF across 10 days ~ one full start
MIN_BF_RP_PER_DAY = 1.0             # ~10 BF across 10 days ~ 5 outings

PARQUET_COLS = [
    "game_pk", "game_date", "batter", "pitcher",
    "p_throws", "stand", "events",
    "home_team", "away_team", "inning", "inning_topbot",
    "estimated_woba_using_speedangle", "woba_value", "woba_denom",
]


def _load_window_pa(window_start: date, window_end: date) -> pd.DataFrame:
    """Load PA-terminating rows from cached parquet within [start, end]."""
    year = window_end.year
    path = CACHE_DIR / f"statcast_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    df = pd.read_parquet(path, columns=PARQUET_COLS)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df = df[(df["game_date"] >= window_start) & (df["game_date"] <= window_end)]
    df = df[df["events"].notna() & ~df["events"].isin(NON_PA_EVENTS)]
    df = df[df["events"].isin(EVENT_TO_OUTCOME)]

    # Per-PA xwOBA value and denominator. For balls in play we trust statcast's
    # speedangle estimate. For K/BB/HBP it's not populated, so we fall back to
    # the realized woba_value (constants 0 / 0.69 / 0.72). Other events with
    # woba_denom NaN are excluded from the average (sac bunts, catcher_interf).
    sa = df["estimated_woba_using_speedangle"].astype("float64")
    wv = df["woba_value"].astype("float64")
    df["xwoba_value"] = np.where(sa.notna(), sa, wv)
    df["xwoba_denom"] = df["woba_denom"].astype("float64")
    df = df.dropna(subset=["xwoba_value", "xwoba_denom"])
    df = df[df["xwoba_denom"] > 0]

    df["batter"] = df["batter"].astype("int64")
    df["pitcher"] = df["pitcher"].astype("int64")
    return df.reset_index(drop=True)


def _player_xwoba(group: pd.DataFrame) -> float:
    return float(group["xwoba_value"].sum() / group["xwoba_denom"].sum())


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


def _per_actor_xwoba(pa: pd.DataFrame, key: str, min_n: int) -> pd.DataFrame:
    """Group by `key`, return DataFrame with columns [key, xwoba, n] filtered to n >= min_n."""
    g = pa.groupby(key, sort=False).agg(
        value_sum=("xwoba_value", "sum"),
        denom_sum=("xwoba_denom", "sum"),
        n=("xwoba_denom", "size"),
    )
    g["xwoba"] = g["value_sum"] / g["denom_sum"]
    g = g[g["n"] >= min_n].reset_index()
    return g[[key, "xwoba", "n"]]


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
    args = ap.parse_args()

    refit_date = pd.Timestamp(args.refit_date).date()
    window_days = args.window_days
    window_end = refit_date
    window_start = window_end - timedelta(days=window_days - 1)
    min_pa_vs_rhp = max(1, int(round(MIN_PA_BATTER_VS_RHP_PER_DAY * window_days)))
    min_pa_vs_lhp = max(1, int(round(MIN_PA_BATTER_VS_LHP_PER_DAY * window_days)))
    min_bf_sp = max(1, int(round(MIN_BF_SP_PER_DAY * window_days)))
    min_bf_rp = max(1, int(round(MIN_BF_RP_PER_DAY * window_days)))

    print(f"[posterior_summaries] refit_date={refit_date}  window={window_start}..{window_end}")
    print(f"[posterior_summaries] min PA: vs_rhp={min_pa_vs_rhp}  vs_lhp={min_pa_vs_lhp}  sp={min_bf_sp}  rp={min_bf_rp}")

    pa = _load_window_pa(window_start, window_end)
    print(f"[posterior_summaries] {len(pa):,} PAs in window")

    # --- Batter leaderboards: vs_rhp and vs_lhp ---
    batter_rows: list[dict] = []
    batter_teams = _batter_team_lookup(pa)
    for split_label, hand, min_pa in (("vs_rhp", "R", min_pa_vs_rhp), ("vs_lhp", "L", min_pa_vs_lhp)):
        sub = pa[pa["p_throws"] == hand]
        agg = _per_actor_xwoba(sub, "batter", min_pa)
        print(f"  batter {split_label}: {len(agg)} qualifying")
        if agg.empty:
            continue
        ids = agg["batter"].to_numpy(dtype=np.int64)
        skill = agg["xwoba"].to_numpy(dtype=np.float64)
        batter_rows.append((split_label, ids, skill))

    # --- Pitcher leaderboards: filter to actual SP / RP via classify_roles ---
    role_series = classify_roles(pa)  # index = pitcher_id, value in {"SP","RP"}
    pitcher_teams = _pitcher_team_lookup(pa)
    pitcher_rows: list[dict] = []
    for split_label, role in (("sp", "SP"), ("rp", "RP")):
        eligible_ids = set(role_series[role_series == role].index.astype(int))
        sub = pa[pa["pitcher"].isin(eligible_ids)]
        min_bf = min_bf_sp if role == "SP" else min_bf_rp
        agg = _per_actor_xwoba(sub, "pitcher", min_bf)
        print(f"  pitcher {split_label}: {len(agg)} qualifying (of {len(eligible_ids)} {role}-classified)")
        if agg.empty:
            continue
        ids = agg["pitcher"].to_numpy(dtype=np.int64)
        skill = agg["xwoba"].to_numpy(dtype=np.float64)
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
