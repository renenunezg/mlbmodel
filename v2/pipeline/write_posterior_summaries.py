"""Extract top-10 / bottom-10 skill leaderboards + sigma summaries from
posterior traces, write to Supabase tables `posterior_skills` and
`posterior_sigmas` so the frontend can render the v2 Diagnostics tab.

Skill score = expected wOBA = sum_k softmax(intercept + offset)[k] * WOBA_WEIGHTS[k]
For batters: split by vs_lhp / vs_rhp.
For pitchers: split by SP / RP role.
Higher = better for batters, lower = better for pitchers (allows less).

Usage:
    python -m v2.pipeline.write_posterior_summaries
"""
from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import statsapi
from sqlalchemy import text

from backend.db import engine
from v2.bayesian._common import POSTERIORS_DIR, WOBA_WEIGHTS
from v2.data.pa_dataset import OUTCOMES

REF_IDX = OUTCOMES.index("OUT")
NON_REF = [i for i in range(len(OUTCOMES)) if i != REF_IDX]
WOBA_FREE = np.array([WOBA_WEIGHTS[OUTCOMES[i]] for i in NON_REF], dtype=np.float64)

TOP_N = 10
SIGMA_NAMES = ["sigma_batter", "sigma_platoon", "sigma_pitcher", "sigma_park"]
CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"


def _xwoba_from_logits(logit_free: np.ndarray) -> np.ndarray:
    """Convert (..., K_FREE) logits (OUT reference, logit=0) to expected wOBA."""
    # Append a 0 for OUT, softmax over 8, weight by wOBA.
    shape = logit_free.shape[:-1]
    full = np.zeros(shape + (len(OUTCOMES),), dtype=np.float64)
    full[..., NON_REF] = logit_free
    full[..., REF_IDX] = 0.0
    z = full - full.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    woba_full = np.zeros(len(OUTCOMES))
    for i, k in enumerate(NON_REF):
        woba_full[k] = WOBA_FREE[i]
    return (p * woba_full).sum(axis=-1)


def _batter_xwoba(intercept: np.ndarray, beta_main: np.ndarray, beta_platoon: np.ndarray):
    """Returns (xwoba_vs_rhp, xwoba_vs_lhp) arrays of shape (n_batters,)."""
    vs_rhp = _xwoba_from_logits(intercept[None, :] + beta_main)
    vs_lhp = _xwoba_from_logits(intercept[None, :] + beta_main + beta_platoon)
    return vs_rhp, vs_lhp


def _pitcher_xwoba(intercept: np.ndarray, beta_pitcher: np.ndarray):
    """beta_pitcher shape (n_pitchers, 2, K_FREE). role 0=SP, 1=RP.
    Returns (xwoba_sp, xwoba_rp) arrays of shape (n_pitchers,).
    """
    sp = _xwoba_from_logits(intercept[None, :] + beta_pitcher[:, 0, :])
    rp = _xwoba_from_logits(intercept[None, :] + beta_pitcher[:, 1, :])
    return sp, rp


def _latest_team_for_actors(actor_kind: str, ids: np.ndarray) -> dict[int, str]:
    """Read 2026 statcast cache and derive most recent team per actor id.

    actor_kind ∈ {'batter', 'pitcher'}. Returns {id: team_abbr}.
    """
    path = CACHE_DIR / "statcast_2026.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(
        path,
        columns=[actor_kind, "home_team", "away_team", "inning_topbot", "game_date"],
    )
    df = df[df[actor_kind].notna()].copy()
    if actor_kind == "batter":
        df["team"] = np.where(df["inning_topbot"] == "Top", df["away_team"], df["home_team"])
    else:
        df["team"] = np.where(df["inning_topbot"] == "Top", df["home_team"], df["away_team"])
    df = df[df[actor_kind].isin(ids)]
    df = df.sort_values("game_date").drop_duplicates(actor_kind, keep="last")
    return dict(zip(df[actor_kind].astype(np.int64), df["team"].astype(str)))


def _fetch_names(ids: list[int], chunk: int = 100) -> dict[int, str]:
    """Batch lookup full names via MLB Stats API. ~17 calls for ~1700 actors."""
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
    """Top + bottom N for one (actor_type, split_label) category."""
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


def _sigma_summary(idata: az.InferenceData, var: str) -> dict[str, float]:
    """Posterior mean of `var`, plus 10th/90th percentiles of its overall magnitude.

    The sigma vectors in the model are per-outcome (K_FREE wide). We summarize
    with the average across outcomes so the result is one scalar per sigma.
    For pitcher, also averages across the 2 roles.
    """
    arr = idata.posterior[var].values  # shape (chain, draw, *)
    flat = arr.mean(axis=tuple(range(2, arr.ndim)))  # (chain, draw)
    return {
        "mean": round(float(flat.mean()), 5),
        "p10": round(float(np.quantile(flat, 0.10)), 5),
        "p90": round(float(np.quantile(flat, 0.90)), 5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit-date", default=str(date.today()))
    args = ap.parse_args()
    refit_date = pd.Timestamp(args.refit_date).date()

    print(f"[posterior_summaries] refit_date={refit_date}")

    bat = az.from_netcdf(POSTERIORS_DIR / "batter_skill.nc")
    pit = az.from_netcdf(POSTERIORS_DIR / "pitcher_skill.nc")
    park = az.from_netcdf(POSTERIORS_DIR / "park_effects.nc")

    # --- Skills ---
    intercept_b = bat.posterior["intercept"].mean(("chain", "draw")).values
    sigma_batter = bat.posterior["sigma_batter"].mean(("chain", "draw")).values
    z_batter = bat.posterior["z_batter"].mean(("chain", "draw")).values
    sigma_platoon = bat.posterior["sigma_platoon"].mean(("chain", "draw")).values
    z_platoon = bat.posterior["z_platoon"].mean(("chain", "draw")).values
    batter_ids = bat.posterior["batter"].values.astype(np.int64)
    beta_main = sigma_batter[None, :] * z_batter
    beta_platoon = sigma_platoon[None, :] * z_platoon
    xwoba_rhp, xwoba_lhp = _batter_xwoba(intercept_b, beta_main, beta_platoon)
    print(f"[posterior_summaries] {len(batter_ids)} batters; rhp range [{xwoba_rhp.min():.3f}, {xwoba_rhp.max():.3f}]")

    intercept_p = pit.posterior["intercept"].mean(("chain", "draw")).values
    sigma_pitcher = pit.posterior["sigma_pitcher"].mean(("chain", "draw")).values  # (2, K_FREE)
    z_pitcher = pit.posterior["z_pitcher"].mean(("chain", "draw")).values  # (n_p, K_FREE)
    pitcher_ids = pit.posterior["pitcher"].values.astype(np.int64)
    beta_pitcher = sigma_pitcher[None, :, :] * z_pitcher[:, None, :]  # (n_p, 2, K_FREE)
    xwoba_sp, xwoba_rp = _pitcher_xwoba(intercept_p, beta_pitcher)
    print(f"[posterior_summaries] {len(pitcher_ids)} pitchers; sp range [{xwoba_sp.min():.3f}, {xwoba_sp.max():.3f}]")

    # Restrict pitcher leaderboards to pitchers with realistic role usage. Both
    # SP and RP offsets are estimated for every pitcher (shared hierarchy), but
    # the "best SP by xwoba_sp" only means something if they actually start.
    # Use the latest team-derivation cache to filter by whether they appeared
    # in inning 1 (a starter proxy) - we'll filter the leaderboard rows after.

    # --- Lookups ---
    print("[posterior_summaries] resolving team + name lookups...")
    batter_teams = _latest_team_for_actors("batter", batter_ids)
    pitcher_teams = _latest_team_for_actors("pitcher", pitcher_ids)

    # Collect all ids that will appear in any top/bottom set (much smaller than fetching all)
    all_skill_ids: set[int] = set()
    for ids, skill in (
        (batter_ids, xwoba_rhp),
        (batter_ids, xwoba_lhp),
        (pitcher_ids, xwoba_sp),
        (pitcher_ids, xwoba_rp),
    ):
        order = np.argsort(skill)
        all_skill_ids.update(int(ids[i]) for i in order[:TOP_N])
        all_skill_ids.update(int(ids[i]) for i in order[-TOP_N:])
    print(f"[posterior_summaries] fetching {len(all_skill_ids)} names from MLB Stats API")
    names = _fetch_names(sorted(all_skill_ids))

    # --- Build leaderboard rows ---
    rows: list[dict] = []
    rows += _build_leaderboard_rows(refit_date, "batter", "vs_rhp", batter_ids, xwoba_rhp, names, batter_teams)
    rows += _build_leaderboard_rows(refit_date, "batter", "vs_lhp", batter_ids, xwoba_lhp, names, batter_teams)
    rows += _build_leaderboard_rows(refit_date, "pitcher", "sp", pitcher_ids, xwoba_sp, names, pitcher_teams)
    rows += _build_leaderboard_rows(refit_date, "pitcher", "rp", pitcher_ids, xwoba_rp, names, pitcher_teams)

    # --- Sigma summaries ---
    sigma_rows = []
    sigma_rows.append({"refit_date": refit_date, "sigma_name": "sigma_batter", **_sigma_summary(bat, "sigma_batter")})
    sigma_rows.append({"refit_date": refit_date, "sigma_name": "sigma_platoon", **_sigma_summary(bat, "sigma_platoon")})
    sigma_rows.append({"refit_date": refit_date, "sigma_name": "sigma_pitcher", **_sigma_summary(pit, "sigma_pitcher")})
    # Park doesn't have a sigma in the trace (sigma_resid is fixed at 0.36),
    # so we summarize the park_log magnitude directly as a stand-in.
    park_log_std = park.posterior["park_log"].std(("venue",)).mean(("chain", "draw")).values
    sigma_rows.append({
        "refit_date": refit_date,
        "sigma_name": "sigma_park",
        "mean": round(float(park_log_std), 5),
        "p10": None,
        "p90": None,
    })

    # --- Write to Supabase ---
    print(f"[posterior_summaries] writing {len(rows)} skill rows + {len(sigma_rows)} sigma rows")
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM posterior_skills WHERE refit_date = :d"), {"d": refit_date})
        if rows:
            pd.DataFrame(rows).to_sql("posterior_skills", con=conn, if_exists="append", index=False)
        conn.execute(text("DELETE FROM posterior_sigmas WHERE refit_date = :d"), {"d": refit_date})
        if sigma_rows:
            pd.DataFrame(sigma_rows).to_sql("posterior_sigmas", con=conn, if_exists="append", index=False)

    print(f"[posterior_summaries] done")


if __name__ == "__main__":
    main()
