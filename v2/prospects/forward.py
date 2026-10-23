"""Forward test: the pre-debut prospect grade against early MLB production, next to the baselines.

For every player whose first qualifying MLB season (100 plate appearances or 40 innings) came in
one of the debut seasons, the snapshot at the end of the season before the debut is scored with
the model fitted on cohorts that matured before any of them. The target is WAR over the debut
season and the one after (one season for the latest debuts). Baselines: the KATOH-style logistic
and linear fit, and the MLB Pipeline top-100 rank from the list published in the debut season,
compared on the intersection of players the list ranked.

Usage:
    python -m v2.prospects.forward --model v2/prospects/models/prospect_model_2017.json --debuts 2022 2023 2024 2025 2026
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backend.log import setup_logging
from v2.players.bio import CACHE_DIR as PLAYER_CACHE_DIR
from v2.prospects import dataset, fit, identity, lists, targets
from v2.prospects.feeds import CACHE_DIR
from v2.prospects.model import POOLS, ProspectModel

log = logging.getLogger(__name__)

RESULT_PATH = CACHE_DIR / "forward_evaluation.json"
EARLY_SEASONS = 2
PROJECTION_SYSTEMS = {"zips": ("zips", "zips"), "steamer": ("steamer", "steamer"), "the_bat": ("thebatx", "thebat")}


def projected_war(season: int, role: str) -> dict[str, pd.Series]:
    """Each cached system's preseason WAR projection for the season by MLBAM id, hitters or pitchers."""
    out = {}
    for system, (bat_type, pit_type) in PROJECTION_SYSTEMS.items():
        path = PLAYER_CACHE_DIR / f"fangraphs_projection_{bat_type if role == 'bat' else pit_type}_{role}_{season}.json"
        if not path.exists():
            continue
        rows = [r for r in json.loads(path.read_text())["rows"] if r.get("xMLBAMID") and r.get("WAR") is not None]
        if len(rows) < 100:  # a stub snapshot, not a leaderboard
            continue
        out[system] = pd.Series({int(r["xMLBAMID"]): float(r["WAR"]) for r in rows})
    return out


def debuts(seasons: list[int], last_season: int) -> pd.DataFrame:
    """player_id, debut season and early WAR for players whose first qualifying MLB season is in ``seasons``."""
    mlb = targets.mlb_seasons(last_season)
    first = mlb[mlb["qualifies"]].groupby("player_id")["season"].min().rename("debut").reset_index()
    first = first[first["debut"].isin(seasons)]
    war = mlb.set_index(["player_id", "season"])["war"]
    early = []
    for pid, debut in first.itertuples(index=False):
        span = [s for s in range(debut, debut + EARLY_SEASONS) if s <= last_season]
        early.append(float(sum(war.get((pid, s), 0.0) for s in span)))
    first["early_war"] = early
    first["early_seasons"] = first["debut"].map(lambda d: len([s for s in range(d, d + EARLY_SEASONS) if s <= last_season]))
    return first


def run(model: ProspectModel, debut_seasons: list[int], last_season: int) -> dict:
    snapshots = sorted({d - 1 for d in debut_seasons})
    data = dataset.build(snapshots)
    data = data[data["gradeable"]].copy()
    deb = debuts(debut_seasons, last_season)
    rows = data.merge(deb, left_on=["player_id"], right_on=["player_id"], how="inner")
    rows = rows[rows["season"] == rows["debut"] - 1].copy()
    log.info(f"forward: {len(rows):,} pre-debut snapshots for debuts {debut_seasons}")
    tables = dataset.load_tables(list(range(min(snapshots) - 1, max(snapshots) + 2)))
    active = pd.concat([tables[0][["player_id", "season"]], tables[1][["player_id", "season"]]]).drop_duplicates()
    bio, _ = identity.people(sorted(active["player_id"].unique()))
    ranked = lists.load(bio, active, [y for y in lists.available_years() if y in debut_seasons])
    ranked = ranked.dropna(subset=["player_id"])[["player_id", "list_year", "rank"]]
    result = {"model_version": model.model_version, "fitted_at": model.fitted_at, "debut_seasons": debut_seasons, "last_season": last_season,
              "evaluated_at": datetime.now(UTC).isoformat(), "pools": {}}
    for pool in POOLS:
        r = rows[rows["role"] == pool].copy()
        if r.empty:
            continue
        pred, _ = model.predict(r, pool)
        base = fit.predict_baseline(model.baseline[pool], r)
        r["model_peak"] = pred["peak_war"].to_numpy()
        r["model_p_regular"] = pred["p_regular"].to_numpy()
        r["baseline_peak"] = base["peak_war"].to_numpy()
        r = r.merge(ranked.rename(columns={"list_year": "debut"}), on=["player_id", "debut"], how="left")
        out = {"n": int(len(r)), "mean_early_war": float(r["early_war"].mean()),
               "spearman_model": fit.spearman(r["model_peak"], r["early_war"]),
               "spearman_baseline": fit.spearman(r["baseline_peak"], r["early_war"]),
               "spearman_p_regular": fit.spearman(r["model_p_regular"], r["early_war"])}
        listed = r[r["rank"].notna()]
        out["listed_n"] = int(len(listed))
        if len(listed) > 5:
            out["listed_spearman_model"] = fit.spearman(listed["model_peak"], listed["early_war"])
            out["listed_spearman_pipeline"] = fit.spearman(-listed["rank"], listed["early_war"])
            out["listed_spearman_baseline"] = fit.spearman(listed["baseline_peak"], listed["early_war"])
            out["model_vs_pipeline_rank_agreement"] = fit.spearman(listed["model_peak"], -listed["rank"])
        # Listed versus unlisted debuts: did the model see the unranked arrivals coming?
        unlisted = r[r["rank"].isna()]
        out["unlisted_mean_early_war"] = float(unlisted["early_war"].mean()) if len(unlisted) else None
        out["unlisted_top_decile_early_war"] = (float(unlisted.nlargest(max(len(unlisted) // 10, 1), "model_peak")["early_war"].mean())
                                                if len(unlisted) else None)
        # Preseason projection systems, where cached for the debut season: the same players, the same target.
        for debut in sorted(r["debut"].unique()):
            for system, war in projected_war(int(debut), pool).items():
                sub = r[r["debut"].eq(debut) & r["player_id"].isin(war.index)]
                if len(sub) < 10:
                    continue
                key = f"{system}_{int(debut)}"
                out[key] = {"n": int(len(sub)), "spearman_model": fit.spearman(sub["model_peak"], sub["early_war"]),
                            "spearman_system": fit.spearman(sub["player_id"].map(war), sub["early_war"]),
                            "spearman_baseline": fit.spearman(sub["baseline_peak"], sub["early_war"])}
        by_debut = r.groupby("debut").apply(lambda g: pd.Series({"n": len(g), "spearman_model": fit.spearman(g["model_peak"], g["early_war"]),
                                                                  "spearman_baseline": fit.spearman(g["baseline_peak"], g["early_war"])}))
        out["by_debut"] = {int(k): {kk: (None if vv != vv else float(vv)) for kk, vv in v.items()} for k, v in by_debut.iterrows()}
        result["pools"][pool] = out
        log.info(f"forward[{pool}]: n={out['n']} spearman model {out['spearman_model']:.3f} baseline {out['spearman_baseline']:.3f} "
                 f"listed n={out['listed_n']}" + (f" model {out['listed_spearman_model']:.3f} pipeline {out['listed_spearman_pipeline']:.3f}"
                                                   if "listed_spearman_model" in out else ""))
    RESULT_PATH.write_text(json.dumps(result, indent=1, default=lambda v: None if isinstance(v, float) and np.isnan(v) else float(v)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--debuts", type=int, nargs="+", default=[2022, 2023, 2024, 2025, 2026])
    parser.add_argument("--last-season", type=int, default=datetime.now(UTC).year)
    args = parser.parse_args()
    setup_logging()
    run(ProspectModel.load(Path(args.model)), args.debuts, args.last_season)


if __name__ == "__main__":
    main()
