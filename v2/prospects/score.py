"""Nightly prospect scoring: the published model applied to every rookie-eligible player, then the three tables.

Reads the feed tables for the scoring window (this season and the two before), hydrates bio and
roster history, builds the snapshot as of the date, predicts per pool, maps peak WAR onto the MLB
grade scale, prices the control years, explains each grade through trait contributions, and
publishes. Rows under the sample floor are published without a grade (``grade_reason`` set).
A pool that failed the publish gate at fit time is published ungraded with ``model_not_validated``.

Usage:
    python -m v2.prospects.score --as-of 2026-10-22 [--no-write] [--model path.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import date
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from backend.log import setup_logging
from v2.players.model import ValueModel
from v2.prospects import dataset, events, features, fit, identity, value
from v2.prospects.feeds import CACHE_DIR
from v2.prospects.model import POOLS, ProspectModel, load_latest_from_db
from v2.prospects.publish import publish

log = logging.getLogger(__name__)

HEADSHOT = "https://img.mlbstatic.com/mlb-photos/image/upload/w_213,q_auto:best/v1/people/{player_id}/headshot/67/current"
STATSAPI = "https://statsapi.mlb.com/api/v1"
LAST_SCORED = CACHE_DIR / "last_scored.json"
SEASON_START_MONTH = 4
RISK_LABELS = ("safe", "medium", "high", "extreme")
ETA_BAND = 1
TRACKED_INPUTS = set(features.HITTER_TRACKED) | set(features.PITCHER_TRACKED) | {"tracked_bip", "tracked_pitches"}
TRAITS = {
    "bat": [
        ("contact", "batting", "Contact", ["contact_rate", "contact_rate_rel", "swstr_rate", "swstr_rate_rel", "k_rate", "k_rate_rel"]),
        ("discipline", "batting", "Discipline", ["bb_rate", "bb_rate_rel", "swing_rate", "swing_rate_rel", "csw_rate", "csw_rate_rel"]),
        ("power", "batting", "Power", ["iso", "iso_rel", "hr_rate", "hr_rate_rel", "xbh_rate", "xbh_rate_rel", "hard_share", "hard_share_rel",
                                      "ev90", "ev_mean", "hard_hit", "tracked_bip"]),
        ("batted_ball", "batting", "Batted ball", ["gb_share", "gb_share_rel", "ld_share", "ld_share_rel", "fb_share", "fb_share_rel",
                                                  "pu_share", "pu_share_rel", "soft_share", "soft_share_rel", "babip", "babip_rel", "la_mean"]),
        ("speed", "running", "Speed", ["sb_rate", "sb_rate_rel", "cs_rate", "cs_rate_rel"]),
        ("defense", "fielding", "Defense", [f"pos_{c}" for c in events.POSITION_CODES]),
        ("age_level", "context", "Age vs level", ["age", "age_rel"]),
        ("track_record", "context", "Track record", ["top_level_rank", "n_levels", "promoted", "top_level_share", "career_max_level_rank",
                                                    "season_sample", "sample_w", "years_pro"]),
        ("pedigree", "context", "Pedigree", ["draft_round", "log_draft_pick", "drafted", "school_high_school", "school_college",
                                            "school_international", "school_undrafted", "bats_left", "bats_switch"]),
    ],
    "pit": [
        ("bat_missing", "pitching", "Bat-missing", ["k_rate", "k_rate_rel", "swstr_rate", "swstr_rate_rel", "csw_rate", "csw_rate_rel",
                                                  "contact_rate", "contact_rate_rel", "whiff_tracked"]),
        ("control", "pitching", "Control", ["bb_rate", "bb_rate_rel"]),
        ("ground_balls", "pitching", "Ground balls", ["gb_share", "gb_share_rel", "fb_share", "fb_share_rel"]),
        ("contact_quality", "pitching", "Contact quality", ["hard_share", "hard_share_rel", "soft_share", "soft_share_rel", "babip", "babip_rel",
                                                          "hr_rate", "hr_rate_rel", "ev_mean_allowed", "hard_hit_allowed"]),
        ("stuff", "pitching", "Stuff", ["fb_velo", "fb_velo_max", "fb_ivb", "fb_hb", "fb_spin", "extension", "tracked_pitches"]),
        ("role", "usage", "Role", ["start_share", "start_share_rel", "outs_per_game", "outs_per_game_rel"]),
        ("age_level", "context", "Age vs level", ["age", "age_rel"]),
        ("track_record", "context", "Track record", ["top_level_rank", "n_levels", "promoted", "top_level_share", "career_max_level_rank",
                                                    "season_sample", "sample_w", "years_pro"]),
        ("pedigree", "context", "Pedigree", ["draft_round", "log_draft_pick", "drafted", "school_high_school", "school_college",
                                            "school_international", "school_undrafted", "throws_left"]),
    ],
}


def scoring_season(as_of: date) -> int:
    return as_of.year if as_of.month >= SEASON_START_MONTH else as_of.year - 1


@cache
def team_names(season: int) -> dict[int, tuple[str, str]]:
    """Team id -> (name, abbreviation) for MLB and every affiliated level in a season."""
    out = {}
    for sport_id in (1, 11, 12, 13, 14, 15, 16, 17):
        try:
            payload = requests.get(f"{STATSAPI}/teams", params={"sportId": sport_id, "season": season}, timeout=60).json()
        except requests.RequestException:
            continue
        for t in payload.get("teams", []):
            out[t["id"]] = (t.get("name"), t.get("abbreviation"))
    return out


def grade_band(sample: pd.Series, floor: pd.Series) -> pd.Series:
    """Half-width of the grade band from the sample relative to the floor: 10 × sqrt(1 − n / (n + floor))."""
    r = sample / (sample + floor)
    return 10.0 * np.sqrt((1 - r).clip(lower=0))


def percentile_within(values: pd.Series, mask: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=values.index)
    ranked = values[mask].rank(pct=True) * 100
    out.loc[ranked.index] = ranked
    return out.round(1)


def risk_label(spread: pd.Series, cuts: list[float]) -> pd.Series:
    idx = np.searchsorted(np.array(cuts), spread.to_numpy(float), side="right")
    return pd.Series([RISK_LABELS[min(i, 3)] for i in idx], index=spread.index).where(spread.notna())


def _traits_for_pool(frame: pd.DataFrame, pool: str, contribs: pd.DataFrame, graded: pd.Series, model: ProspectModel) -> list[dict]:
    """Trait rows: each trait's contribution to peak WAR, graded within the pool, with its inputs."""
    rows = []
    names = model.feature_names[pool]
    pct = {n: percentile_within(frame[n].astype(float), graded) for n in names if n in frame}
    for trait, panel, label, inputs in TRAITS[pool]:
        cols = [c for c in inputs if c in contribs.columns]
        contribution = contribs[cols].sum(axis=1)
        ref = contribution[graded]
        sd = ref.std()
        grade = (50 + 10 * (contribution - ref.mean()) / sd).clip(20, 80) if sd > 0 else pd.Series(50.0, index=frame.index)
        tracked = all(c in TRACKED_INPUTS for c in cols)
        for pid in frame.index:
            if not graded[pid]:
                continue
            available = [c for c in cols if pd.notna(frame.at[pid, c])]
            tracked_missing = tracked and not available
            rows.append({
                "player_id": int(frame.at[pid, "player_id"]), "trait": trait, "panel": panel,
                "grade": None if tracked_missing else round(float(grade[pid]), 1),
                "grade_low": None, "grade_high": None,
                "percentile": None if tracked_missing else round(float(percentile_within(contribution, graded)[pid]), 1),
                "value": round(float(contribution[pid]), 3), "in_grade": not tracked_missing,
                "kind": "tracked" if tracked else "measured", "sample": int(frame.at[pid, "sample"]),
                "inputs": [{"input": c, "label": label if len(cols) == 1 else c, "value": None if pd.isna(frame.at[pid, c]) else round(float(frame.at[pid, c]), 4),
                            "percentile": None if pd.isna(pct[c][pid]) else round(float(pct[c][pid]), 1),
                            "contribution": round(float(contribs.at[pid, c]), 4)} for c in cols],
                "as_of": frame.at[pid, "as_of"].date() if hasattr(frame.at[pid, "as_of"], "date") else frame.at[pid, "as_of"],
                "model_version": model.model_version,
            })
    return rows


def score(as_of: date, model: ProspectModel) -> tuple[pd.DataFrame, pd.DataFrame]:
    season = scoring_season(as_of)
    seasons = list(range(season - dataset.WINDOW + 1, season + 1))
    tables = dataset.load_tables(seasons)
    active = set(tables[0].loc[tables[0]["season"] >= season - 1, "player_id"]) | set(tables[1].loc[tables[1]["season"] >= season - 1, "player_id"])
    ids = sorted(set(tables[0]["player_id"]) | set(tables[1]["player_id"]))
    bio, roster = identity.people(ids, refresh_ids=active)
    baselines = (pd.DataFrame(model.level_age["bat"]), pd.DataFrame(model.level_age["pit"]))
    frame = dataset.snapshot(season, tables, bio, roster, baselines, as_of)
    frame = frame[frame["player_id"].isin(active) & frame["rookie_eligible"] & frame["in_age_range"]].reset_index(drop=True)
    log.info(f"score[{as_of}]: {len(frame):,} rookie-eligible players, {int(frame['gradeable'].sum()):,} above the floor")
    price = ValueModel(price=dict(model.price)).price_model()
    b = bio.set_index("player_id")
    names = team_names(season)
    mlb_ids = identity.mlb_team_ids()
    forty = identity.on_forty_man(roster, as_of, mlb_ids)
    values = []
    traits = []
    for pool in POOLS:
        rows = frame[frame["role"] == pool].copy()
        if rows.empty:
            continue
        validated = bool(model.validated.get(pool, False))
        pred, contribs = model.predict(rows, pool, contributions=True)
        graded = rows["gradeable"] & validated
        ref = model.grade_reference[pool]
        rows["peak_war"] = pred["peak_war"].where(graded)
        rows["peak_grade"] = fit.to_grade(pred["peak_war"], ref).where(graded).round(1)
        band = grade_band(rows["sample"], rows["floor"])
        rows["peak_grade_low"] = (rows["peak_grade"] - band).clip(lower=20).round(1)
        rows["peak_grade_high"] = (rows["peak_grade"] + band).clip(upper=80).round(1)
        rows["peak_grade_percentile"] = percentile_within(rows["peak_grade"], graded)
        rows["grade_reason"] = np.where(graded, None, np.where(validated, "insufficient_sample", "model_not_validated"))
        rows["p_regular"] = pred["p_regular"].where(graded).round(3)
        rows["p_mlb"] = pred["p_mlb"].where(graded).round(3)
        rows["expected_control_war"] = pred["control_war"].where(graded).round(2)
        roles = rows.apply(lambda r: value.price_role(pool, r.get("start_share", np.nan)), axis=1)
        tv = value.trade_values(pred, roles, price, model.arb_fractions, season)
        for c in ("trade_value_musd", "trade_value_low", "trade_value_high", "market_value_musd", "eta_season"):
            rows[c] = tv[c].where(graded)
        rows["eta_low"] = rows["eta_season"] - ETA_BAND
        rows["eta_high"] = rows["eta_season"] + ETA_BAND
        rows["risk"] = risk_label(fit.risk_score(pred), model.risk_cutpoints[pool]).where(graded)
        rows["outlook"] = [value.outlook(p.control_war, p.p_mlb, p.eta_years, a, r, price, model.arb_fractions, season) if g else []
                           for p, a, r, g in zip(pred.itertuples(index=False), rows["age"], roles, graded)]
        rows["tip"] =[_tip(rows.loc[i], contribs["peak_war"].loc[i], contribs["regular"].loc[i], model, pool, price, season) if graded[i] else {}
                       for i in rows.index]
        traits.extend(_traits_for_pool(rows, pool, contribs["peak_war"], graded, model))
        values.append(rows)
    out = pd.concat(values, ignore_index=True)
    out["player_name"] = out["player_id"].map(b["full_name"])
    out["org_id"] = out["org_id"].astype("Int64")
    out["org"] = out["org_id"].map(lambda i: names.get(int(i), (None, None))[1] if pd.notna(i) else None)
    out["team"] = out["team_id"].map(lambda i: names.get(int(i), (None, None))[0] if pd.notna(i) else None)
    out["level"] = out["top_level"]
    out["source"] = "milb"
    out["position_group"] = np.where(out["role"].eq("pit"), np.where(out.get("start_share", pd.Series(0, index=out.index)).fillna(0) >= 0.5, "SP", "RP"),
                                     out["primary_pos"].fillna("DH"))
    out["secondary_positions"] = [[] for _ in range(len(out))]
    out["bats"] = out["player_id"].map(b["bats"])
    out["throws"] = out["player_id"].map(b["throws"])
    out["headshot_url"] = out["player_id"].map(lambda i: HEADSHOT.format(player_id=int(i)))
    out["model_version"] = model.model_version
    out["season"] = season
    out["draft_year"] = out["player_id"].map(b["draft_year"])
    out["draft_round"] = out["player_id"].map(b["draft_round"])
    out["draft_pick"] = out["player_id"].map(b["draft_pick"])
    out["signing_bonus_musd"] = out["player_id"].map(b["signing_bonus_musd"])
    school = out["player_id"].map(b["school_type"])
    out["acquired"] = np.where(out["draft_year"].notna(), "draft", np.where(school.eq("international"), "international", "undrafted"))
    out["on_40_man"] = out["player_id"].map(forty).fillna(False).astype(bool)
    out["mlb_pa_or_ip"] = np.where(out["role"].eq("pit"), out["mlb_ip"], out["mlb_ab"])
    out["service_days"] = out["mlb_days"].round().astype("Int64")
    out["injured"] = False
    out["as_of"] = as_of
    out["age"] = out["age"].round(1)
    return out, pd.DataFrame(traits)


def _tip(row: pd.Series, peak_contrib: pd.Series, regular_contrib: pd.Series, model: ProspectModel, pool: str, price, season: int) -> dict:
    ref = model.grade_reference[pool]
    top_peak = peak_contrib.abs().sort_values(ascending=False).head(8).index
    return {
        "peak_grade": {"peak_war": round(float(row["peak_war"]), 2), "reference_mean": round(ref["mean"], 2), "reference_sd": round(ref["sd"], 2),
                       "reference": ref.get("source"), "contributions": [{"input": c, "contribution": round(float(peak_contrib[c]), 3),
                                                                           "value": None if pd.isna(row.get(c)) else round(float(row[c]), 4)} for c in top_peak]},
        "p_regular": {"value": float(row["p_regular"]), "definition": "a season of 2.0 WAR or more within seven seasons",
                      "contributions": [{"input": c, "contribution": round(float(regular_contrib[c]), 3)}
                                        for c in regular_contrib.abs().sort_values(ascending=False).head(6).index]},
        "trade_value": {"expected_control_war": float(row["expected_control_war"]), "eta_season": None if pd.isna(row["eta_season"]) else int(row["eta_season"]),
                        "price_musd": round(price.price_musd, 2), "discount_per_year": value.DISCOUNT_PER_YEAR, "control_seasons": value.CONTROL_SEASONS,
                        "pre_arb_seasons": value.PRE_ARB_SEASONS, "arb_fractions": {str(k): round(v, 3) for k, v in model.arb_fractions.items()}},
        "sample": {"window": float(row["sample"]), "season": float(row["season_sample"]), "floor": float(row["floor"]),
                   "tracked_share": None if pd.isna(row.get("tracked_share")) else round(float(row["tracked_share"]), 3)},
    }


def fingerprint(as_of: date, frame: pd.DataFrame, model: ProspectModel) -> str:
    h = hashlib.sha256()
    h.update(str(as_of).encode())
    h.update(model.fitted_at.encode())
    h.update(pd.util.hash_pandas_object(frame[["player_id", "sample", "season_sample", "top_level"]].astype(str), index=False).to_numpy().tobytes())
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument("--model", type=str, default=None, help="model JSON on disk instead of the newest database row")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--force", action="store_true", help="publish even when the fingerprint is unchanged")
    args = parser.parse_args()
    setup_logging()
    model = ProspectModel.load(Path(args.model)) if args.model else load_latest_from_db()
    values, traits = score(args.as_of, model)
    log.info(f"score: {len(values):,} rows, {int(values['peak_grade'].notna().sum()):,} graded, {len(traits):,} trait rows")
    if args.no_write:
        out = CACHE_DIR / f"scored_{args.as_of}.parquet"
        values.drop(columns=["outlook", "tip"]).to_parquet(out, index=False)
        log.info(f"score: wrote {out}")
        return
    fp = fingerprint(args.as_of, values, model)
    if LAST_SCORED.exists() and not args.force and json.loads(LAST_SCORED.read_text()).get("fingerprint") == fp:
        log.info("score: inputs unchanged since the last publish; skipping")
        return
    publish(values, traits)
    LAST_SCORED.write_text(json.dumps({"fingerprint": fp, "as_of": str(args.as_of)}))


if __name__ == "__main__":
    main()
