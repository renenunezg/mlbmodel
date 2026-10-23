"""Nightly prospect scoring: the published model applied to every rookie-eligible player, then one table.

Reads the feed tables for the scoring window (this season and the two before), hydrates bio and
roster history for players new to the cache and refreshes the few whose eligibility can still move
(an open entry with an MLB club), builds the snapshot as of the date, predicts per pool on the rows
above the floor, maps peak WAR onto the MLB grade scale, prices the control years, summarizes the
traits, attaches per-pitch stuff for tracked pitchers (independent of the grade), and publishes when
the published columns changed. Rows under the floor are published
without a grade (``grade_reason`` set); a pool that failed the publish gate at fit time is published
ungraded with ``model_not_validated``.

Usage:
    python -m v2.prospects.score --as-of 2026-10-22 [--no-write] [--model path.json] [--stuff path.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from backend.log import setup_logging
from v2.players.bio import STARTER_SHARE
from v2.players.model import ValueModel
from v2.players.score import HEADSHOT
from v2.prospects import adjust, dataset, events, features, identity, stuff, value
from v2.prospects.feeds import CACHE_DIR, STATSAPI, fetch_json
from v2.prospects.model import POOLS, ProspectModel, load_latest_from_db, load_stuff_from_db
from v2.prospects.publish import VALUE_COLUMNS, publish

log = logging.getLogger(__name__)

LAST_SCORED = CACHE_DIR / "last_scored.json"
SEASON_START_MONTH = 4
TRACKED_INPUTS = set(features.HITTER_TRACKED) | set(features.PITCHER_TRACKED) | {"tracked_bip", "tracked_pitches"}
TRAITS = {
    "bat": [
        ("contact", ["contact_rate", "contact_rate_rel", "swstr_rate", "swstr_rate_rel", "k_rate", "k_rate_rel"]),
        ("discipline", ["bb_rate", "bb_rate_rel", "swing_rate", "swing_rate_rel", "csw_rate", "csw_rate_rel"]),
        ("power", ["iso", "iso_rel", "hr_rate", "hr_rate_rel", "xbh_rate", "xbh_rate_rel", "hard_share", "hard_share_rel",
                   "ev90", "ev_mean", "hard_hit", "tracked_bip"]),
        ("batted_ball", ["gb_share", "gb_share_rel", "ld_share", "ld_share_rel", "fb_share", "fb_share_rel", "pu_share", "pu_share_rel",
                         "soft_share", "soft_share_rel", "babip", "babip_rel", "la_mean"]),
        ("speed", ["sb_rate", "sb_rate_rel", "cs_rate", "cs_rate_rel"]),
        ("defense", [f"pos_{c}" for c in events.POSITION_CODES]),
        ("age_level", ["age", "age_rel"]),
        ("track_record", ["top_level_rank", "n_levels", "promoted", "top_level_share", "career_max_level_rank", "season_sample", "sample_w",
                          "years_pro"]),
        ("pedigree", ["draft_round", "log_draft_pick", "drafted", "school_high_school", "school_college", "school_international",
                      "school_undrafted", "bats_left", "bats_switch"]),
    ],
    "pit": [
        ("bat_missing", ["k_rate", "k_rate_rel", "swstr_rate", "swstr_rate_rel", "csw_rate", "csw_rate_rel", "contact_rate", "contact_rate_rel",
                         "whiff_tracked"]),
        ("control", ["bb_rate", "bb_rate_rel"]),
        ("ground_balls", ["gb_share", "gb_share_rel", "fb_share", "fb_share_rel"]),
        ("contact_quality", ["hard_share", "hard_share_rel", "soft_share", "soft_share_rel", "babip", "babip_rel", "hr_rate", "hr_rate_rel",
                             "ev_mean_allowed", "hard_hit_allowed"]),
        ("stuff", ["fb_velo", "fb_velo_max", "fb_ivb", "fb_hb", "fb_spin", "extension", "tracked_pitches"]),
        ("role", ["start_share", "start_share_rel", "outs_per_game", "outs_per_game_rel"]),
        ("age_level", ["age", "age_rel"]),
        ("track_record", ["top_level_rank", "n_levels", "promoted", "top_level_share", "career_max_level_rank", "season_sample", "sample_w",
                          "years_pro"]),
        ("pedigree", ["draft_round", "log_draft_pick", "drafted", "school_high_school", "school_college", "school_international",
                      "school_undrafted", "throws_left"]),
    ],
}


def scoring_season(as_of: date) -> int:
    return as_of.year if as_of.month >= SEASON_START_MONTH else as_of.year - 1


def team_names(season: int) -> dict[int, tuple[str, str]]:
    """Team id -> (name, abbreviation) for MLB and every affiliated level in a season."""
    out = {}
    for sport_id in (1, 11, 12, 13, 14, 15, 16, 17):
        try:
            teams = fetch_json(f"{STATSAPI}/teams", {"sportId": sport_id, "season": season}).get("teams", [])
        except FileNotFoundError:  # a level that did not exist that season (short-season A after 2020)
            continue
        for t in teams:
            out[t["id"]] = (t.get("name"), t.get("abbreviation"))
    return out


def grade_band(sample: pd.Series, floor: pd.Series) -> pd.Series:
    """Half-width of the grade band, a heuristic that widens with a short window: 10 * sqrt(floor / (n + floor))."""
    return 10.0 * np.sqrt(floor / (sample + floor))


def percentile_within(values: pd.Series) -> pd.Series:
    return (values.rank(pct=True) * 100).round(1)


def trait_summary(frame: pd.DataFrame, pool: str, contribs: pd.DataFrame) -> tuple[list[list], list[list]]:
    """Per player: trait grades and percentiles as integer lists in the pool's TRAITS order (null where untracked)."""
    grades = [[] for _ in frame.index]
    pcts = [[] for _ in frame.index]
    for _trait, inputs in TRAITS[pool]:
        cols = [c for c in inputs if c in contribs.columns]
        contribution = contribs[cols].sum(axis=1)
        sd = contribution.std()
        grade = ((50 + 10 * (contribution - contribution.mean()) / sd).clip(20, 80) if sd > 0 else pd.Series(50.0, index=frame.index)).round()
        pct = percentile_within(contribution).round()
        missing = frame[cols].isna().all(axis=1) if all(c in TRACKED_INPUTS for c in cols) else pd.Series(False, index=frame.index)
        for i, pid in enumerate(frame.index):
            grades[i].append(None if missing[pid] else int(grade[pid]))
            pcts[i].append(None if missing[pid] else int(pct[pid]))
    return grades, pcts


def tip(contribs: pd.DataFrame) -> list[list[list]]:
    """Per player, the four inputs that moved the peak grade most, as [input, contribution] pairs."""
    return [[[c, round(float(row[c]), 2)] for c in row.abs().sort_values(ascending=False).head(4).index] for _, row in contribs.iterrows()]


STUFF_COLUMNS = ["pitch_mix", "command_grade", "stuff_level", "stuff_pitches"]


STUFF_TRAIT = [t for t, _ in TRAITS["pit"]].index("stuff")


def pitcher_gate(rows: pd.DataFrame, tracked: pd.Series, adjustment: dict) -> pd.Series:
    """Grade reason per pitcher row: graded (None) only when the stuff rules passed and the pitcher is above the floor and tracked."""
    if not (adjustment.get("adoption_passed") and adjustment.get("elite_passed")):
        return pd.Series("model_not_validated", index=rows.index, dtype=object)
    reason = pd.Series([None] * len(rows), index=rows.index, dtype=object)
    reason[~tracked.reindex(rows.index).fillna(False).astype(bool)] = "not_tracked"
    reason[~rows["gradeable"].astype(bool)] = "insufficient_sample"
    return reason


def gate_ready(stuff_model: stuff.StuffModel | None) -> bool:
    """Whether the published adjustment reads injured-list history (models fitted before it carry no feature list)."""
    return stuff_model is not None and "il_days" in stuff_model.adjustment.get("features", [])


def stuff_columns(arsenal: pd.DataFrame, pitchers: pd.DataFrame) -> pd.DataFrame:
    """Per tracked pitcher, the published stuff columns: the packed pitch mix (seven smallints per type, by usage), command, level, sample."""
    if pitchers.empty:
        return pd.DataFrame(columns=["player_id", *STUFF_COLUMNS])

    def pack(g: pd.DataFrame) -> list:
        out = []
        for r in g.itertuples(index=False):
            out += [stuff.PITCH_CODES.index(r.pitch_type) + 1, int(round(100 * r.usage)), int(round(10 * r.velo)), int(round(10 * r.ivb)),
                    int(round(10 * r.hb)), int(round(r.spin)) if pd.notna(r.spin) else None, int(round(r.stuff_grade))]
        return out

    mix = arsenal.groupby("pitcher")[["pitch_type", "usage", "velo", "ivb", "hb", "spin", "stuff_grade"]].apply(pack).rename("pitch_mix")
    p = pitchers.set_index("pitcher").join(mix, how="left")
    level = p["stuff_level"].map(stuff.LEVEL_CODES) * np.where(p["stuff_adjusted"], 1, -1)
    out = pd.DataFrame({"pitch_mix": p["pitch_mix"], "command_grade": p["command_grade"].round().astype("Int64"),
                        "stuff_level": level.astype("Int64"), "stuff_pitches": p["pitches"].clip(upper=32767).round().astype("Int64")},
                       index=p.index)
    return out.reset_index().rename(columns={"pitcher": "player_id"})[["player_id", *STUFF_COLUMNS]]


def score(as_of: date, model: ProspectModel, stuff_model: stuff.StuffModel | None = None) -> pd.DataFrame:
    season = scoring_season(as_of)
    tables = dataset.load_tables(list(range(season - dataset.WINDOW + 1, season + 1)))
    active = set(tables[0].loc[tables[0]["season"] >= season - 1, "player_id"]) | set(tables[1].loc[tables[1]["season"] >= season - 1, "player_id"])
    ids = sorted(set(tables[0]["player_id"]) | set(tables[1]["player_id"]))
    mlb_ids = identity.mlb_team_ids()
    _, cached_roster = identity.people(ids)
    bio, roster = identity.people(ids, refresh_ids=identity.mlb_linked(cached_roster, mlb_ids))
    frame = dataset.snapshot(season, tables, bio, roster, dataset.baselines(tables, bio), as_of)
    frame = frame[frame["player_id"].isin(active) & frame["rookie_eligible"] & frame["in_age_range"]].reset_index(drop=True)
    log.info(f"score[{as_of}]: {len(frame):,} rookie-eligible players, {int(frame['gradeable'].sum()):,} above the floor")
    price = ValueModel(price=dict(model.price)).price_model()
    arsenal, tracked_pitchers = (stuff.window_arsenal(stuff_model, season, features.WINDOW_WEIGHTS) if stuff_model is not None
                                 else (pd.DataFrame(), pd.DataFrame()))
    feats = adjust.with_injuries(adjust.summarize(arsenal, tracked_pitchers), season, refresh_current=True) if gate_ready(stuff_model) \
        else adjust.summarize(arsenal, tracked_pitchers)
    gate = stuff_model is not None and bool(stuff_model.adjustment)
    pools = []
    for pool in POOLS:
        rows = frame[frame["role"] == pool].copy()
        if rows.empty:
            continue
        if pool == "pit" and gate:
            # Pitchers are graded one by one: tracked, above the floor, and only once the stuff rules passed.
            rows["grade_reason"] = pitcher_gate(rows, rows["player_id"].isin(feats.index), stuff_model.adjustment)
            graded = rows[rows["grade_reason"].isna()]
        else:
            validated = bool(model.validated.get(pool, False))
            rows["grade_reason"] = np.where(rows["gradeable"], None, "insufficient_sample") if validated else "model_not_validated"
            graded = rows[rows["gradeable"]] if validated else rows.iloc[0:0]
        if not graded.empty:
            pred, contribs = model.predict(graded, pool, contributions=True)
            if pool == "pit" and gate:
                pred = adjust.apply(pred, feats.reindex(graded["player_id"]).set_axis(graded.index), stuff_model.adjustment)
            band = grade_band(graded["sample"], graded["floor"])
            rows.loc[graded.index, "peak_war"] = pred["peak_war"].round(2)
            rows.loc[graded.index, "peak_grade"] = model.to_grade(pred["peak_war"], pool).round(1)
            rows.loc[graded.index, "peak_grade_low"] = (rows.loc[graded.index, "peak_grade"] - band).clip(lower=20).round(1)
            rows.loc[graded.index, "peak_grade_high"] = (rows.loc[graded.index, "peak_grade"] + band).clip(upper=80).round(1)
            rows.loc[graded.index, "peak_grade_percentile"] = percentile_within(rows.loc[graded.index, "peak_grade"])
            rows.loc[graded.index, "p_regular"] = pred["p_regular"].round(3)
            rows.loc[graded.index, "p_mlb"] = pred["p_mlb"].round(3)
            rows.loc[graded.index, "expected_control_war"] = pred["control_war"].round(2)
            roles = graded.apply(lambda r: value.price_role(pool, r.get("start_share", np.nan)), axis=1)
            tv = value.trade_values(pred, roles, price, model.arb_fractions, season)
            for c in ("trade_value_musd", "trade_value_low", "trade_value_high", "market_value_musd", "eta_season"):
                rows.loc[graded.index, c] = tv[c]
            rows.loc[graded.index, "risk"] = model.risk(pred, pool)
            rows.loc[graded.index, "price_role"] = roles
            g, pc = trait_summary(graded, pool, contribs["peak_war"])
            if pool == "pit" and gate:
                # The stuff trait of a graded pitcher is his usage-weighted stuff grade, not the base model's (untracked) inputs.
                arsenal_grade = feats["arsenal_stuff"].reindex(graded["player_id"]).to_numpy()
                pct = pd.Series(arsenal_grade).rank(pct=True).mul(100).round().to_numpy()
                for i in range(len(g)):
                    g[i][STUFF_TRAIT] = int(round(arsenal_grade[i]))
                    pc[i][STUFF_TRAIT] = int(pct[i])
            rows.loc[graded.index, "trait_grades"] = pd.Series(g, index=graded.index, dtype=object)
            rows.loc[graded.index, "trait_pcts"] = pd.Series(pc, index=graded.index, dtype=object)
            rows.loc[graded.index, "tip"] = pd.Series(tip(contribs["peak_war"]), index=graded.index, dtype=object)
        pools.append(rows)
    out = pd.concat(pools, ignore_index=True)
    if stuff_model is not None:
        cols = stuff_columns(arsenal, tracked_pitchers)
        pit = out["role"].eq("pit")
        out = out.merge(cols[cols["player_id"].isin(out.loc[pit, "player_id"])], on="player_id", how="left")
        log.info(f"score: stuff for {out['stuff_pitches'].notna().sum():,} of {int(pit.sum()):,} pitchers")
    for c in STUFF_COLUMNS:
        if c not in out:
            out[c] = None
    for c in ("trait_grades", "trait_pcts", "tip"):
        out[c] = out[c].where(out[c].notna(), None) if c in out else None
    b = bio.set_index("player_id")
    names = team_names(season)
    forty = identity.on_forty_man(roster, as_of, mlb_ids)
    out["player_name"] = out["player_id"].map(b["full_name"])
    out["org_id"] = out["org_id"].astype("Int64")
    out["org"] = out["org_id"].map(lambda i: names.get(int(i), (None, None))[1] if pd.notna(i) else None)
    out["team"] = out["team_id"].map(lambda i: names.get(int(i), (None, None))[0] if pd.notna(i) else None)
    out["level"] = out["top_level"]
    out["position_group"] = np.where(out["role"].eq("pit"),
                                     np.where(out["start_share"].fillna(0) >= STARTER_SHARE, "SP", "RP"), out["primary_pos"].fillna("DH"))
    out["bats"] = out["player_id"].map(b["bats"])
    out["throws"] = out["player_id"].map(b["throws"])
    out["headshot_url"] = out["player_id"].map(lambda i: HEADSHOT.format(player_id=int(i)))
    out["model_version"] = model.model_version
    out["season"] = season
    for c in ("draft_year", "draft_round", "draft_pick", "signing_bonus_musd"):
        out[c] = out["player_id"].map(b[c])
    school = out["player_id"].map(b["school_type"])
    out["acquired"] = np.where(out["draft_year"].notna(), "draft", np.where(school.eq("international"), "international", "undrafted"))
    out["on_40_man"] = out["player_id"].map(forty).fillna(False).astype(bool)
    out["mlb_ab_or_ip"] = np.where(out["role"].eq("pit"), out["mlb_ip"], out["mlb_ab"])
    out["service_days"] = out["mlb_days"].round().astype("Int64")
    out["as_of"] = as_of
    out["age"] = out["age"].round(1)
    return out


def fingerprint(values: pd.DataFrame, model: ProspectModel) -> str:
    """Hash of the published columns except the date, so an unchanged board is not rewritten."""
    h = hashlib.sha256(model.fitted_at.encode())
    columns = [c for c in VALUE_COLUMNS if c != "as_of"]
    h.update(values[columns].astype(str).to_csv(index=False).encode())
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument("--model", type=str, default=None, help="model JSON on disk instead of the newest database row")
    parser.add_argument("--stuff", type=str, default=None, help="stuff model JSON on disk instead of the newest database row")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--force", action="store_true", help="publish even when the fingerprint is unchanged")
    args = parser.parse_args()
    setup_logging()
    model = ProspectModel.load(Path(args.model)) if args.model else load_latest_from_db()
    stuff_model = stuff.StuffModel.load(Path(args.stuff)) if args.stuff else load_stuff_from_db()
    values = score(args.as_of, model, stuff_model)
    log.info(f"score: {len(values):,} rows, {int(values['peak_grade'].notna().sum()):,} graded")
    if args.no_write:
        out = CACHE_DIR / f"scored_{args.as_of}.parquet"
        values.drop(columns=["tip", "trait_grades", "trait_pcts"]).to_parquet(out, index=False)
        log.info(f"score: wrote {out}")
        return
    fp = fingerprint(values, model)
    if LAST_SCORED.exists() and not args.force and json.loads(LAST_SCORED.read_text()).get("fingerprint") == fp:
        log.info("score: the board is unchanged since the last publish; skipping")
        return
    publish(values)
    LAST_SCORED.write_text(json.dumps({"fingerprint": fp, "as_of": str(args.as_of)}))


if __name__ == "__main__":
    main()
