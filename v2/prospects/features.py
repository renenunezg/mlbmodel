"""Snapshot features: one row per player at the end of a season, pooled over a recent-season window.

Counts from each (season, level) row are pooled with the player model's window weights (this
season 5, last season 4, two back 3) and rates are formed from the pooled counts. Each rate also
has a level-relative version: the plate-appearance-weighted mean of (rate minus the level-season
mean), so a Double-A strikeout rate and a complex-league one are compared to their own leagues.
Age is relative to the fitted median age of regulars at the highest level played this season.

Tracked features come only from tracked opportunities and are missing, never zero, for players
without tracking. ``tracked_bip`` and ``tracked_pitches`` are their denominators.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from v2.players.model import DEFAULT_WINDOW
from v2.prospects import events

WINDOW_WEIGHTS = dict(DEFAULT_WINDOW)  # the player model's recent-season weights: this season 5, last 4, two back 3
REGULAR_PA = 200  # a level regular, for the age baseline
REGULAR_BF = 150
AGE_DATE = (7, 1)  # season age is age on July 1
FALLBACK_LEVEL_AGE = {"dsl": 18.5, "rk": 19.5, "rk+": 20.5, "a-": 21.0, "a": 21.5, "a+": 22.5, "aa": 24.0, "aaa": 26.0, "afl": 23.5}
HITTER_RATES = {
    "k_rate": ("K", "pa"), "bb_rate": ("BB", "pa"), "hr_rate": ("HR", "pa"), "xbh_rate": ("XBH", "pa"),
    "swing_rate": ("swings", "n_pitches"), "contact_rate": ("contacts", "swings"), "swstr_rate": ("swinging_strikes", "n_pitches"),
    "csw_rate": ("CSW", "n_pitches"), "gb_share": ("bb_gb", "bip"), "ld_share": ("bb_ld", "bip"), "fb_share": ("bb_fb", "bip"),
    "pu_share": ("bb_pu", "bip"), "hard_share": ("bb_hard", "bip"), "soft_share": ("bb_soft", "bip"),
    "sb_rate": ("sb", "pa"), "cs_rate": ("cs", "SBA"), "babip": ("BIPH", "BIPAB"), "iso": ("TBX", "AB"),
    "hbp_rate": ("HBP", "pa"), "foul_share": ("fouls", "swings"), "called_rate": ("called_strikes", "n_pitches"),
    "strike_rate": ("STRIKES", "n_pitches"), "pitches_per_pa": ("n_pitches", "seq_pa"),
}
PITCHER_RATES = {
    "k_rate": ("K", "bf"), "bb_rate": ("BB", "bf"), "hr_rate": ("HR", "bf"), "swstr_rate": ("swinging_strikes", "n_pitches"),
    "csw_rate": ("CSW", "n_pitches"), "contact_rate": ("contacts", "swings"), "gb_share": ("bb_gb", "bip"), "fb_share": ("bb_fb", "bip"),
    "hard_share": ("bb_hard", "bip"), "soft_share": ("bb_soft", "bip"), "babip": ("BIPH", "BIPAB"), "start_share": ("starts", "games"),
    "outs_per_game": ("outs", "games"),
    "pu_share": ("bb_pu", "bip"), "ld_share": ("bb_ld", "bip"), "xbh_rate": ("XBH", "bf"), "hbp_rate": ("HBP", "bf"),
    "foul_share": ("fouls", "swings"), "called_rate": ("called_strikes", "n_pitches"), "strike_rate": ("STRIKES", "n_pitches"),
    "pitches_per_pa": ("n_pitches", "seq_pa"),
}
TREND_RATES = {"bat": ["k_rate", "bb_rate", "iso", "swstr_rate"], "pit": ["k_rate", "bb_rate", "swstr_rate", "gb_share"]}
# Tracked overlay features and the tracked opportunity each is pooled by; a row without the measurement is left out of both sides.
HITTER_TRACKED = {"ev90": "tracked_bip", "ev_mean": "tracked_bip", "hard_hit": "tracked_bip", "la_mean": "tracked_bip"}
PITCHER_TRACKED = {"fb_velo": "tracked_pitches", "fb_velo_max": "tracked_pitches", "fb_ivb": "tracked_pitches", "fb_hb": "tracked_pitches",
                   "fb_spin": "tracked_pitches", "extension": "tracked_pitches", "whiff_tracked": "tracked_pitches",
                   "ev_mean_allowed": "tracked_bip", "hard_hit_allowed": "tracked_bip"}
SCHOOL_TYPES = ("high_school", "college", "international", "undrafted")
HITTER_SAMPLE = "pa"
PITCHER_SAMPLE = "bf"


def _derived_counts(t: pd.DataFrame, sample: str) -> pd.DataFrame:
    t = t.copy()
    t["XBH"] = t["2B"] + t["3B"] + t["HR"]
    t["CSW"] = t["called_strikes"] + t["swinging_strikes"]
    t["STRIKES"] = (t["n_pitches"] - t["balls"]).clip(lower=0)
    t["AB"] = (t[sample] - t["BB"] - t["HBP"]).clip(lower=0)
    t["TBX"] = t["2B"] + 2 * t["3B"] + 3 * t["HR"]
    t["BIPH"] = t["1B"] + t["2B"] + t["3B"]
    t["BIPAB"] = (t["AB"] - t["K"] - t["HR"]).clip(lower=0)
    if "sb" in t:
        t["sb"] = t["sb"].fillna(0)
        t["cs"] = t["cs"].fillna(0)
        t["SBA"] = t["sb"] + t["cs"]
    for c in ("starts", "games", "outs"):
        if c in t:
            t[c] = t[c].fillna(0)
    return t


def level_age_baseline(table: pd.DataFrame, bio: pd.DataFrame, sample: str, min_sample: int) -> pd.DataFrame:
    """Median season age of level regulars per level and season; falls back to a fixed ladder where thin."""
    t = table[table[sample] >= min_sample].merge(bio[["player_id", "birth_date"]], on="player_id", how="left")
    t["age"] = season_age(t["birth_date"], t["season"]).to_numpy()
    med = t.groupby(["level", "season"])["age"].agg(["median", "size"]).reset_index()
    med["age_baseline"] = np.where(med["size"] >= 30, med["median"], med["level"].map(FALLBACK_LEVEL_AGE))
    return med[["level", "season", "age_baseline"]]


def season_age(birth, season) -> pd.Series:
    """Age on July 1 of the season; ``birth`` and ``season`` are aligned arrays or series."""
    b = pd.Series(pd.to_datetime(np.asarray(birth), errors="coerce"))
    s = pd.Series(np.asarray(season)).astype(int).astype(str)
    when = pd.to_datetime(s + f"-{AGE_DATE[0]:02d}-{AGE_DATE[1]:02d}")
    return ((when - b).dt.days / 365.25).round(2)


def _level_means(t: pd.DataFrame, rates: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Sample-weighted rate per level and season (the league context each rate is measured against)."""
    sums = t.groupby(["level", "season"])[sorted({c for pair in rates.values() for c in pair})].sum()
    out = pd.DataFrame(index=sums.index)
    for name, (num, den) in rates.items():
        out[f"lvl_{name}"] = sums[num] / sums[den].where(sums[den] > 0)
    return out.reset_index()


def snapshot_rows(table: pd.DataFrame, snapshot_season: int, rates: dict[str, tuple[str, str]], sample: str,
                  tracked: dict[str, str]) -> pd.DataFrame:
    """Pooled features for every player with a (season, level) row inside the window ending at ``snapshot_season``."""
    t = _derived_counts(table, sample)
    t = t[(t["season"] <= snapshot_season) & (t["season"] > snapshot_season - len(WINDOW_WEIGHTS))].copy()
    if t.empty:
        return pd.DataFrame()
    t["w"] = (snapshot_season - t["season"]).map(WINDOW_WEIGHTS)
    means = _level_means(_derived_counts(table, sample), rates)
    t = t.merge(means, on=["level", "season"], how="left")
    t["level_rank"] = t["level"].map(events.LEVEL_RANK)
    count_cols = sorted({c for pair in rates.values() for c in pair})
    weighted = t[count_cols].mul(t["w"], axis=0)
    weighted["player_id"] = t["player_id"]
    pooled = weighted.groupby("player_id").sum()
    out = pd.DataFrame(index=pooled.index)
    out["sample_w"] = pooled[sample]
    out["sample"] = t.groupby("player_id")[sample].sum()
    for name, (num, den) in rates.items():
        out[name] = pooled[num] / pooled[den].where(pooled[den] > 0)
        rel = (t[num] / t[den].where(t[den] > 0) - t[f"lvl_{name}"]) * t["w"] * t[den]
        out[f"{name}_rel"] = rel.groupby(t["player_id"]).sum() / (t["w"] * t[den]).groupby(t["player_id"]).sum().where(lambda s: s > 0)
    out["k_minus_bb_rel"] = out["k_rate_rel"] - out["bb_rate_rel"]
    # Trend: this season's level-relative rate minus last season's, each pooled by sample within the season.
    for name in TREND_RATES["bat" if sample == "pa" else "pit"]:
        num, den = rates[name]
        rel = (t[num] / t[den].where(t[den] > 0) - t[f"lvl_{name}"]) * t[den]
        by_season = rel.groupby([t["player_id"], t["season"]]).sum() / t[den].groupby([t["player_id"], t["season"]]).sum().where(lambda s: s > 0)
        this_s = by_season.xs(snapshot_season, level="season") if snapshot_season in by_season.index.get_level_values("season") else pd.Series(dtype=float)
        last_s = by_season.xs(snapshot_season - 1, level="season") if snapshot_season - 1 in by_season.index.get_level_values("season") else pd.Series(dtype=float)
        out[f"{name}_trend"] = (this_s.reindex(out.index) - last_s.reindex(out.index))
    # This season's shape: levels played, the highest with real sample, promotion. The Arizona Fall
    # League is a showcase, not a rung: its rows pool into the rates but never set the level or club.
    affiliated = t[t["level"] != "afl"]
    this = affiliated[affiliated["season"] == snapshot_season]
    g = this.groupby("player_id")
    out["season_sample"] = g[sample].sum().reindex(out.index).fillna(0.0)
    out["n_levels"] = g["level"].nunique().reindex(out.index).fillna(0)
    top = this.sort_values(["level_rank", sample], ascending=[False, False]).groupby("player_id").head(1).set_index("player_id")
    out["top_level"] = top["level"].reindex(out.index)
    out["top_level_rank"] = top["level_rank"].reindex(out.index)
    out["top_level_sample"] = top[sample].reindex(out.index).fillna(0.0)
    out["top_level_share"] = (out["top_level_sample"] / out["season_sample"].where(out["season_sample"] > 0)).fillna(0.0)
    latest = affiliated.sort_values(["season", "level_rank", sample]).groupby("player_id").tail(1).set_index("player_id")
    out["org_id"] = latest["org_id"].reindex(out.index) if "org_id" in latest else np.nan
    out["team_id"] = latest["team_id"].reindex(out.index) if "team_id" in latest else np.nan
    out["last_season"] = latest["season"].reindex(out.index)
    prev = affiliated[affiliated["season"] < snapshot_season].groupby("player_id")["level_rank"].max()
    out["prev_top_level_rank"] = prev.reindex(out.index)
    out["promoted"] = (out["top_level_rank"] > out["prev_top_level_rank"]).astype(float).where(out["prev_top_level_rank"].notna())
    out["career_max_level_rank"] = affiliated.groupby("player_id")["level_rank"].max().reindex(out.index)
    # Tracked overlay: each measurement pooled over the rows that carry it, weighted by its own tracked opportunities.
    for name, den_name in tracked.items():
        have = t[name].notna()
        den = (t[den_name].fillna(0.0) * t["w"]).where(have, 0.0)
        num = (t[name] * den).where(have, 0.0)
        out[name] = num.groupby(t["player_id"]).sum() / den.groupby(t["player_id"]).sum().where(lambda s: s > 0)
    for den_name in sorted(set(tracked.values())):
        out[den_name] = t.groupby("player_id")[den_name].sum().reindex(out.index).fillna(0.0)
    out["tracked_share"] = (t.groupby("player_id")["tracked_pa"].sum() / t.groupby("player_id")[sample].sum()).reindex(out.index)
    out["season"] = snapshot_season
    return out.reset_index()


def with_bio(rows: pd.DataFrame, bio: pd.DataFrame, baseline: pd.DataFrame, pos: pd.DataFrame | None = None) -> pd.DataFrame:
    """Age, age relative to level, pedigree and position columns."""
    b = bio.set_index("player_id")
    out = rows.copy()
    out["age"] = season_age(b["birth_date"].reindex(out["player_id"]).to_numpy(), out["season"]).to_numpy()
    base = baseline.set_index(["level", "season"])["age_baseline"]
    keys = pd.MultiIndex.from_frame(out[["top_level", "season"]])
    out["level_age"] = base.reindex(keys).to_numpy()
    missing = out["level_age"].isna() & out["top_level"].notna()
    if missing.any():
        raise ValueError(f"No level age baseline for {sorted(set(map(tuple, out.loc[missing, ['top_level', 'season']].to_numpy())))[:5]}")
    out["age_rel"] = out["age"] - out["level_age"]
    out["draft_round"] = pd.to_numeric(b["draft_round"].reindex(out["player_id"]).to_numpy(), errors="coerce")
    out["draft_pick"] = pd.to_numeric(b["draft_pick"].reindex(out["player_id"]).to_numpy(), errors="coerce")
    out["log_draft_pick"] = np.log(out["draft_pick"].clip(lower=1))
    out["drafted"] = out["draft_pick"].notna().astype(float)
    school = b["school_type"].reindex(out["player_id"]).to_numpy()
    for s in SCHOOL_TYPES:
        out[f"school_{s}"] = (school == s).astype(float)
    out["years_pro"] = out["season"] - pd.to_numeric(b["draft_year"].reindex(out["player_id"]).to_numpy(), errors="coerce")
    bats = b["bats"].reindex(out["player_id"]).to_numpy()
    out["bats_left"] = (bats == "L").astype(float)
    out["bats_switch"] = (bats == "S").astype(float)
    out["throws_left"] = (b["throws"].reindex(out["player_id"]).to_numpy() == "L").astype(float)
    if pos is not None:
        out = out.merge(pos, on="player_id", how="left")
    return out


def position_shares(table: pd.DataFrame, snapshot_season: int) -> pd.DataFrame:
    """Window-weighted share of games at each position per hitter."""
    cols = [f"pos_{c}" for c in events.POSITION_CODES]
    t = table[(table["season"] <= snapshot_season) & (table["season"] > snapshot_season - len(WINDOW_WEIGHTS))].copy()
    t = t[t["games"].notna()]
    t["w"] = (snapshot_season - t["season"]).map(WINDOW_WEIGHTS) * t["games"]
    num = t[cols].fillna(0.0).mul(t["w"], axis=0).groupby(t["player_id"]).sum()
    den = t.groupby("player_id")["w"].sum()
    out = num.div(den.where(den > 0), axis=0)
    out["primary_pos"] = out[cols].idxmax(axis=1).str.replace("pos_", "").where(out[cols].max(axis=1) > 0)
    return out.reset_index()


def hitter_features(table: pd.DataFrame, bio: pd.DataFrame, snapshot_season: int, baseline: pd.DataFrame) -> pd.DataFrame:
    rows = snapshot_rows(table, snapshot_season, HITTER_RATES, HITTER_SAMPLE, HITTER_TRACKED)
    if rows.empty:
        return rows
    out = with_bio(rows, bio, baseline, position_shares(table, snapshot_season))
    out["role"] = "bat"
    return out


def pitcher_features(table: pd.DataFrame, bio: pd.DataFrame, snapshot_season: int, baseline: pd.DataFrame) -> pd.DataFrame:
    rows = snapshot_rows(table, snapshot_season, PITCHER_RATES, PITCHER_SAMPLE, PITCHER_TRACKED)
    if rows.empty:
        return rows
    out = with_bio(rows, bio, baseline)
    out["role"] = "pit"
    return out


def hitter_feature_names() -> list[str]:
    base = list(HITTER_RATES) + [f"{r}_rel" for r in HITTER_RATES] + ["k_minus_bb_rel"] + [f"{r}_trend" for r in TREND_RATES["bat"]]
    shape = ["sample_w", "season_sample", "n_levels", "top_level_rank", "top_level_share", "promoted", "career_max_level_rank",
             "age", "age_rel", "draft_round", "log_draft_pick", "drafted", "years_pro", "bats_left", "bats_switch"]
    school = [f"school_{s}" for s in SCHOOL_TYPES]
    pos = [f"pos_{c}" for c in events.POSITION_CODES]
    return base + shape + school + pos + list(HITTER_TRACKED) + ["tracked_bip"]


def pitcher_feature_names() -> list[str]:
    base = list(PITCHER_RATES) + [f"{r}_rel" for r in PITCHER_RATES] + ["k_minus_bb_rel"] + [f"{r}_trend" for r in TREND_RATES["pit"]]
    shape = ["sample_w", "season_sample", "n_levels", "top_level_rank", "top_level_share", "promoted", "career_max_level_rank",
             "age", "age_rel", "draft_round", "log_draft_pick", "drafted", "years_pro", "throws_left"]
    school = [f"school_{s}" for s in SCHOOL_TYPES]
    return base + shape + school + list(PITCHER_TRACKED) + ["tracked_pitches", "tracked_bip"]


def snapshot_date(season: int) -> date:
    return date(season, 11, 1)
