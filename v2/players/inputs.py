"""Per-player skill inputs by season, from the Statcast cache and Savant boards.

One long frame per season (player_id, season, role, input, value, sample),
cached on disk, so a nightly run recomputes only the current season. The same
``window`` and ``shrunk_rates`` functions serve fitting and scoring.

Usage:
    python -m v2.players.inputs --season 2025 [--player 592450]
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backend.data.game_types import regular_season_pitches
from backend.log import setup_logging
from v2.data.pa_dataset import EVENT_TO_OUTCOME, NON_PA_EVENTS, OUTCOMES
from v2.players.savant import load_board

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "players"
STATCAST_DIR = Path(__file__).resolve().parents[2] / "cache"

SWING = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play"}
WHIFF = {"swinging_strike", "swinging_strike_blocked"}
FASTBALLS = {"FF", "SI", "FA"}
IN_ZONE_MAX = 9  # Statcast zones 1-9 are the strike zone, 11-14 outside
SWEET_SPOT = (8.0, 32.0)
HARD_HIT = 95.0
# Statcast barrel: 98 mph at 26-30 degrees, the window widening one degree
# downward and two upward per mph, to 8-50 degrees at 116 mph.
BARREL_EV, BARREL_LOW, BARREL_HIGH = 98.0, 26.0, 30.0
PULL_ANGLE_DEGREES = 15.0
SPRAY_ORIGIN = (125.42, 198.27)
PITCHES_PER_INNING = 15.0
# Bat tracking (2024 onward): a hard swing is 75 mph or more; a squared-up ball
# carries at least 80 percent of the exit velocity the bat and pitch speeds allow,
# with the pitch about 8 percent slower at the plate than at release.
HARD_SWING = 75.0
SQUARED_UP_SHARE = 0.8
PLATE_SPEED_SHARE = 0.916
PA_PER_GAME = 4.5
STATCAST_COLUMNS = [
    "game_pk", "game_date", "game_type", "batter", "pitcher", "stand", "p_throws", "events", "description",
    "zone", "pitch_type", "pitch_number", "type", "release_speed", "pfx_x", "pfx_z",
    "launch_speed", "launch_angle", "bb_type", "hc_x", "hc_y", "estimated_woba_using_speedangle", "bat_speed",
    "woba_value", "woba_denom", "inning", "inning_topbot", "home_team", "away_team",
]


@dataclass(frozen=True)
class InputSpec:
    name: str
    role: str  # bat, fld, run, pit
    trait: str
    higher_is_better: bool
    source: str  # statcast or board:<name>


INPUTS: dict[str, InputSpec] = {s.name: s for s in [
    InputSpec("contact.whiff_rate", "bat", "contact", False, "statcast"),
    InputSpec("contact.zone_contact_rate", "bat", "contact", True, "statcast"),
    InputSpec("contact.squared_up_rate", "bat", "contact", True, "statcast"),
    InputSpec("contact.k_rate", "bat", "contact", False, "statcast"),
    InputSpec("power.barrel_rate", "bat", "power", True, "statcast"),
    InputSpec("power.ev90", "bat", "power", True, "statcast"),
    InputSpec("power.hard_hit_rate", "bat", "power", True, "statcast"),
    InputSpec("power.bat_speed", "bat", "power", True, "statcast"),
    InputSpec("power.hard_swing_rate", "bat", "power", True, "statcast"),
    InputSpec("power.hr_rate", "bat", "power", True, "statcast"),
    InputSpec("discipline.chase_rate", "bat", "discipline", False, "statcast"),
    InputSpec("discipline.zone_swing_rate", "bat", "discipline", True, "statcast"),
    InputSpec("discipline.bb_rate", "bat", "discipline", True, "statcast"),
    InputSpec("bbq.xwobacon", "bat", "bbq", True, "statcast"),
    InputSpec("bbq.sweet_spot_rate", "bat", "bbq", True, "statcast"),
    InputSpec("bbq.pulled_fb_rate", "bat", "bbq", True, "statcast"),
    InputSpec("bbq.gb_rate", "bat", "bbq", False, "statcast"),
    InputSpec("platoon.vs_lhp_delta", "bat", "platoon", True, "statcast"),
    InputSpec("range.oaa_per_600", "fld", "range", True, "board:oaa"),
    InputSpec("range.oaa_infront", "fld", "range", True, "board:oaa"),
    InputSpec("range.oaa_lateral", "fld", "range", True, "board:oaa"),
    InputSpec("range.oaa_behind", "fld", "range", True, "board:oaa"),
    InputSpec("arm.strength", "fld", "arm", True, "board:arm_strength"),
    InputSpec("arm.cs_above_average", "fld", "arm", True, "board:catcher_throwing"),
    InputSpec("arm.pop_time", "fld", "arm", False, "board:catcher_throwing"),
    InputSpec("reliability.blocking_runs", "fld", "reliability", True, "board:catcher_blocking"),
    InputSpec("reliability.framing_runs", "fld", "reliability", True, "board:catcher_framing"),
    InputSpec("baserunning.runs", "run", "baserunning", True, "board:baserunning"),
    InputSpec("baserunning.sb_runs", "run", "baserunning", True, "board:baserunning"),
    InputSpec("speed.sprint_speed", "run", "speed", True, "board:sprint_speed"),
    InputSpec("stuff.velo", "pit", "stuff", True, "statcast"),
    InputSpec("stuff.movement_vs_fb", "pit", "stuff", True, "statcast"),
    InputSpec("stuff.arsenal_rv100", "pit", "stuff", True, "board:pitch_arsenal_stats"),
    InputSpec("stuff.whiff_rate", "pit", "stuff", True, "statcast"),
    InputSpec("command.zone_rate", "pit", "command", True, "statcast"),
    InputSpec("command.first_pitch_strike", "pit", "command", True, "statcast"),
    InputSpec("command.bb_rate", "pit", "command", False, "statcast"),
    InputSpec("batmiss.k_rate", "pit", "batmiss", True, "statcast"),
    InputSpec("batmiss.chase_induced", "pit", "batmiss", True, "statcast"),
    InputSpec("contactmgmt.xwobacon_allowed", "pit", "contactmgmt", False, "statcast"),
    InputSpec("contactmgmt.barrel_rate_allowed", "pit", "contactmgmt", False, "statcast"),
    InputSpec("contactmgmt.gb_rate", "pit", "contactmgmt", True, "statcast"),
    InputSpec("contactmgmt.hr_rate", "pit", "contactmgmt", False, "statcast"),
    InputSpec("workload.ip_per_outing", "pit", "workload", True, "statcast"),
    InputSpec("workload.pitches_per_outing", "pit", "workload", True, "statcast"),
    InputSpec("workload.outings_per_month", "pit", "workload", True, "statcast"),
]}
LONG_COLUMNS = ["player_id", "season", "role", "input", "value", "sample"]


def _load(season: int) -> pd.DataFrame:
    path = STATCAST_DIR / f"statcast_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run python -m v2.data.build_cache --years {season}")
    frame = regular_season_pitches(pd.read_parquet(path, columns=STATCAST_COLUMNS), season)
    for col in ("batter", "pitcher", "zone", "pitch_number", "launch_angle"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    for col in ("release_speed", "pfx_x", "pfx_z", "launch_speed", "hc_x", "hc_y",
                "estimated_woba_using_speedangle", "woba_value", "woba_denom", "bat_speed"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    return frame


def _is_barrel(ls: pd.Series, la: pd.Series) -> pd.Series:
    widen = (ls - BARREL_EV).clip(lower=0)
    low = (BARREL_LOW - widen).clip(lower=8.0)
    high = (BARREL_HIGH + 2.0 * widen).clip(upper=50.0)
    return (ls >= BARREL_EV) & (la >= low) & (la <= high)


def _rate(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.where(den > 0)


def _rows(actor: str, values: pd.Series, sample: pd.Series, name: str, season: int) -> pd.DataFrame:
    spec = INPUTS[name]
    frame = pd.DataFrame({"player_id": values.index.astype("int64"), "value": values.values})
    frame["sample"] = sample.reindex(values.index).fillna(0).values
    frame = frame.dropna(subset=["value"])
    frame["season"], frame["role"], frame["input"] = season, spec.role, name
    return frame[LONG_COLUMNS]


def _batted_ball_inputs(bip: pd.DataFrame, key: str, prefix_map: dict[str, str], season: int) -> list[pd.DataFrame]:
    g = bip.groupby(key)
    n = g.size()
    out = []
    if "barrel" in prefix_map:
        out.append(_rows(key, _rate(g["barrel"].sum(), n), n, prefix_map["barrel"], season))
    if "xwobacon" in prefix_map:
        out.append(_rows(key, g["estimated_woba_using_speedangle"].mean(), n, prefix_map["xwobacon"], season))
    if "gb" in prefix_map:
        out.append(_rows(key, _rate(g["gb"].sum(), n), n, prefix_map["gb"], season))
    return out


def _statcast_inputs(pitches: pd.DataFrame, season: int) -> pd.DataFrame:
    p = pitches
    p = p.assign(
        swing=p["description"].isin(SWING),
        whiff=p["description"].isin(WHIFF),
        in_zone=p["zone"] <= IN_ZONE_MAX,
        pa_end=p["events"].notna() & ~p["events"].isin(NON_PA_EVENTS),
    )
    p["outcome"] = p["events"].map(EVENT_TO_OUTCOME)
    bip = p[p["description"].eq("hit_into_play") & p["launch_speed"].notna()].copy()
    bip["barrel"] = _is_barrel(bip["launch_speed"], bip["launch_angle"])
    bip["gb"] = bip["bb_type"].eq("ground_ball")
    bip["sweet"] = bip["launch_angle"].between(*SWEET_SPOT)
    spray = np.degrees(np.arctan2(bip["hc_x"] - SPRAY_ORIGIN[0], SPRAY_ORIGIN[1] - bip["hc_y"]))
    bip["pulled"] = np.where(bip["stand"].eq("R"), spray < -PULL_ANGLE_DEGREES, spray > PULL_ANGLE_DEGREES)
    bip["fb"] = bip["bb_type"].isin(["fly_ball", "line_drive"])
    out: list[pd.DataFrame] = []

    # Batters.
    gb = p.groupby("batter")
    swings, whiffs = gb["swing"].sum(), gb["whiff"].sum()
    zone_p = p[p["in_zone"]].groupby("batter")
    out_p = p[~p["in_zone"] & p["zone"].notna()].groupby("batter")
    pa = p[p["pa_end"]].groupby("batter")
    n_pa = pa.size()
    out.append(_rows("batter", _rate(whiffs, swings), swings, "contact.whiff_rate", season))
    out.append(_rows("batter", 1 - _rate(zone_p["whiff"].sum(), zone_p["swing"].sum()), zone_p["swing"].sum(), "contact.zone_contact_rate", season))
    out.append(_rows("batter", _rate(pa["outcome"].agg(lambda s: s.eq("K").sum()), n_pa), n_pa, "contact.k_rate", season))
    out.append(_rows("batter", _rate(pa["outcome"].agg(lambda s: s.eq("HR").sum()), n_pa), n_pa, "power.hr_rate", season))
    out.append(_rows("batter", _rate(pa["outcome"].agg(lambda s: s.eq("BB").sum()), n_pa), n_pa, "discipline.bb_rate", season))
    out.append(_rows("batter", _rate(out_p["swing"].sum(), out_p.size()), out_p.size(), "discipline.chase_rate", season))
    out.append(_rows("batter", _rate(zone_p["swing"].sum(), zone_p.size()), zone_p.size(), "discipline.zone_swing_rate", season))
    gbip = bip.groupby("batter")
    n_bip = gbip.size()
    out.append(_rows("batter", gbip["launch_speed"].quantile(0.9), n_bip, "power.ev90", season))
    tracked = p[p["swing"] & p["bat_speed"].notna()]
    if not tracked.empty:
        tg = tracked.groupby("batter")
        out.append(_rows("batter", tg["bat_speed"].mean(), tg.size(), "power.bat_speed", season))
        out.append(_rows("batter", _rate((tracked["bat_speed"] >= HARD_SWING).groupby(tracked["batter"]).sum(), tg.size()), tg.size(), "power.hard_swing_rate", season))
        tb = bip[bip["bat_speed"].notna() & bip["release_speed"].notna()]
        ceiling = 1.23 * tb["bat_speed"] + 0.23 * PLATE_SPEED_SHARE * tb["release_speed"]
        squared = (tb["launch_speed"] >= SQUARED_UP_SHARE * ceiling).groupby(tb["batter"])
        out.append(_rows("batter", squared.mean(), squared.size(), "contact.squared_up_rate", season))
    out.append(_rows("batter", _rate((bip["launch_speed"] >= HARD_HIT).groupby(bip["batter"]).sum(), n_bip), n_bip, "power.hard_hit_rate", season))
    out.append(_rows("batter", _rate(gbip["sweet"].sum(), n_bip), n_bip, "bbq.sweet_spot_rate", season))
    fb = bip[bip["fb"]].groupby("batter")
    out.append(_rows("batter", _rate(fb["pulled"].sum(), fb.size()), fb.size(), "bbq.pulled_fb_rate", season))
    out.extend(_batted_ball_inputs(bip, "batter", {"barrel": "power.barrel_rate", "xwobacon": "bbq.xwobacon", "gb": "bbq.gb_rate"}, season))
    paw = p[p["pa_end"] & p["woba_denom"].notna()]
    overall = paw.groupby("batter")["woba_value"].sum() / paw.groupby("batter")["woba_denom"].sum()
    lhp = paw[paw["p_throws"].eq("L")].groupby("batter")
    vs_lhp = lhp["woba_value"].sum() / lhp["woba_denom"].sum()
    out.append(_rows("batter", (vs_lhp - overall.reindex(vs_lhp.index)), lhp.size(), "platoon.vs_lhp_delta", season))

    # Pitchers.
    gp = p.groupby("pitcher")
    pswings, pwhiffs = gp["swing"].sum(), gp["whiff"].sum()
    n_pitches = gp.size()
    ppa = p[p["pa_end"]].groupby("pitcher")
    n_ppa = ppa.size()
    fastball = p[p["pitch_type"].isin(FASTBALLS)].groupby("pitcher")
    fb_mean = fastball[["pfx_x", "pfx_z"]].mean()
    out.append(_rows("pitcher", fastball["release_speed"].mean(), fastball.size(), "stuff.velo", season))
    secondary = p[~p["pitch_type"].isin(FASTBALLS) & p["pfx_x"].notna()].join(fb_mean, on="pitcher", rsuffix="_fb")
    sep = np.hypot(secondary["pfx_x"] - secondary["pfx_x_fb"], secondary["pfx_z"] - secondary["pfx_z_fb"])
    sep_g = sep.groupby(secondary["pitcher"])
    out.append(_rows("pitcher", sep_g.mean(), sep_g.size(), "stuff.movement_vs_fb", season))
    out.append(_rows("pitcher", _rate(pwhiffs, pswings), pswings, "stuff.whiff_rate", season))
    zone_share = p["in_zone"].groupby(p["pitcher"]).mean()
    out.append(_rows("pitcher", zone_share, n_pitches, "command.zone_rate", season))
    first = p[p["pitch_number"].eq(1)].groupby("pitcher")
    out.append(_rows("pitcher", first["type"].agg(lambda s: s.isin(["S", "X"]).mean()), first.size(), "command.first_pitch_strike", season))
    out.append(_rows("pitcher", _rate(ppa["outcome"].agg(lambda s: s.eq("BB").sum()), n_ppa), n_ppa, "command.bb_rate", season))
    out.append(_rows("pitcher", _rate(ppa["outcome"].agg(lambda s: s.eq("K").sum()), n_ppa), n_ppa, "batmiss.k_rate", season))
    out.append(_rows("pitcher", _rate(ppa["outcome"].agg(lambda s: s.eq("HR").sum()), n_ppa), n_ppa, "contactmgmt.hr_rate", season))
    pout = p[~p["in_zone"] & p["zone"].notna()].groupby("pitcher")
    out.append(_rows("pitcher", _rate(pout["swing"].sum(), pout.size()), pout.size(), "batmiss.chase_induced", season))
    out.extend(_batted_ball_inputs(bip, "pitcher", {"barrel": "contactmgmt.barrel_rate_allowed", "xwobacon": "contactmgmt.xwobacon_allowed", "gb": "contactmgmt.gb_rate"}, season))
    outings = p.groupby(["pitcher", "game_pk"]).agg(pitches=("pitcher", "size"), outs=("events", lambda s: s.map(EVENT_TO_OUTCOME).eq("OUT").sum() + s.eq("strikeout").sum()), month=("game_date", lambda s: s.iloc[0].month))
    per = outings.groupby("pitcher")
    n_out = per.size()
    out.append(_rows("pitcher", per["outs"].mean() / 3.0, n_out, "workload.ip_per_outing", season))
    out.append(_rows("pitcher", per["pitches"].mean(), n_out, "workload.pitches_per_outing", season))
    out.append(_rows("pitcher", n_out / per["month"].nunique(), n_out, "workload.outings_per_month", season))
    return pd.concat(out, ignore_index=True)


def _board_inputs(season: int) -> pd.DataFrame:
    out: list[pd.DataFrame] = []

    def add(frame: pd.DataFrame, column: str, sample: str | None, name: str, scale: float = 1.0) -> None:
        if column not in frame or frame.empty:
            return
        s = pd.to_numeric(frame[column], errors="coerce") * scale
        s.index = frame["player_id"].values
        n = pd.to_numeric(frame[sample], errors="coerce") if sample else pd.Series(1.0, index=frame.index)
        n.index = frame["player_id"].values
        out.append(_rows("board", s.groupby(level=0).mean(), n.groupby(level=0).sum(), name, season))

    oaa = load_board("oaa", season)
    if not oaa.empty:
        # The board has no chances column; innings are unknown here, so OAA is per season
        # and the window weighting handles exposure. Sample is the attempt proxy: |OAA| rows.
        oaa = oaa.assign(sample=1.0, lateral=pd.to_numeric(oaa["outs_above_average_lateral_toward3bline"], errors="coerce")
                         + pd.to_numeric(oaa["outs_above_average_lateral_toward1bline"], errors="coerce"))
        add(oaa, "outs_above_average", "sample", "range.oaa_per_600")
        add(oaa, "outs_above_average_infront", "sample", "range.oaa_infront")
        add(oaa, "lateral", "sample", "range.oaa_lateral")
        add(oaa, "outs_above_average_behind", "sample", "range.oaa_behind")
    arm = load_board("arm_strength", season)
    add(arm, "arm_overall", "total_throws", "arm.strength")
    throwing = load_board("catcher_throwing", season)
    add(throwing, "caught_stealing_above_average", "sb_attempts", "arm.cs_above_average")
    add(throwing, "pop_time", "sb_attempts", "arm.pop_time")
    blocking = load_board("catcher_blocking", season)
    add(blocking, "catcher_blocking_runs", "pitches", "reliability.blocking_runs")
    framing = load_board("catcher_framing", season)
    add(framing, "rv_tot", "pitches", "reliability.framing_runs")
    running = load_board("baserunning", season)
    if not running.empty:
        running = running.assign(sb=pd.to_numeric(running["runner_runs_SB2"], errors="coerce") + pd.to_numeric(running["runner_runs_SB3"], errors="coerce"))
    add(running, "runner_runs_tot", "N_runner_moved", "baserunning.runs")
    add(running, "sb", "N_runner_moved", "baserunning.sb_runs")
    sprint = load_board("sprint_speed", season)
    add(sprint, "sprint_speed", "competitive_runs", "speed.sprint_speed")
    arsenal = load_board("pitch_arsenal_stats", season)
    if not arsenal.empty:
        arsenal = arsenal.assign(rv=pd.to_numeric(arsenal["run_value"], errors="coerce"), n=pd.to_numeric(arsenal["pitches"], errors="coerce"))
        per = arsenal.groupby("player_id").agg(rv=("rv", "sum"), n=("n", "sum"))
        out.append(_rows("board", 100.0 * per["rv"] / per["n"], per["n"], "stuff.arsenal_rv100", season))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=LONG_COLUMNS)


def season_inputs(season: int, *, refresh: bool = False) -> pd.DataFrame:
    """Long input frame for one season, cached at cache/players/inputs_{season}.parquet."""
    path = CACHE_DIR / f"inputs_{season}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    pitches = _load(season)
    frame = pd.concat([_statcast_inputs(pitches, season), _board_inputs(season)], ignore_index=True)
    frame = frame.astype({"player_id": "int64", "season": "int64", "value": "float64", "sample": "float64"})
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    log.info(f"inputs[{season}]: {len(frame):,} rows for {frame.player_id.nunique():,} players")
    return frame


def outcome_counts(season: int) -> pd.DataFrame:
    """Plate-appearance outcome counts per player-season for batters (with vs-LHP split) and pitchers."""
    path = CACHE_DIR / f"outcomes_{season}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    p = _load(season)
    pa = p[p["events"].notna() & ~p["events"].isin(NON_PA_EVENTS)].copy()
    pa["outcome"] = pa["events"].map(EVENT_TO_OUTCOME)
    pa = pa.dropna(subset=["outcome"])
    frames = []
    for key, role in (("batter", "bat"), ("pitcher", "pit")):
        counts = pd.crosstab(pa[key], pa["outcome"]).reindex(columns=list(OUTCOMES), fill_value=0)
        counts.index.name = "player_id"
        counts = counts.reset_index()
        counts["role"] = role
        if role == "bat":
            lhp = pd.crosstab(pa.loc[pa["p_throws"].eq("L"), key], pa.loc[pa["p_throws"].eq("L"), "outcome"])
            lhp = lhp.reindex(columns=list(OUTCOMES), fill_value=0).add_prefix("vs_lhp_")
            counts = counts.join(lhp, on="player_id").fillna(0)
        frames.append(counts)
    out = pd.concat(frames, ignore_index=True).fillna(0)
    out["season"] = season
    out["player_id"] = out["player_id"].astype("int64")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def window(frames_by_season: dict[int, pd.DataFrame], weights: dict[int, float]) -> pd.DataFrame:
    """Sample-and-season-weighted mean of each input over the window.

    ``weights`` maps season to weight. Each (player, role, input) value is the mean
    of season values weighted by ``weight × sample``; ``sample`` is the weighted sum.
    """
    parts = []
    for season, frame in frames_by_season.items():
        w = weights.get(season, 0.0)
        if w <= 0 or frame.empty:
            continue
        f = frame.copy()
        f["w"] = w * f["sample"].clip(lower=1.0)
        f["wv"] = f["w"] * f["value"]
        f["ws"] = w * f["sample"]
        parts.append(f)
    if not parts:
        return pd.DataFrame(columns=LONG_COLUMNS)
    stacked = pd.concat(parts, ignore_index=True)
    agg = stacked.groupby(["player_id", "role", "input"]).agg(wv=("wv", "sum"), w=("w", "sum"), sample=("ws", "sum")).reset_index()
    agg["value"] = agg["wv"] / agg["w"]
    agg["season"] = max(s for s, w in weights.items() if w > 0)
    return agg[LONG_COLUMNS]


def window_counts(counts_by_season: dict[int, pd.DataFrame], weights: dict[int, float]) -> pd.DataFrame:
    """Season-weighted outcome counts per (player, role)."""
    parts = []
    for season, frame in counts_by_season.items():
        w = weights.get(season, 0.0)
        if w <= 0 or frame.empty:
            continue
        f = frame.drop(columns=["season"]).copy()
        numeric = [c for c in f.columns if c not in ("player_id", "role")]
        f[numeric] = f[numeric] * w
        parts.append(f)
    stacked = pd.concat(parts, ignore_index=True)
    return stacked.groupby(["player_id", "role"]).sum().reset_index()


def shrunk_rates(counts: pd.DataFrame, group_of: pd.Series, strength: float) -> pd.DataFrame:
    """Dirichlet-multinomial shrink of each player's outcome counts toward his group's rates.

    ``group_of`` maps player_id to position group. Returns one row per player with the
    eight shrunk rates, ``n`` (weighted PA), and ``group``.
    """
    cols = list(OUTCOMES)
    frame = counts[["player_id", "role", *cols]].copy()
    frame["group"] = frame["player_id"].map(group_of)
    frame["group"] = frame["group"].fillna(pd.Series(np.where(frame["role"].eq("pit"), "RP", "DH"), index=frame.index))
    totals = frame.groupby("group")[cols].sum()
    prior = totals.div(totals.sum(axis=1), axis=0)
    n = frame[cols].sum(axis=1)
    prior_rows = prior.reindex(frame["group"]).values
    shrunk = (frame[cols].values + strength * prior_rows) / (n.values[:, None] + strength)
    out = pd.DataFrame(shrunk, columns=cols)
    out.insert(0, "player_id", frame["player_id"].values)
    out["role"], out["group"], out["n"] = frame["role"].values, frame["group"].values, n.values
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute per-player valuation inputs for a season.")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--player", type=int)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    frame = season_inputs(args.season, refresh=args.refresh)
    counts = outcome_counts(args.season)
    if args.player:
        print(frame[frame.player_id == args.player].sort_values(["role", "input"]).to_string(index=False))
        print(counts[counts.player_id == args.player].to_string(index=False))
    else:
        print(frame.groupby(["role", "input"]).agg(players=("player_id", "nunique"), mean=("value", "mean")).round(4).to_string())


if __name__ == "__main__":
    setup_logging()
    main()
