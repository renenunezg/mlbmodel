"""Month-start forward evaluation: the Grade against rest-of-season production.

For each season and month start, the Grade is rebuilt from what was knowable that
morning: a grade model whose development pairs end two seasons earlier, prior
seasons' cached inputs, and the current season's Statcast pitches before the
date. Savant season boards cannot be cut by date, so the current season
contributes no board inputs (a conservative omission the live scorer does not
make). The target is each player's rest-of-season batting or pitching run value
per full season from the pitches on or after the date; fielding and baserunning
boards are season-level and stay out of the target. Because that hitter target
is batting-only while the Grade is fitted to batting plus fielding and running,
hitters are also scored with the same grade weights refitted to the batting-only
target on the same development pairs (``grade_batting``): that is the
like-for-like comparison with the batting baselines.

Baselines: season-to-date run value, in-season Marcel, prior-season fWAR,
season-to-date fWAR pace where the FanGraphs date-range leaderboard has been
cached (``--war-urls`` prints what to cache), and the preseason ZiPS, THE BAT
(X for hitters) and Steamer projections where their FanGraphs pages have been
cached from Wayback Machine snapshots (``fangraphs_projection_{type}_{stats}_{season}.json``).
The first row of each season is the preseason grade, built on the regular
season's opening day with no current-season data, which is the projections'
own footing; later month starts pit an updating grade against static
preseason projections.

Usage:
    python -m v2.players.forward --seasons 2024 2025 2026 [--months 4 5 6 7 8 9]
    python -m v2.players.forward --war-urls 2024
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from backend.log import setup_logging
from v2.data.pa_dataset import OUTCOMES
from v2.players import bio, expected, features, inputs, minors
from v2.players.fit import (
    MIN_FIT_SAMPLE,
    MIN_TARGET_SAMPLE,
    SeasonData,
    build_pair,
    choose_window_and_shrink,
    configure_hitter_shrink,
    fit_grade_weights,
    fit_trait_weights,
    marcel,
    season_rates,
    spearman,
)
from v2.players.inputs import CACHE_DIR, STATCAST_DIR
from v2.players.model import ValueModel
from v2.players.production import load_war
from v2.players.runs import FULL_SEASON_PA, RELIEVER_INNINGS, STARTER_INNINGS

log = logging.getLogger(__name__)

FIRST_PAIR_SEASON = 2021
MONTHS = (4, 5, 6, 7, 8, 9)
POOLS = (features.HITTER_POOL, "SP", "RP")
PROJECTION_SYSTEMS = ("zips", "the_bat", "steamer")
# Preseason page types per system: hitters, pitchers (THE BAT X projects hitters, THE BAT pitchers).
PRESEASON_PAGES = {"zips": ("zips", "zips"), "the_bat": ("thebatx", "thebat"), "steamer": ("steamer", "steamer")}
# Rest-of-season sets exist only as dated page snapshots; a checkpoint on a snapshot's date gets them as extra baselines.
ROS_SYSTEMS = {"zips_ros": "rzips", "the_bat_ros": "rthebatx"}
PROJECTED_STATS = {"bat": ("wOBA", "WAR", "PA"), "pit": ("ERA", "FIP", "WAR", "IP")}
RESULT_PATH = CACHE_DIR / "forward_evaluation.json"
FANGRAPHS = ("https://www.fangraphs.com/api/leaders/major-league/data?age=&pos=all&stats={stats}&lg=all&qual=0"
             "&season={season}&season1={season}&startdate={season}-03-01&enddate={end}&month=1000&hand=&team=0"
             "&pageitems=5000&pagenum=1&ind=0&rost=0&players=&position=all&type=8&postseason=&sortdir=default&sortstat=WAR")


def _has_statcast(season: int) -> bool:
    return (STATCAST_DIR / f"statcast_{season}.parquet").exists()


# The grade model knowable before the evaluated season.


def batting_only(pairs: list[dict]) -> list[dict]:
    """The same development pairs with each hitter's target reduced to batting runs."""
    out = []
    for p in pairs:
        tn = p["target_next"].copy()
        tn["target"] = tn["bat_runs"].where(tn["role"].eq("bat"), tn["target"])
        out.append({**p, "target_next": tn})
    return out


def variant_suffix(*, tree: bool = False, expected_outcomes: bool = False, expected_blend: bool = False, components: bool = False,
                   direct_batting: bool = False) -> str:
    return "".join(f"_{name}" for name, on in (("tree", tree), ("expected", expected_outcomes), ("blend", expected_blend),
                                               ("components", components), ("direct", direct_batting)) if on)


def grade_models(last_dev_season: int, *, tree: bool = False, expected_outcomes: bool = False, expected_blend: bool = False,
                 components: bool = False, direct_batting: bool = False, cache_dir=CACHE_DIR) -> tuple[ValueModel, ValueModel]:
    """(full, batting-only) grade models from pairs ending in ``last_dev_season + 1``, cached on disk.

    Both share window, shrink and trait weights; only the hitter grade weights differ.
    The switches select the candidate hitter paths (per-node shrink tree, Statcast
    expected outcomes, component grades) and name the cache file; a component
    model's batting component is its own batting-only grade, so both paths share one file.
    """
    suffix = variant_suffix(tree=tree, expected_outcomes=expected_outcomes, expected_blend=expected_blend, components=components, direct_batting=direct_batting)
    path = cache_dir / f"forward_grade_model_{last_dev_season}{suffix}.json"
    batting_path = cache_dir / f"forward_grade_model_{last_dev_season}{suffix}_batting.json"
    if path.exists() and batting_path.exists():
        return ValueModel.load(path), ValueModel.load(batting_path)
    dev_first = list(range(FIRST_PAIR_SEASON, last_dev_season + 1))
    dev_seasons = list(range(FIRST_PAIR_SEASON, last_dev_season + 2))
    seasons = [s for s in dev_seasons if _has_statcast(s)]
    if last_dev_season + 1 not in seasons:
        raise FileNotFoundError(f"Grade model through {last_dev_season} needs Statcast for {last_dev_season + 1}")
    data = SeasonData(seasons)
    model = ValueModel.empty(dev_seasons, (last_dev_season + 1, last_dev_season + 2))
    model.window_weights, model.shrink_strength, model.aaa_scale = choose_window_and_shrink(data, dev_first)
    model.aaa_input_offsets = minors.input_offsets(data.inputs, data.aaa_inputs)
    model.linear_weights = data.linear_weights[last_dev_season]
    model.pa_per_ip = data.pa_per_ip[last_dev_season]
    model.evaluation = {"hitter_shrink": configure_hitter_shrink(data, dev_first, dev_seasons, model, tree=tree, expected_outcomes=expected_outcomes, expected_blend=expected_blend)}
    model.component_grades, model.direct_batting = components, direct_batting
    pairs = [build_pair(data, t, model) for t in dev_first]
    model.trait_weights = fit_trait_weights(pairs)
    model.grade_weights = fit_grade_weights(pairs, model)
    model.model_version = f"forward-grade-{last_dev_season}{suffix}"
    model.save(path)
    batting = ValueModel.from_payload(model.to_payload())
    if not components:
        batting.grade_weights = fit_grade_weights(batting_only(pairs), model)
    batting.model_version = f"forward-grade-{last_dev_season}{suffix}-batting"
    batting.save(batting_path)
    log.info(f"grade model through {last_dev_season}{suffix}: window {model.window_weights}, shrink {model.shrink_strength}, "
             f"tree {model.shrink_tree or 'off'}, expected scale {model.expected_scale}, expected blend {model.expected_blend}")
    return model, batting


# One season's fixed material.


class SeasonMaterial:
    """The evaluated season's pitches and the prior seasons' cached frames."""

    def __init__(self, season: int, model: ValueModel, batting_model: ValueModel):
        self.season, self.model, self.batting_model = season, model, batting_model
        self._pitches: pd.DataFrame | None = None
        self.prior = [s for s in (season - 1, season - 2) if _has_statcast(s)]
        self.prior_inputs = {s: inputs.season_inputs(s) for s in self.prior}
        self.prior_aaa_inputs = {s: minors.season_inputs(s) for s in self.prior}
        self.prior_counts = {s: inputs.outcome_counts(s) for s in self.prior}
        self.prior_positions = bio.positions(self.prior)
        self.prior_pseudo = {s: minors.pseudo_counts(s) for s in self.prior}
        self.ages = bio.ages(season)
        war = load_war(season - 1)
        self.prior_war = war[war["season"].eq(season - 1)].set_index(["player_id", "role"])["war"]
        self.events = {s: expected.pa_events(s) for s in (*self.prior, season)} if model.expected_model else {}
        self.prior_expected = {s: expected.expected_counts(self.events[s], self.prior_inputs[s], model.expected_model, s) for s in self.prior} if self.events else {}

    @property
    def pitches(self) -> pd.DataFrame:
        """The season's regular-season pitches, loaded on first use (a caller with pre-cut frames never needs them)."""
        if self._pitches is None:
            self._pitches = inputs.load_pitches(self.season)
            self._pitches["game_date"] = pd.to_datetime(self._pitches["game_date"]).dt.date
        return self._pitches

    def weights(self) -> dict[int, float]:
        return {self.season - back: w for back, w in self.model.window_weights.items()}

    def current_frames(self, as_of: date) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
        """(inputs, outcome counts, positions) from the season's pitches before ``as_of``; None before the first pitch."""
        before = self.pitches[self.pitches["game_date"] < as_of]
        if before.empty:
            return None
        return inputs.statcast_inputs(before, self.season), inputs.count_outcomes(before, self.season), bio.positions([self.season], before=as_of)


def run_value(counts: pd.DataFrame, group: pd.Series, model: ValueModel) -> pd.DataFrame:
    """Per-full-season run value and sample from outcome counts, for each player's expected role."""
    rows = features.role_rows(counts, group).set_index("player_id")
    n = rows[list(OUTCOMES)].sum(axis=1)
    per_pa = rows[list(OUTCOMES)].div(n.where(n > 0), axis=0).to_numpy(float) @ np.array([model.linear_weights[k] for k in OUTCOMES])
    innings = np.where(group.reindex(rows.index).eq("SP"), STARTER_INNINGS, RELIEVER_INNINGS)
    runs = np.where(rows["role"].eq("bat"), per_pa * FULL_SEASON_PA, -per_pa * model.pa_per_ip * innings)
    return pd.DataFrame({"runs": runs, "n": n.to_numpy(float), "role": rows["role"].to_numpy()}, index=rows.index)


def grade_as_of(m: SeasonMaterial, as_of: date, current: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Grade, shrunk runs, sample, pool and group per player from inputs before ``as_of``.

    ``current`` supplies the season's pre-cut (inputs, counts, positions) frames; by default they are built from the pitches.
    """
    current = current if current is not None else m.current_frames(as_of)
    weights = m.weights()
    frames, counts, pos = dict(m.prior_inputs), dict(m.prior_counts), m.prior_positions
    if current is not None:
        frames[m.season], counts[m.season], season_positions = current
        pos = pd.concat([pos, season_positions], ignore_index=True)
    players = pd.Index(pd.concat([f["player_id"] for f in frames.values()]).unique())
    for s in m.prior:
        frames[s] = pd.concat([frames[s], minors.translated_inputs(m.prior_aaa_inputs[s], m.model.aaa_input_offsets, m.model.shrink_strength, m.model.aaa_scale, players)], ignore_index=True)
    win = inputs.window(frames, weights)
    groups = bio.primary_position(pos, weights).set_index("player_id")["position_group"]
    values, _ = features.wide_inputs(win)
    group = groups.reindex(values.index).fillna("DH")
    pool = pd.Series(features.pool_of(group), index=values.index)
    z = features.oriented_z(values, pool)
    expected_by_season = None
    if m.model.expected_model:
        expected_by_season = dict(m.prior_expected)
        if m.season in counts:
            events = m.events[m.season]
            expected_by_season[m.season] = expected.expected_counts(events[events["game_date"] < as_of], frames[m.season], m.model.expected_model, m.season)
    rates = features.outcome_rates(m.model, counts, groups, weights, m.prior_pseudo, expected_by_season)
    sample = features.role_rows(rates, group).set_index("player_id")["n"].reindex(values.index).fillna(0.0)
    leverage = values["workload.leverage"] if "workload.leverage" in values.columns else None
    shrunk = features.shrunk_runs(rates, m.model, group, leverage).reindex(values.index)
    comps, _ = features.trait_composites(z, pool, m.model)
    x = features.grade_features(comps, shrunk, pool, m.ages, z if m.model.direct_batting else None)
    batting = (features.project_runs(x, group, m.model, component="batting") if m.model.component_grades
               else features.project_runs(x, group, m.batting_model))
    out = pd.DataFrame({"grade": features.project_runs(x, group, m.model), "grade_batting": batting,
                        "shrunk_only": x["shrunk_runs"], "sample": sample, "group": group}, index=values.index)
    out["fit_pool"] = np.where(group.isin(features.PITCHER_GROUPS), group, features.HITTER_POOL)
    out["to_date_runs"] = run_value(counts[m.season], group, m.model)["runs"].reindex(out.index) if m.season in counts else np.nan
    rates_by_season = {s: season_rates(c) for s, c in counts.items()}
    lw = {s: m.model.linear_weights for s in counts}
    bat = marcel(rates_by_season, lw, m.season, "bat")
    pit = marcel(rates_by_season, lw, m.season, "pit")
    out["marcel"] = np.where(out["fit_pool"].eq(features.HITTER_POOL), bat.reindex(out.index), pit.reindex(out.index))
    role = np.where(out["fit_pool"].eq(features.HITTER_POOL), "bat", "pit")
    out["prior_fwar"] = m.prior_war.reindex(pd.MultiIndex.from_arrays([out.index, role])).to_numpy()
    key = pd.MultiIndex.from_arrays([out.index, role])
    out["fwar_pace"] = war_pace(m.season, as_of).reindex(key).to_numpy()
    for system, values in projections(m.season, as_of).items():
        out[system] = values.reindex(key).to_numpy()
    return out


def rest_of_season(m: SeasonMaterial, as_of: date, group: pd.Series) -> pd.DataFrame:
    after = m.pitches[m.pitches["game_date"] >= as_of]
    return run_value(inputs.count_outcomes(after, m.season), group, m.model)


# Season-to-date fWAR from cached FanGraphs date-range leaderboards.


def war_urls(season: int, as_of: date) -> dict[str, str]:
    end = (as_of - timedelta(days=1)).isoformat()
    return {role: FANGRAPHS.format(stats=stats, season=season, end=end) for role, stats in (("bat", "bat"), ("pit", "pit"))}


def war_to_date_path(season: int, as_of: date):
    return CACHE_DIR / f"fangraphs_war_to_date_{season}_{as_of.isoformat()}.json"


def war_pace(season: int, as_of: date) -> pd.Series:
    """fWAR per 600 PA (hitters) or per 180 IP (pitchers) through the day before ``as_of``; empty when not cached."""
    path = war_to_date_path(season, as_of)
    if not path.exists():
        return pd.Series(dtype="float64", index=pd.MultiIndex.from_arrays([[], []], names=["player_id", "role"]))
    payload = json.loads(path.read_text())
    parts = []
    for role, volume, unit in (("bat", "PA", FULL_SEASON_PA), ("pit", "IP", STARTER_INNINGS)):
        rows = payload[role]["data"] if isinstance(payload[role], dict) else payload[role]
        frame = pd.DataFrame([{"player_id": r.get("xMLBAMID"), "war": r.get("WAR"), "volume": r.get(volume)} for r in rows])
        frame = frame.dropna(subset=["player_id"]).astype({"player_id": "int64"})
        if role == "pit":
            whole = np.floor(frame["volume"])
            frame["volume"] = whole + (frame["volume"] - whole) * 10 / 3  # FanGraphs writes 12.1 for twelve and a third
        frame["pace"] = frame["war"] / frame["volume"].where(frame["volume"] > 0) * unit
        frame["role"] = role
        parts.append(frame[["player_id", "role", "pace"]])
    return pd.concat(parts).drop_duplicates(["player_id", "role"]).set_index(["player_id", "role"])["pace"]


# Preseason projection systems from cached FanGraphs pages.


def _fangraphs_ids() -> dict[str, int]:
    """FanGraphs playerid to MLBAM id, from every cached projection set that carries both (the 2024 pages carry only the former)."""
    ids: dict[str, int] = {}
    for path in sorted(CACHE_DIR.glob("fangraphs_projection_*.json")):
        for r in json.loads(path.read_text())["rows"]:
            if r.get("xMLBAMID") and r.get("playerid"):
                ids[str(r["playerid"])] = int(r["xMLBAMID"])
    return ids


def projection_page(system: str, role: str, season: int, as_of: date | None = None):
    """Cache path of a system's page: the preseason set, or the rest-of-season snapshot dated ``as_of``."""
    if system in ROS_SYSTEMS:
        return CACHE_DIR / f"fangraphs_projection_{ROS_SYSTEMS[system]}_{role}_{season}_{as_of.isoformat()}.json"
    return CACHE_DIR / f"fangraphs_projection_{PRESEASON_PAGES[system][role == 'pit']}_{role}_{season}.json"


def projection_stats(season: int, as_of: date | None = None) -> dict[str, pd.DataFrame]:
    """Each cached system's projected statistics by (player_id, role): wOBA, WAR and PA for hitters; ERA, FIP, WAR and IP for pitchers.

    Preseason sets always; rest-of-season sets when a snapshot dated ``as_of`` is cached (hitters only).
    """
    ids = _fangraphs_ids()

    def read(path, role: str) -> pd.DataFrame:
        rows = json.loads(path.read_text())["rows"]
        frame = pd.DataFrame([{"player_id": r.get("xMLBAMID") or ids.get(str(r.get("playerid"))), **{c: r.get(c) for c in PROJECTED_STATS[role]}} for r in rows])
        frame = frame.dropna(subset=["player_id"]).astype({"player_id": "int64"}).drop_duplicates("player_id")
        index = pd.MultiIndex.from_arrays([frame["player_id"], [role] * len(frame)], names=["player_id", "role"])
        return frame[list(PROJECTED_STATS[role])].astype(float).set_index(index)

    out: dict[str, pd.DataFrame] = {}
    systems = [*(ROS_SYSTEMS if as_of is not None else ()), *PRESEASON_PAGES]
    for system in systems:
        roles = ("bat",) if system in ROS_SYSTEMS else ("bat", "pit")
        parts = [read(path, role) for role in roles if (path := projection_page(system, role, season, as_of)).exists()]
        if parts:
            out[system] = pd.concat(parts)
    return out


def projections(season: int, as_of: date | None = None) -> dict[str, pd.Series]:
    """Each cached system's ordering: wOBA for hitters, FIP (sign flipped) for pitchers, by (player_id, role)."""
    out: dict[str, pd.Series] = {}
    for system, stats in projection_stats(season, as_of).items():
        bat = stats.index.get_level_values("role") == "bat"
        value = np.where(bat, stats.get("wOBA", np.nan), -stats.get("FIP", np.nan))
        out[system] = pd.Series(value, index=stats.index).dropna()
    return out


# Scoring.


def checkpoint(m: SeasonMaterial, as_of: date) -> pd.DataFrame:
    """Every graded player at ``as_of`` with the rest-of-season target, its sample, and whether he is scored.

    ``eligible`` means the window held ``MIN_FIT_SAMPLE`` weighted PA or batters faced and the rest of the season ``MIN_TARGET_SAMPLE``.
    """
    graded = grade_as_of(m, as_of)
    target = rest_of_season(m, as_of, graded["group"])
    graded["target"] = target["runs"].reindex(graded.index)
    graded["target_n"] = target["n"].reindex(graded.index).fillna(0.0)
    graded["eligible"] = (graded["sample"] >= MIN_FIT_SAMPLE) & (graded["target_n"] >= MIN_TARGET_SAMPLE) & graded["target"].notna()
    return graded


def score_month(m: SeasonMaterial, as_of: date) -> list[dict]:
    frame = checkpoint(m, as_of)
    frame = frame[frame["eligible"]]
    columns = ["grade", "grade_batting", "shrunk_only", "to_date_runs", "marcel", "prior_fwar", "fwar_pace", *PROJECTION_SYSTEMS, *ROS_SYSTEMS]
    rows = []
    for pool in POOLS:
        members = frame[frame["fit_pool"].eq(pool)]
        row = {"season": m.season, "as_of": as_of.isoformat(), "pool": pool, "n": len(members)}
        for column in columns:
            if column == "grade_batting" and pool != features.HITTER_POOL:
                continue  # pitcher weights are the same in both models
            row[column] = spearman(members[column], members["target"]) if column in members else np.nan
        rows.append(row)
        log.info(" ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
    return rows


def snapshot_dates(season: int) -> list[date]:
    """Dates of cached rest-of-season snapshots for a season."""
    found = set()
    for typ in ROS_SYSTEMS.values():
        for path in CACHE_DIR.glob(f"fangraphs_projection_{typ}_bat_{season}_*.json"):
            found.add(date.fromisoformat(path.stem.rsplit("_", 1)[1]))
    return sorted(found)


def evaluate(seasons: list[int], months: tuple[int, ...] = MONTHS, snapshots: bool = False) -> pd.DataFrame:
    rows = []
    for season in seasons:
        m = SeasonMaterial(season, *grade_models(season - 2))
        start, end = bio.season_dates(season)
        dates = snapshot_dates(season) if snapshots else [start] + [date(season, month, 1) for month in months]
        for as_of in dates:
            if not start <= as_of <= end:
                continue
            rows.extend(score_month(m, as_of))
    return pd.DataFrame(rows)


def summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean correlation per pool and projection, and how many month starts it beat each baseline."""
    baselines = ["shrunk_only", "to_date_runs", "marcel", "prior_fwar", "fwar_pace", *PROJECTION_SYSTEMS, *ROS_SYSTEMS]
    out = []
    for pool, g in frame.groupby("pool"):
        for projection in ("grade", "grade_batting"):
            if projection not in g or g[projection].isna().all():
                continue
            row = {"pool": pool, "projection": projection, "months": len(g), "mean": g[projection].mean()}
            for b in baselines:
                paired = g[[projection, b]].dropna()
                row[b] = paired[b].mean() if len(paired) else np.nan
                row[f"beats_{b}"] = f"{int((paired[projection] > paired[b]).sum())}/{len(paired)}"
            out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Month-start forward evaluation of the Grade.")
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025, 2026])
    ap.add_argument("--months", type=int, nargs="+", default=list(MONTHS))
    ap.add_argument("--war-urls", type=int, metavar="SEASON", help="Print the FanGraphs URLs to cache for the fWAR pace baseline")
    ap.add_argument("--snapshots", action="store_true", help="Score on the dates of cached rest-of-season snapshots instead of month starts")
    args = ap.parse_args()
    if args.war_urls:
        for month in args.months:
            as_of = date(args.war_urls, month, 1)
            for role, url in war_urls(args.war_urls, as_of).items():
                print(f"{war_to_date_path(args.war_urls, as_of).name} {role} {url}")
        return
    frame = evaluate(args.seasons, tuple(args.months), snapshots=args.snapshots)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "forward_snapshot_evaluation.json" if args.snapshots else RESULT_PATH).write_text(json.dumps({"rows": frame.to_dict("records"), "summary": summary(frame).to_dict("records")}, indent=1, default=str))
    pd.set_option("display.width", 200)
    print(frame.round(3).to_string(index=False))
    print()
    print(summary(frame).round(3).to_string(index=False))


if __name__ == "__main__":
    setup_logging()
    main()
