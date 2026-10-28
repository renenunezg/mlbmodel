"""The grade against the projection systems at the forward checkpoints, in the block the site publishes.

At each season's opening day, the first of April to September, and every cached rest-of-season
snapshot date, the grade is rebuilt from what was knowable that morning (``forward``) and ranked
against ZiPS, THE BAT (X for hitters) and Steamer on the rest-of-season target: the batting-only
grade against projected wOBA on batting runs per 600 PA, the full grade against the projected WAR
rate on the fielding-inclusive runs, and the pitcher grade against projected ERA, FIP and WAR per
inning on runs prevented per 180 or 65 innings. Each comparison pools the per-checkpoint Spearman
correlations on the players the grade and that system both cover, over the preseason checkpoints,
the in-season month starts, the thin and established history buckets, each season's preseason and
the snapshot dates; the interval is a player-cluster bootstrap of the mean difference.

``fit`` attaches the block to ``model.evaluation["forward"]``; ``mlb.player_value_evaluation``
serves it to the site's performance page, so its keys are a contract.

Usage:
    python -m v2.players.head_to_head --last-dev-season 2025 [--output PATH]
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backend.log import setup_logging
from v2.players import bio, features, forward, inputs
from v2.players.fit import fielding_running_runs, season_rates
from v2.players.inputs import CACHE_DIR
from v2.players.runs import FULL_SEASON_PA, PUBLISHED_POSITIONAL_RUNS, RELIEVER_INNINGS, STARTER_INNINGS

log = logging.getLogger(__name__)

FIRST_SEASON = 2024  # the first season with three seasons of development pairs behind its forward grade model
RESAMPLES = 4000
SEED = 7
MIN_CHECKPOINT_PLAYERS = 30  # a checkpoint with fewer common players (September relievers) does not enter a comparison
# Weighted plate appearances or batters faced from the seasons before the evaluated one below which a player's history is thin.
THIN_CUTOFF = {features.HITTER_POOL: 300.0, "SP": 300.0, "RP": 200.0}
# The systems' WAR carries a positional adjustment the runs target lacks; the position-neutral comparison removes it at the conventional rate.
RUNS_PER_WIN = 10.0
# Pages the cache is known not to hold; any other missing page stops the fit.
KNOWN_GAPS = {("zips", "pit", 2024)}
SYSTEM_LABELS = {"zips": "ZiPS", "steamer": "Steamer", "the_bat": "THE BAT X (hitters) / THE BAT (pitchers)",
                 "zips_ros": "ZiPS rest-of-season snapshot", "the_bat_ros": "THE BAT X rest-of-season snapshot"}
METHOD = "month_start_forward_vs_projection_systems"
SOURCE = ("v2.players.forward grade models rebuilt per season; cached FanGraphs projection pages (Wayback captures); "
          f"paired and cluster bootstraps, {RESAMPLES:,} resamples")
STATISTIC = "Spearman rank correlation; diff is grade minus system; ci95 from the bootstrap; wins are checkpoints the grade ranked better"
POPULATION = "100 weighted PA or BF of history in the window and 100 PA or BF from the checkpoint to season end; common players per comparison"
CAVEATS = [
    "Full hitter target uses the season's Savant fielding, framing, blocking and baserunning boards prorated by 600 / season PA; boards cannot be cut by date.",
    "Raw WAR per PA carries the positional adjustment the target lacks; the position-neutral comparison removes the published positional runs by the grade's position group at 10 runs per win.",
    "The target's fielding is OAA-based, the same metric family the grade reads; the systems' WAR uses FanGraphs' own defensive inputs.",
    "Population: 100 weighted PA or BF of MLB history and 100 PA or BF after the date; rookies without MLB history are not scored; rest-of-season gate conditions on survival.",
    "2024 FanGraphs pages carry no MLBAM ids and are joined through a FanGraphs-to-MLBAM map from the 2025 and 2026 pages (99.3 percent or better of the eligible 2024 population joins).",
    "System pages are Wayback Machine captures; some postdate opening day (2024 hitters 2024-04-07, 2024 pitchers June 2024, 2025 ZiPS pitchers 2025-08-08, 2026 ZiPS April 2026); contents are consistent with frozen preseason sets but unverifiable from the cache.",
    "No ZiPS pitcher page for 2024; THE BAT projects fewer pitchers, so common-player n is smaller for pitchers.",
    "Month-start checkpoints give an updating grade current-season pitches while preseason projections stay fixed; the dated rest-of-season snapshots are the equal-information in-season test.",
    "Forward grade models are fitted on pairs ending before each evaluated season with the live design; the design itself was chosen with 2026 outcomes in hand, and the live row has no untouched holdout.",
    "Starters: FanGraphs pitcher WAR is FIP-based with park and leverage terms; the per-180-IP runs target rewards ball-in-play prevention, which FIP excludes.",
]

BATTING_TARGET = "rest-of-season batting runs above average per 600 PA"
FULL_TARGET = "rest-of-season runs above average per 600 PA including fielding and baserunning"
# comparison -> (grade column, system statistic column, target column, grade label, system statistic label, target label)
HITTER_COMPARISONS = {
    "batting_grade_vs_woba": ("grade_batting", "woba", "target", "grade_batting (batting-only refit, not published)", "projected wOBA", BATTING_TARGET),
    "full_grade_vs_war_rate": ("grade", "war_pa", "target_full", "grade", "projected WAR per PA", FULL_TARGET),
    "full_grade_vs_woba_batting_target": ("grade", "woba", "target", "grade", "projected wOBA", BATTING_TARGET),
    "full_grade_vs_war_rate_position_neutral": ("grade", "war_neutral", "target_full", "grade",
                                                "projected WAR per 600 PA minus published positional runs at 10 runs per win", FULL_TARGET),
}
PITCHER_COMPARISONS = {
    "grade_vs_era": ("grade", "era", "target", "grade", "projected ERA (lower is better)"),
    "grade_vs_fip": ("grade", "fip", "target", "grade", "projected FIP (lower is better)"),
    "grade_vs_war_ip": ("grade", "war_ip", "target", "grade", "projected WAR per IP"),
}


def seasons_for(last_dev_season: int) -> list[int]:
    """The seasons a model developed through ``last_dev_season`` is evaluated on: every completed one through its last development target."""
    return list(range(FIRST_SEASON, last_dev_season + 2))


def require_material(seasons: list[int]) -> None:
    """Stop before the fit when a season's Statcast or a system's page is missing, so a row is never published without the block by accident."""
    missing = [f"statcast_{s}.parquet" for s in seasons if not (inputs.STATCAST_DIR / f"statcast_{s}.parquet").exists()]
    for season in seasons:
        for system in forward.PRESEASON_PAGES:
            for role in ("bat", "pit"):
                path = forward.projection_page(system, role, season)
                if not path.exists() and (system, role, season) not in KNOWN_GAPS:
                    missing.append(path.name)
    if missing:
        raise FileNotFoundError("Forward head-to-head material missing (run with --skip-forward to publish without it): " + ", ".join(missing))


# Checkpoint frames: one row per eligible player, checkpoint and season.


def system_columns(frame: pd.DataFrame, stats: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Each system's statistics on the frame's players, oriented so higher is better: wOBA, WAR per PA, position-neutral WAR per 600 PA, ERA, FIP, WAR per IP."""
    role = np.where(frame["fit_pool"].eq(features.HITTER_POOL), "bat", "pit")
    key = pd.MultiIndex.from_arrays([frame.index, role])
    positional = frame["group"].map(PUBLISHED_POSITIONAL_RUNS).fillna(0.0).to_numpy() / RUNS_PER_WIN
    out = {}
    for system, table in stats.items():
        s = table.reindex(key)
        if "wOBA" in s:
            pa = s["PA"].where(s["PA"] > 0).to_numpy()
            out[f"{system}.woba"] = s["wOBA"].to_numpy()
            out[f"{system}.war_pa"] = s["WAR"].to_numpy() / pa
            out[f"{system}.war_neutral"] = s["WAR"].to_numpy() / pa * FULL_SEASON_PA - positional
        if "ERA" in s:
            out[f"{system}.era"] = -s["ERA"].to_numpy()
            out[f"{system}.fip"] = -s["FIP"].to_numpy()
            out[f"{system}.war_ip"] = s["WAR"].to_numpy() / s["IP"].where(s["IP"] > 0).to_numpy()
    return pd.DataFrame(out, index=frame.index)


def full_target(season: int, frame: pd.DataFrame) -> pd.Series:
    """The batting target plus the season's fielding and baserunning runs per 600 PA for hitters; the batting target itself for pitchers."""
    board = inputs.season_inputs(season).pivot_table(index="player_id", columns="input", values="value", aggfunc="first")
    rates = season_rates(inputs.outcome_counts(season))
    n = rates[rates["role"].eq("bat")].set_index("player_id")["n"]
    fld, run = fielding_running_runs(board, n, frame["group"])
    extra = (fld + run).reindex(frame.index).fillna(0.0)
    return frame["target"] + extra.where(frame["fit_pool"].eq(features.HITTER_POOL), 0.0)


def checkpoint_dates(season: int) -> tuple[date, list[date], list[date]]:
    """(opening day, in-season month starts, rest-of-season snapshot dates) inside the regular season."""
    start, end = bio.season_dates(season)
    months = [d for month in forward.MONTHS if start < (d := date(season, month, 1)) <= end]
    snapshots = [d for d in forward.snapshot_dates(season) if start <= d <= end]
    return start, months, snapshots


def checkpoint_frames(seasons: list[int], **switches) -> tuple[pd.DataFrame, dict]:
    """Every checkpoint's eligible players with grades, targets and system statistics, and each season's forward grade model summary."""
    frames, models = [], {}
    for season in seasons:
        model, batting = forward.grade_models(season - 2, **switches)
        models[str(season)] = {"model_version": model.model_version, "dev_pairs": [f"{s}->{s + 1}" for s in model.dev_seasons[:-1]],
                               "window_weights": {str(k): v for k, v in model.window_weights.items()},
                               "shrink_strength": model.shrink_strength, "aaa_scale": model.aaa_scale}
        m = forward.SeasonMaterial(season, model, batting)
        opener, months, snapshots = checkpoint_dates(season)
        history = None
        for as_of in sorted({opener, *months, *snapshots}):
            graded = forward.checkpoint(m, as_of)
            if as_of == opener:
                history = graded["sample"]  # no current-season pitches yet: the window holds only the seasons before
            frame = graded.loc[graded["eligible"], ["grade", "grade_batting", "sample", "group", "fit_pool", "target"]].copy()
            frame["history"] = history.reindex(frame.index).fillna(0.0)
            frame["target_full"] = full_target(season, frame)
            frame = frame.join(system_columns(frame, forward.projection_stats(season, as_of)))
            frame["season"], frame["as_of"] = season, as_of.isoformat()
            frame["preseason"], frame["month_start"], frame["snapshot"] = as_of == opener, as_of in months, as_of in snapshots
            frame.index.name = "player_id"
            frames.append(frame.reset_index())
            log.info(f"{season} {as_of}: {len(frame)} eligible players")
    return pd.concat(frames, ignore_index=True), models


# Pooled comparisons with the player-cluster bootstrap.


def _ranks(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Average ranks of x under integer weights w (resamples by players), ties shared across tied players."""
    order = np.argsort(x, kind="stable")
    xs = x[order]
    starts = np.r_[0, np.flatnonzero(np.diff(xs)) + 1]
    group_weight = np.add.reduceat(w[:, order], starts, axis=1)
    group_rank = np.cumsum(group_weight, axis=1) - (group_weight - 1) / 2.0
    ranked = np.repeat(group_rank, np.diff(np.r_[starts, len(xs)]), axis=1)
    out = np.empty_like(ranked)
    out[:, order] = ranked
    return out


def _weighted_corr(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    sw = w.sum(axis=1, keepdims=True)
    ma, mb = (w * a).sum(axis=1, keepdims=True) / sw, (w * b).sum(axis=1, keepdims=True) / sw
    cov = (w * (a - ma) * (b - mb)).sum(axis=1)
    return cov / np.sqrt((w * (a - ma) ** 2).sum(axis=1) * (w * (b - mb) ** 2).sum(axis=1))


def spearman_draws(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Spearman correlation of x and y under each row of player weights."""
    return _weighted_corr(_ranks(x, w), _ranks(y, w), w)


def compare(frame: pd.DataFrame, grade: str, system: str, target: str, checkpoints: list[str]) -> dict | None:
    """Mean over checkpoints of rho(grade) minus rho(system) on their common players, with a player-cluster bootstrap interval.

    None when no checkpoint has ``MIN_CHECKPOINT_PLAYERS`` common players.
    """
    rows = frame[frame["as_of"].isin(checkpoints)].dropna(subset=[grade, system, target])
    sizes = rows.groupby("as_of").size()
    cps = sorted(sizes.index[sizes >= MIN_CHECKPOINT_PLAYERS])
    if not cps:
        return None
    rows = rows[rows["as_of"].isin(cps)]
    players = pd.Index(rows["player_id"].unique())
    rng = np.random.default_rng(SEED)
    draws = np.stack([np.bincount(rng.integers(0, len(players), len(players)), minlength=len(players)) for _ in range(RESAMPLES)]).astype(float)
    diff, rho_grade, rho_system, wins = np.zeros(RESAMPLES), [], [], 0
    for cp in cps:
        g = rows[rows["as_of"].eq(cp)]
        x, s, y = g[grade].to_numpy(float), g[system].to_numpy(float), g[target].to_numpy(float)
        one = np.ones((1, len(g)))
        a, b = spearman_draws(x, y, one)[0], spearman_draws(s, y, one)[0]
        rho_grade.append(a), rho_system.append(b)
        wins += a > b
        w = draws[:, players.get_indexer(g["player_id"])]
        diff += spearman_draws(x, y, w) - spearman_draws(s, y, w)
    lo, hi = np.nanpercentile(diff / len(cps), [2.5, 97.5])
    return {"rho_grade": round(float(np.mean(rho_grade)), 4), "rho_system": round(float(np.mean(rho_system)), 4),
            "diff": round(float(np.mean(rho_grade) - np.mean(rho_system)), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "wins": int(wins), "checkpoints": len(cps), "mean_n": int(round(float(sizes[cps].mean())))}


def _systems(frame: pd.DataFrame, grade: str, statistic: str, target: str, checkpoints: list[str], systems) -> dict:
    out = {system: compare(frame, grade, f"{system}.{statistic}", target, checkpoints) for system in systems}
    return {k: v for k, v in out.items() if v is not None}


def comparison_block(frame: pd.DataFrame, pool: str, grade: str, statistic: str, target: str, labels: tuple[str, str, str],
                     preseason: list[str], months: list[str], snapshots: list[str]) -> dict:
    """One comparison's pooled results: preseason and in-season, the history buckets, each season's preseason, and the snapshots for hitters."""
    cutoff = THIN_CUTOFF[pool]
    thin, established = frame[frame["history"] < cutoff], frame[frame["history"] >= cutoff]
    block = {"grade_column": labels[0], "system_statistic": labels[1], "target": labels[2],
             "preseason": _systems(frame, grade, statistic, target, preseason, forward.PROJECTION_SYSTEMS),
             "in_season": _systems(frame, grade, statistic, target, months, forward.PROJECTION_SYSTEMS),
             "buckets": {phase: {"thin": _systems(thin, grade, statistic, target, cps, forward.PROJECTION_SYSTEMS),
                                 "established": _systems(established, grade, statistic, target, cps, forward.PROJECTION_SYSTEMS),
                                 "cutoff_weighted_pa": int(cutoff)}
                         for phase, cps in (("preseason", preseason), ("in_season", months))},
             "by_season_preseason": {cp[:4]: _systems(frame, grade, statistic, target, [cp], forward.PROJECTION_SYSTEMS) for cp in preseason}}
    if pool == features.HITTER_POOL:
        block["rest_of_season_snapshots"] = _systems(frame, grade, statistic, target, snapshots, forward.ROS_SYSTEMS)
    return block


def forward_block(last_dev_season: int, **switches) -> dict:
    """The published forward block for a model developed through ``last_dev_season``, built with the same hitter-grade switches."""
    seasons = seasons_for(last_dev_season)
    require_material(seasons)
    return assemble(*checkpoint_frames(seasons, **switches), seasons)


def assemble(frame: pd.DataFrame, models: dict, seasons: list[int]) -> dict:
    """The block from the checkpoint frames: every pool's comparisons plus the method, population and material notes."""
    sizes = frame.groupby("as_of").size()
    frame = frame[frame["as_of"].map(sizes) >= MIN_CHECKPOINT_PLAYERS]  # a mid-September snapshot with a handful of players is no checkpoint
    preseason = sorted(frame.loc[frame["preseason"], "as_of"].unique())
    months = sorted(frame.loc[frame["month_start"], "as_of"].unique())
    snapshots = sorted(frame.loc[frame["snapshot"], "as_of"].unique())
    pools = {}
    for pool in forward.POOLS:
        members = frame[frame["fit_pool"].eq(pool)]
        pools[pool] = {}
        if pool == features.HITTER_POOL:
            for name, (grade, statistic, target, *labels) in HITTER_COMPARISONS.items():
                pools[pool][name] = comparison_block(members, pool, grade, statistic, target, tuple(labels), preseason, months, snapshots)
        else:
            innings = STARTER_INNINGS if pool == "SP" else RELIEVER_INNINGS
            for name, (grade, statistic, target, *labels) in PITCHER_COMPARISONS.items():
                labels = (*labels, f"rest-of-season runs prevented versus average per {innings:.0f} IP")
                pools[pool][name] = comparison_block(members, pool, grade, statistic, target, labels, preseason, months, snapshots)
    return {"method": METHOD, "source": SOURCE, "statistic": STATISTIC, "population": POPULATION, "seasons": seasons,
            "openers": {cp[:4]: cp for cp in preseason},
            "checkpoints": {"preseason": len(preseason), "in_season_month_starts": len(months), "rest_of_season_snapshots": len(snapshots)},
            "systems": SYSTEM_LABELS, "grade_models": models, "pools": pools, "caveats": CAVEATS,
            "computed_at": datetime.now(UTC).isoformat(timespec="seconds")}


def main() -> None:
    ap = argparse.ArgumentParser(description="The grade against the projection systems at the forward checkpoints.")
    ap.add_argument("--last-dev-season", type=int, required=True, help="The fit's development season; the block covers every completed season through its next-year target")
    ap.add_argument("--output", type=Path, default=CACHE_DIR / "forward_head_to_head.json")
    args = ap.parse_args()
    block = forward_block(args.last_dev_season)
    args.output.write_text(json.dumps(block, indent=1))
    for pool, comparisons in block["pools"].items():
        for name, c in comparisons.items():
            for phase in ("preseason", "in_season"):
                print(pool, name, phase, " | ".join(f"{s} {r['rho_grade']:.3f} vs {r['rho_system']:.3f} {r['diff']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] {r['wins']}/{r['checkpoints']}"
                                                   for s, r in c[phase].items()))


if __name__ == "__main__":
    setup_logging()
    main()
