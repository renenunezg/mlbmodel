"""xWAR: fWAR with the luck on contact taken out.

    hitters:   xWAR = fWAR - (realized gap - earned gap) / runs per win
    pitchers:  xWAR = fWAR + (13 / 9) x (home run gap - earned home runs) / runs per win

Usage:
    python -m v2.xwar.build [--season 2026]

fWAR is the Stats API's ``sabermetrics.war``, the number the site shows as Impact WAR, summed over a player's
hitting and pitching. A hitter's realized gap is what his tracked balls in play did against the batting run value
contact model (exit velocity, launch angle, his sprint speed), in runs. A pitcher's is the home runs he allowed
against the contact model's expected home runs; fWAR pitching is FIP, so home runs are the only contact in it,
and 13/9 is what FIP charges for one. The earned part is the share of the gap the player's own record supports
(``gap.earned``); the rest is treated as luck and removed. Strikeouts, walks, fielding, baserunning, position
and replacement level stay as fWAR has them.

Reads cache/batting_value/hitter_seasons.parquet, cache/pitch_value/pitcher_seasons.parquet and the cached fWAR
feed. Writes cache/xwar/player_seasons.parquet: one row per player and season from 2021.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from backend.log import setup_logging
from v2.batting_value import run as batting_run
from v2.pitch_value import run as pitch_run
from v2.pitch_value.pitches import STATCAST_DIR
from v2.players import production
from v2.players import runs as player_runs
from v2.xwar import gap

log = logging.getLogger(__name__)

OUT_DIR = STATCAST_DIR / "xwar"
# The first season with a contact model behind it.
FIRST_SEASON = 2021
# FIP = (13 HR + 3 (BB + HBP) - 2 K) / IP + constant, so a home run costs 13/9 of a run.
FIP_HOME_RUN_RUNS = 13 / 9
KEYS = ["player_id", "season"]


def hitter_side(hitters: pd.DataFrame, shrink: gap.Shrink | None) -> pd.DataFrame:
    """Per hitter-season: the realized gap on contact and the part of it he keeps, with the wOBA on contact he
    had and the one his contact called for (shown beside the gap; the gap in runs is what moves his WAR)."""
    out = pd.DataFrame({
        "player_id": hitters["batter"], "season": hitters["season"], "player_name": hitters["name"], "team": hitters["team"],
        "pa": hitters["pa"], "balls": hitters["tracked_balls"], "realized_gap_runs": hitters["gap_runs"],
        "woba_contact": hitters["woba_contact"], "expected_woba_contact": hitters["expected_woba_contact"],
    })
    return out.assign(earned_gap_runs=gap.earned(out, "player_id", "realized_gap_runs", "balls", shrink))


def pitcher_side(pitchers: pd.DataFrame, shrink: gap.Shrink | None) -> pd.DataFrame:
    """Per pitcher-season: home runs against expected home runs and the part of that gap he keeps."""
    out = pd.DataFrame({
        "player_id": pitchers["pitcher"], "season": pitchers["season"], "player_name": pitchers["name"], "team": pitchers["team"],
        "balls_allowed": pitchers["in_play"], "home_runs_allowed": pitchers["home_runs"],
        "expected_home_runs": pitchers["expected_home_runs"], "home_run_gap": pitchers["home_run_gap"],
    })
    return out.assign(earned_home_runs=gap.earned(out, "player_id", "home_run_gap", "balls_allowed", shrink))


def player_seasons(hitters: pd.DataFrame, pitchers: pd.DataFrame, war: pd.DataFrame, runs_per_win: dict[int, float]) -> pd.DataFrame:
    """One row per player and season in the fWAR feed that has a hitter or pitcher season behind it."""
    hit, pit = hitter_side(hitters, gap.HITTER), pitcher_side(pitchers, gap.PITCHER)
    total = war[war["season"] >= FIRST_SEASON].groupby(KEYS, as_index=False)["war"].sum()
    out = total.merge(hit, on=KEYS, how="left").merge(pit, on=KEYS, how="left", suffixes=("", "_pitcher"))
    for column in ("player_name", "team"):
        out[column] = out[column].fillna(out.pop(f"{column}_pitcher"))
    known = out["balls"].notna() | out["balls_allowed"].notna()
    sides = pd.concat([hit[KEYS], pit[KEYS]]).drop_duplicates()
    unmatched = len(sides.merge(total, on=KEYS, how="left", indicator=True).query("_merge == 'left_only'"))
    dropped = out[~known]
    log.info(f"{int(known.sum()):,} player-seasons; {len(dropped)} in the fWAR feed with no Statcast season dropped "
             f"(largest |WAR| {dropped['war'].abs().max() if len(dropped) else 0.0:.2f}); {unmatched} Statcast seasons with no fWAR row dropped")
    out = out[known].reset_index(drop=True)
    out["hitter_runs"] = (out["earned_gap_runs"] - out["realized_gap_runs"]).fillna(0.0)
    out["pitcher_runs"] = (FIP_HOME_RUN_RUNS * (out["home_run_gap"] - out["earned_home_runs"])).fillna(0.0)
    out["runs_per_win"] = out["season"].map(runs_per_win)
    out["xwar"] = out["war"] + (out["hitter_runs"] + out["pitcher_runs"]) / out["runs_per_win"]
    return out.sort_values(["season", "xwar"], ascending=[True, False]).reset_index(drop=True)


def build(last_season: int) -> pd.DataFrame:
    hitters = pd.read_parquet(batting_run.OUT_DIR / "hitter_seasons.parquet")
    pitchers = pd.read_parquet(pitch_run.OUT_DIR / "pitcher_seasons.parquet")
    seasons = range(FIRST_SEASON, last_season + 1)
    hitters, pitchers = hitters[hitters["season"] <= last_season], pitchers[pitchers["season"] <= last_season]
    missing = sorted(set(seasons) - (set(hitters["season"]) & set(pitchers["season"])))
    if missing:
        # A hitter's earned gap pools every season to date, so a hole in the history would misstate it silently.
        raise ValueError(f"the batting or pitch value season files lack {missing}")
    return player_seasons(hitters, pitchers, production.load_war(last_season), {s: player_runs.runs_per_win(s) for s in seasons})


def write(frame: pd.DataFrame, out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_dir / "player_seasons.parquet", index=False)
    log.info(f"wrote {out_dir / 'player_seasons.parquet'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, help="the last season to build (default: the latest in the batting value file)")
    args = parser.parse_args()
    setup_logging()
    season = args.season or int(pd.read_parquet(batting_run.OUT_DIR / "hitter_seasons.parquet", columns=["season"])["season"].max())
    write(build(season))


if __name__ == "__main__":
    main()
