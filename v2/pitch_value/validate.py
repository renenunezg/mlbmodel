"""Does a season's pitch value say more about next season's runs allowed than ERA or FIP do?

Each pair is a pitcher's season and his next one. The target is runs allowed per nine innings in the second
season; the predictors are the first season's ERA, FIP, xFIP and pitch value per 100 pitches (raw, with expected
contact, fully adjusted, and context-neutral). Correlations are over pairs; intervals resample pitchers, so a
pitcher's several pairs move together.

The pairs are split by first season: design choices are made on the development pairs and the confirmation
pairs are read once a design is fixed.

Usage:
    python -m v2.pitch_value.validate [--confirm]

Reads cache/pitch_value/pitcher_seasons.parquet (run ``python -m v2.pitch_value.run`` first) and the MLB
Stats API season lines, kept under cache/pitch_value/lines_{season}.parquet once a season is over. With every
season's lines cached, a run needs neither the network nor the database configuration.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date

import numpy as np
import pandas as pd

from backend.data.game_types import regular_season_pitches
from backend.log import setup_logging
from v2.pitch_value.pitches import STATCAST_DIR
from v2.pitch_value.run import OUT_DIR

log = logging.getLogger(__name__)

# Innings required in both seasons of a pair. The middle one is the headline; the others show it is not the cut.
INNINGS_CUTS = (40, 80, 120)
HEADLINE_CUT = 80
PREDICTORS = ("era", "fip", "xfip", "raw_per_100", "expected_per_100", "per_100", "neutral_per_100")
BASELINES = ("era", "fip", "xfip")
HEADLINE = "neutral_per_100"
# First seasons of the pairs a design may be chosen on, and of the pairs kept back to confirm it.
DEVELOPMENT = (2021, 2022, 2023)
CONFIRMATION = (2024, 2025)
# Statcast batted-ball types FanGraphs counts as fly balls for xFIP (infield flies included).
FLY_BALLS = ("fly_ball", "popup")
BOOTSTRAP_DRAWS = 2000
# Month and day from which a season's regular-season lines no longer change (the regular season ends by early October).
LINES_FINAL = (11, 1)


def fly_balls(season: int) -> pd.Series:
    """Fly balls allowed per pitcher, from the Statcast cache."""
    frame = pd.read_parquet(STATCAST_DIR / f"statcast_{season}.parquet", columns=["game_type", "game_date", "pitcher", "bb_type"])
    frame = regular_season_pitches(frame, season)
    return frame[frame["bb_type"].isin(FLY_BALLS)].groupby("pitcher").size().astype(float)


def official(season: int) -> pd.DataFrame:
    """ERA, FIP, xFIP, runs per nine and innings for every MLB pitcher in a season.

    xFIP is FIP with the pitcher's home runs replaced by his fly balls times the league's home runs per fly ball.
    """
    path = OUT_DIR / f"lines_{season}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    # Imported here: the module loads the database engine, which a run over cached lines does not need.
    from v2 import season_stats

    lines = season_stats.official_lines(season, "pitching", (1,))
    out = season_stats.pitching_production(lines, season_stats.fip_constants(lines))
    runs = lines.assign(runs=lines["runs"].astype(float)).groupby("player_id")["runs"].sum()
    flies = fly_balls(season).reindex(out.index).fillna(0.0)
    expected_hr = flies * out["hr"].sum() / flies.sum()
    xfip = out["fip"] + 39.0 * (expected_hr - out["hr"]) / out["outs"].where(out["outs"] > 0)
    out = pd.DataFrame({"season": season, "innings": out["outs"] / 3.0, "era": out["era"], "fip": out["fip"], "xfip": xfip,
                        "ra9": 27.0 * runs / out["outs"].where(out["outs"] > 0)}).reset_index()
    if date.today() >= date(season, *LINES_FINAL):
        out.to_parquet(path, index=False)
    return out


def pairs(values: pd.DataFrame) -> pd.DataFrame:
    seasons = sorted(values["season"].unique())
    lines = pd.concat([official(s) for s in seasons], ignore_index=True)
    values = values.assign(raw_per_100=100 * values["raw_runs"] / values["pitches"],
                           expected_per_100=100 * values["expected_runs"] / values["pitches"])
    first = values.merge(lines, left_on=["season", "pitcher"], right_on=["season", "player_id"])
    second = lines.assign(season=lines["season"] - 1)[["season", "player_id", "innings", "ra9"]]
    return first.merge(second, on=["season", "player_id"], suffixes=("", "_next"))


def correlations(frame: pd.DataFrame, rng: np.random.Generator) -> dict:
    target = frame["ra9_next"].to_numpy()
    r = {p: float(np.corrcoef(frame[p], target)[0, 1]) for p in PREDICTORS}
    pitchers = frame["pitcher"].unique()
    rows = {p: np.flatnonzero(frame["pitcher"].to_numpy() == p) for p in pitchers}
    draws = {p: [] for p in PREDICTORS}
    for _ in range(BOOTSTRAP_DRAWS):
        index = np.concatenate([rows[p] for p in rng.choice(pitchers, len(pitchers))])
        sample = frame.iloc[index]
        for p in PREDICTORS:
            draws[p].append(np.corrcoef(sample[p], sample["ra9_next"])[0, 1])
    draws = {p: np.array(d) for p, d in draws.items()}
    interval = lambda d: [float(x) for x in np.percentile(d, [2.5, 97.5])]
    return {
        "pairs": len(frame), "pitchers": len(pitchers),
        "r": r, "r_interval": {p: interval(draws[p]) for p in PREDICTORS},
        "gain_over": {base: {"r": r[HEADLINE] - r[base], "interval": interval(draws[HEADLINE] - draws[base])}
                      for base in BASELINES},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true", help="also read the confirmation pairs")
    args = parser.parse_args()
    setup_logging()
    frame = pairs(pd.read_parquet(OUT_DIR / "pitcher_seasons.parquet"))
    rng = np.random.default_rng(0)
    sets = {"development": DEVELOPMENT} | ({"confirmation": CONFIRMATION, "all": DEVELOPMENT + CONFIRMATION} if args.confirm else {})
    report = {}
    for name, first_seasons in sets.items():
        for cut in INNINGS_CUTS:
            kept = frame[frame["season"].isin(first_seasons) & (frame["innings"] >= cut) & (frame["innings_next"] >= cut)]
            block = report.setdefault(name, {})[str(cut)] = correlations(kept, rng)
            log.info(f"{name}, {cut}+ innings both seasons: {block['pairs']} pairs, {block['pitchers']} pitchers")
            for p in PREDICTORS:
                lo, hi = block["r_interval"][p]
                log.info(f"  {p:>17}  r {block['r'][p]:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
            for base, gain in block["gain_over"].items():
                log.info(f"  {HEADLINE} over {base}: {gain['r']:+.3f}  [{gain['interval'][0]:+.3f}, {gain['interval'][1]:+.3f}]")
    (OUT_DIR / "validation.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
