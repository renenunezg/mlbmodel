"""Carry in Triple-A and the Florida State League, kept apart from the MLB study: each level is measured against its own
league with its own ballpark-season effects. Neither level has bat tracking, so there is no swing physics or lab here."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from v2.carry import data, expected
from v2.carry.study import rounded, split_half, year_to_year

log = logging.getLogger(__name__)

LEVELS = ("aaa", "fsl")
# Minor-league seasons are shorter and rosters turn over; Single-A hitters reach 90 mph in the air far less often.
QUALIFIED_AIR = {"aaa": 60, "fsl": 30}
MIN_ROW_AIR = {"aaa": 40, "fsl": 25}
MIN_PARK_AIR = 150
ROW_SEASONS = 3


def _level(level: str, season: int) -> tuple[pd.DataFrame, dict]:
    years = [y for y in range(season - 5, season + 1) if (data.STATCAST_DIR / f"statcast_{level}_{y}.parquet").exists()]
    no_weather = pd.DataFrame({"game_pk": pd.Series(dtype="int64"), "temp_f": pd.Series(dtype="float64")})
    bips, counts = [], []
    for year in years:
        pitches = data.load_pitches(year, level)
        bips.append(data.batted_balls(pitches, no_weather))
        counts.append(data.hitter_counts(pitches))
    air = expected.add_carry(expected.air_balls(pd.concat(bips, ignore_index=True)))
    log.info(f"{level}: {len(air)} balls in the air over {years[0]}-{years[-1]}, per-ball sd {air.carry.std():.1f} ft")
    h = air.groupby(["batter", "season"]).agg(n_air=("carry", "size"), carry=("carry", "mean"), carry_raw=("carry_raw", "mean"),
                                              exit_velo=("launch_speed", "mean")).reset_index()
    h = h.merge(pd.concat(counts, ignore_index=True)[["batter", "season", "team", "pa", "hr"]], on=["batter", "season"])
    q = h[h.n_air >= QUALIFIED_AIR[level]]
    s = air[air.season == season]
    parks = s.groupby(["venue_id", "venue_name"]).agg(n=("carry_raw", "size"), measured=("carry_raw", "mean"),
                                                      elevation=("elevation_ft", "first")).reset_index()
    parks = parks[parks.n >= MIN_PARK_AIR]
    parks["measured"] -= s.carry_raw.mean()
    summary = {"seasons": years, "split_half": rounded(split_half(s, "carry", QUALIFIED_AIR[level])),
               "year_to_year": rounded(year_to_year(q, "carry")), "qualified": int((q.season == season).sum()),
               "qualified_air": QUALIFIED_AIR[level],
               "parks": [{"venue": p.venue_name, "elevation": None if not np.isfinite(p.elevation) else int(p.elevation),
                          "measured": rounded(p.measured, 1), "n": int(p.n)} for p in parks.sort_values("measured", ascending=False).itertuples()]}
    rows = h[(h.season > season - ROW_SEASONS) & (h.n_air >= MIN_ROW_AIR[level])].assign(level=level)
    return rows, summary


def build(season: int) -> tuple[pd.DataFrame, dict]:
    frames, payload = [], {}
    for level in LEVELS:
        rows, payload[level] = _level(level, season)
        frames.append(rows)
    rows = pd.concat(frames, ignore_index=True)
    names = data.hitter_names(sorted(int(b) for b in rows.batter.unique()))
    rows = rows.assign(player_id=rows.batter.astype(int), player_name=rows.batter.map(names))
    for col in ("carry", "carry_raw", "exit_velo"):
        rows[col] = rows[col].round(2)
    return rows, payload
