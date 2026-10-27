"""Build the carry study for one season and publish it.

    MLBMODEL_DB_WRITES=1 /opt/homebrew/bin/python3.13 -m v2.carry.build --season 2026
    /opt/homebrew/bin/python3.13 -m v2.carry.build --season 2026 --no-write   # writes cache/carry/study_{season}.json only

Carry uses 2021 onward; swing physics needs bat tracking (2024 onward); the changer study starts with 2023's partial
tracking. Triple-A and Florida State League carry are built separately (v2.carry.minors). Runs locally once per offseason (about ten minutes, most of it the lab grids).
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from backend.log import setup_logging
from v2.carry import changers, data, expected, lab, minors, physics, study, swing

log = logging.getLogger(__name__)

MODEL_VERSION = "carry-v1"
FIRST_CARRY_SEASON = 2021
FIRST_CHANGER_SEASON = 2023
FIRST_PHYSICS_SEASON = 2024
MIN_HITTER_AIR = 50
SWING_COLUMNS = ["batter", "season", "plate_z", "swing_path_tilt", "attack_angle", "bat_speed"]


def _hitter_table(air: pd.DataFrame, spun: pd.DataFrame, profiles: pd.DataFrame, counts: pd.DataFrame, slope: float,
                  seasons: list[int]) -> pd.DataFrame:
    a = air[air.season.isin(seasons)].groupby(["batter", "season"]).agg(
        n_air=("carry", "size"), carry=("carry", "mean"), carry_raw=("carry_raw", "mean"))
    s = spun.groupby(["batter", "season"]).agg(n_spin=("spin_effect", "size"), spin_effect=("spin_effect", "mean"),
                                               backspin=("backspin", "mean"), sidespin=("sidespin", "mean"))
    league = spun.groupby("season").spin_effect.mean()
    h = a.join(s).reset_index().merge(profiles, on=["batter", "season"], how="left").merge(counts, on=["batter", "season"])
    h["physics"] = slope * (h.spin_effect - h.season.map(league))
    return h[h.n_air >= MIN_HITTER_AIR].reset_index(drop=True)


def build(season: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    temps = data.weather()
    bips, counts, swings = [], [], []
    for year in range(FIRST_CARRY_SEASON, season + 1):
        pitches = data.load_pitches(year)
        bips.append(data.batted_balls(pitches, temps))
        counts.append(data.hitter_counts(pitches))
        if year >= FIRST_CHANGER_SEASON:
            swings.append(pitches.loc[pitches.description.isin(data.SWINGS), SWING_COLUMNS])
        log.info(f"{year}: {len(bips[-1])} balls in play")
    bip = pd.concat(bips, ignore_index=True)
    counts = pd.concat(counts, ignore_index=True)
    curves = swing.pitch_height_curves(bip[bip.season >= FIRST_PHYSICS_SEASON])
    profiles = changers.swing_profiles(pd.concat(swings, ignore_index=True), curves)

    air = expected.add_carry(expected.air_balls(bip))
    air = air.join(swing.at_belt_high(air, curves))
    log.info(f"carry: {len(air)} balls in the air, per-ball sd {air.carry.std():.1f} ft")

    spun, typicals, uncorrected = [], {}, {}
    for year in range(FIRST_PHYSICS_SEASON, season + 1):
        balls = swing.spin_balls(air[air.season == year])
        out, typicals[year] = swing.solve(balls, balls.tilt_rel.to_numpy(), detail=True)
        spun.append(balls.join(out))
        if year >= season - 1:
            raw, _ = swing.solve(balls, balls.tilt_rel.to_numpy(), tilt_correction=0.0)
            uncorrected[year] = balls.assign(spin_effect=raw.spin_effect)
        log.info(f"{year}: collision solved for {len(balls)} balls, median efficiency {out.q.median():.3f}")
    spun = pd.concat(spun, ignore_index=True)

    def share(frame: pd.DataFrame) -> float:
        h = frame.groupby(["batter", "season"]).agg(n=("carry", "size"), carry=("carry", "mean"), e=("spin_effect", "mean")).reset_index()
        now, before = h[(h.season == season) & (h.n >= 60)], h[(h.season == season - 1) & (h.n >= 60)]
        c = np.polyfit(now.e, now.carry, 1)
        return 1 - ((before.carry - np.polyval(c, before.e)) ** 2).sum() / ((before.carry - before.carry.mean()) ** 2).sum()

    hitters = _hitter_table(air, spun, profiles, counts, 1.0, list(range(FIRST_PHYSICS_SEASON, season + 1)))
    explained = study.explained(hitters, air, season, share(pd.concat(uncorrected.values())))
    slope = explained["coef"][0]
    hitters = _hitter_table(air, spun, profiles, counts, slope, list(range(FIRST_PHYSICS_SEASON, season + 1)))

    current = spun[spun.season == season]
    grids, generic = lab.grids(current, current, typicals[season], slope)
    hitters["lab_carry"] = [grids.get(int(p)) if s == season else None for p, s in zip(hitters.batter, hitters.season)]

    carry_by_season = air.groupby(["batter", "season"]).carry.mean().reset_index()
    p = changers.pairs(profiles, counts, carry_by_season)
    coors = air[(air.season == season) & (air.venue_name == "Coors Field")].air_density.mean()
    parks = study.air_and_parks(spun, season)
    payload = {
        "season": season, "model_version": MODEL_VERSION,
        "puzzle": study.examples(air, season),
        "repeatability": study.repeatability(air, hitters, season),
        "air": parks,
        "spin_curve": study.spin_curves(parks["league_density"], float(coors)),
        "sidespin": study.sidespin(current),
        "collision": {"efficiency": round(float(current.q.median()), 3), "balls": int(len(current))},
        "explained": explained,
        "swing_types": study.swing_types(profiles, counts, bip, hitters),
        "pitch_height": {k: [round(c, 4) for c in v] for k, v in curves.items()},
        "changers": {"coefficients": changers.coefficients(p), "groups": changers.groups(p)},
        "lab": {"steps": lab.STEPS, "generic": generic, "min_balls": lab.MIN_BALLS},
        "physics": {"cd_high_speed": physics.CD_HIGH_SPEED, "cd_low_speed_rise": physics.CD_LOW_SPEED_RISE,
                    "cd_transition_mph": round(physics.CD_TRANSITION / physics.MPH), "spin_scale": physics.SPIN_SCALE,
                    "bat_axis_tilt_share": physics.BAT_AXIS_TILT_SHARE, "tilt_attack_correction": physics.TILT_ATTACK_CORRECTION},
    }
    minor_rows, payload["minors"] = minors.build(season)
    names = data.hitter_names(sorted(int(b) for b in hitters.batter.unique()))
    out = hitters.assign(player_id=hitters.batter.astype(int), player_name=hitters.batter.map(names))
    for col in ("backspin", "sidespin"):
        out[col] = out[col].round().astype("Int64")
    for col in ("carry", "carry_raw", "physics", "tilt", "attack", "bat_speed"):
        out[col] = out[col].round(2)
    return payload, out, minor_rows


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    payload, hitters, minor_rows = build(args.season)
    path = data.CACHE_DIR / f"study_{args.season}.json"
    path.write_text(json.dumps(payload))
    hitters.to_parquet(data.CACHE_DIR / f"hitters_{args.season}.parquet")
    minor_rows.to_parquet(data.CACHE_DIR / f"minors_{args.season}.parquet")
    log.info(f"payload {len(json.dumps(payload)) / 1024:.0f} kB, {len(hitters)} hitter rows, {len(minor_rows)} minor-league rows -> {path}")
    if not args.no_write:
        from v2.carry.publish import publish
        publish(args.season, MODEL_VERSION, payload, hitters, minor_rows)


if __name__ == "__main__":
    main()
