"""Aggregates for each section of the carry page, all computed from the per-ball frames the build prepares."""
from __future__ import annotations

import numpy as np
import pandas as pd

from v2.carry import physics as ph
from v2.carry.changers import MIN_PA, MIN_SWINGS, rates

QUALIFIED_AIR = 80               # balls in the air for a hitter-season to rank or enter reliability
REFERENCE = (103.0, 28.0)        # exit mph, launch degrees: a typical home-run-range fly ball for the spin curve
PUZZLE_EXIT = 100.0
PUZZLE_LAUNCH = (20.0, 35.0)     # fly balls, where carry decides outs and home runs
ATTACK_TYPES = [(-np.inf, 5, "Level"), (5, 9, "Slight lift"), (9, 13, "Lift"), (13, np.inf, "Uppercut")]
TILT_TYPES = [(-np.inf, 29, "Flat"), (29, 35, "Average"), (35, np.inf, "Steep")]


def rounded(x: float, nd: int = 2) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def split_half(balls: pd.DataFrame, col: str, min_n: int) -> float:
    b = balls.sort_values(["game_date", "game_pk", "at_bat_number"]).copy()
    b["half"] = b.groupby("batter").cumcount() % 2
    m = b.groupby(["batter", "half"])[col].agg(["mean", "size"]).unstack()
    m = m[(m["size"] >= min_n / 2).all(axis=1)]
    r = m["mean"][0].corr(m["mean"][1])
    return 2 * r / (1 + r)


def year_to_year(table: pd.DataFrame, col: str) -> float:
    nxt = table.assign(season=table.season - 1)
    p = table.merge(nxt, on=["batter", "season"], suffixes=("_0", "_1"))
    return p[f"{col}_0"].corr(p[f"{col}_1"])


def examples(air: pd.DataFrame, season: int) -> dict:
    """The puzzle: the two best- and two worst-carrying qualified hitters, each with his most telling hard-hit fly ball, and
    every ball for the best and worst hitter (expected vs actual distance against exit velocity and launch angle only)."""
    a = air[air.season == season]
    h = a.groupby("batter").agg(n=("carry_raw", "size"), carry=("carry_raw", "mean"))
    h = h[h.n >= QUALIFIED_AIR].sort_values("carry")
    picks = list(h.index[-2:][::-1]) + list(h.index[:2])
    balls = []
    for pid in picks:
        mine = a[(a.batter == pid) & (a.launch_speed >= PUZZLE_EXIT) & a.launch_angle.between(*PUZZLE_LAUNCH)]
        ball = mine.loc[mine.carry_raw.idxmax() if h.carry[pid] > 0 else mine.carry_raw.idxmin()]
        balls.append({"player_id": int(pid), "date": str(pd.Timestamp(ball.game_date).date()), "exit": rounded(ball.launch_speed, 1),
                      "launch": int(round(ball.launch_angle)), "distance": int(ball.hit_distance_sc),
                      "expected": int(round(ball.hit_distance_sc - ball.carry_raw)), "spray": rounded(ball.spray, 0)})
    every = {}
    for pid in (picks[0], picks[2]):
        mine = a[a.batter == pid]
        every[str(int(pid))] = [[int(round(d - c)), int(d)] for d, c in zip(mine.hit_distance_sc, mine.carry_raw)]
    return {"balls": balls, "every_ball": every}


def repeatability(air: pd.DataFrame, hitters: pd.DataFrame, season: int) -> dict:
    q = hitters[hitters.n_air >= QUALIFIED_AIR]
    return {"split_half": rounded(split_half(air[air.season == season], "carry", QUALIFIED_AIR)),
            "year_to_year": rounded(year_to_year(q, "carry")), "hitters": int((q.season == season).sum())}


def air_and_parks(spun: pd.DataFrame, season: int) -> dict:
    """Distance gained per unit of air density, measured and by the flight model, on the same balls (the swing-physics
    balls, each flown with its own launch and estimated spin), density bins, and each ballpark's measured carry next to
    what its air alone does to its own balls."""
    league = float(spun.air_density.mean())
    X = np.column_stack([np.ones(len(spun)), spun.air_density])
    measured = np.linalg.lstsq(X, spun.carry_raw, rcond=None)[0][1]
    v0 = ph.launch_velocity(spun.launch_speed.to_numpy(), spun.launch_angle.to_numpy(), spun.launch_dir.to_numpy())
    spin = spun[["spin_x", "spin_y", "spin_z"]].to_numpy()
    at_own = spun.pred_distance.to_numpy()
    at_league = ph.fly(v0, spin, np.full(len(spun), league))[0]
    thinner = ph.fly(v0, spin, spun.air_density.to_numpy() + 0.05)[0]
    model = float(np.nanmean((thinner - at_own) / 0.05))
    bins = pd.cut(spun.air_density, np.arange(0.90, 1.32, 0.04))
    by_density = spun.groupby(bins, observed=True).carry_raw.agg(["mean", "size"])
    by_density = by_density[by_density["size"] >= 300]
    s = spun.assign(air=at_own - at_league)[spun.season == season]
    parks = s.groupby(["venue_id", "venue_name"]).agg(n=("carry_raw", "size"), measured=("carry_raw", "mean"),
                                                      elevation=("elevation_ft", "first"), air=("air", "mean")).reset_index()
    parks = parks[parks.n >= 150]
    parks["measured"] -= s.carry_raw.mean()
    parks["air"] -= s.air.mean()
    return {"ft_per_density_measured": rounded(measured, 1), "ft_per_density_model": rounded(model, 1), "league_density": rounded(league, 3),
            "by_density": [[rounded(i.mid, 2), rounded(r["mean"], 1), int(r["size"])] for i, r in by_density.iterrows()],
            "parks": [{"venue": v.venue_name, "elevation": int(v.elevation), "measured": rounded(v.measured, 1), "air": rounded(v.air, 1),
                       "n": int(v.n)} for v in parks.sort_values("measured", ascending=False).itertuples()]}


def spin_curves(league_density: float, coors_density: float) -> dict:
    """Distance of the reference fly ball against backspin, at league air and at Coors Field air."""
    rpm = np.arange(0, 4001, 250, dtype=float)
    v0 = ph.launch_velocity(np.full(len(rpm), REFERENCE[0]), np.full(len(rpm), REFERENCE[1]), np.zeros(len(rpm)))
    spin = (rpm * 2 * np.pi / 60)[:, None] * ph.backspin_axis(v0)
    curves = {name: [int(round(d)) for d in ph.fly(v0, spin, np.full(len(rpm), dens))[0]]
              for name, dens in (("league", league_density), ("coors", coors_density))}
    return {"exit": REFERENCE[0], "launch": REFERENCE[1], "rpm": rpm.astype(int).tolist(), **curves}


def sidespin(spun: pd.DataFrame) -> list[dict]:
    """Hook and slice by direction: pulled balls curve toward the pull line, opposite-field balls toward the other line."""
    b = spun.assign(hook_pull=np.where(spun.stand == "R", -1, 1) * spun.hook)
    g = b.groupby(pd.cut(b.pull, [-90, -20, -5, 5, 20, 90], labels=["far oppo", "oppo", "center", "pull", "far pull"]), observed=True)
    return [{"direction": str(k), "curve": rounded(v.hook_pull.mean(), 1), "feet": rounded(v.sidespin_effect.mean(), 1), "n": int(len(v))}
            for k, v in g]


def explained(hitters: pd.DataFrame, air: pd.DataFrame, season: int, share_without_correction: float) -> dict:
    s = hitters[(hitters.season == season) & (hitters.n_spin >= 60)]
    prior = hitters[(hitters.season == season - 1) & (hitters.n_spin >= 60)]
    coef = np.polyfit(s.spin_effect, s.carry, 1)
    pred = np.polyval(coef, prior.spin_effect)
    share = 1 - ((prior.carry - pred) ** 2).sum() / ((prior.carry - prior.carry.mean()) ** 2).sum()
    leftover = hitters.assign(left=hitters.carry - np.polyval(coef, hitters.spin_effect))
    return {"coef": [rounded(coef[0], 4), rounded(coef[1], 3)], "share": rounded(share), "share_without_steep_correction": rounded(share_without_correction),
            "ceiling": rounded(split_half(air[air.season == season - 1], "carry", 60)),
            "leftover_year_to_year": rounded(year_to_year(leftover[leftover.n_spin >= 50], "left")),
            "backspin_carry_corr": rounded(s.backspin.corr(s.carry))}


def swing_types(profiles: pd.DataFrame, counts: pd.DataFrame, bip: pd.DataFrame, hitters: pd.DataFrame) -> dict:
    """Outcome profiles of hitters grouped by belt-high attack angle and by belt-high tilt."""
    bb = bip.assign(gb=bip.launch_angle < 10, pu=bip.launch_angle > 50).groupby(["batter", "season"])[["gb", "pu"]].mean()
    t = (profiles.merge(rates(counts), on=["batter", "season"]).merge(bb.reset_index(), on=["batter", "season"])
         .merge(hitters[["batter", "season", "carry"]], on=["batter", "season"], how="left"))
    t = t[(t.tracked >= MIN_SWINGS) & (t.pa >= MIN_PA)]
    cols = ["carry", "gb", "pu", "ba", "obp", "slg", "ops", "hr600", "so600", "whiff"]

    def table(col: str, types: list) -> list[dict]:
        out = []
        for lo, hi, name in types:
            g = t[(t[col] >= lo) & (t[col] < hi)]
            out.append({"type": name, "lo": None if not np.isfinite(lo) else lo, "hi": None if not np.isfinite(hi) else hi,
                        "n": int(len(g)), **{c: rounded(g[c].mean(), 4) for c in cols}})
        return out
    return {"attack": table("attack", ATTACK_TYPES), "tilt": table("tilt", TILT_TYPES),
            "seasons": sorted(int(s) for s in t.season.unique())}
