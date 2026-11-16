"""Does a player's gap on contact repeat, and how much of it should he keep?

Usage:
    python -m v2.xwar.validate [--confirm]

Reads the season files of batting and pitch run value. For hitters the gap is what their tracked balls in play
did against the contact model (``gap_runs``); for pitchers it is home runs against expected home runs
(``home_run_gap``). On the development pairs (first seasons 2021 to 2023), for each side:

- the gate, fixed before any result was read: the gap per ball repeats a season later with a 95% interval above
  zero, at 200+ tracked balls in both seasons. A side that fails keeps none of its gap;
- the shrink: k, the phantom balls that pull a gap rate toward zero, and the weight of each earlier season
  relative to the one after it, fitted together as the pair that best predicts next season's rate from the
  seasons to date. It is set beside the latest season alone and every season weighted alike.

``--confirm`` then reads the pairs the choices never saw (first seasons 2024 and 2025), once, with the pinned
shrink: each side's repeat, and how well a hitter's season says what his next season's hitting fWAR per 600 PA
will be when it is read as fWAR, as fWAR with the whole gap removed, and as xWAR.

Writes cache/xwar/validation.json. The results are pinned by hand in ``gap.py``; this does not run in CI.
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from backend.log import setup_logging
from v2.batting_value import run as batting_run
from v2.pitch_value import run as pitch_run
from v2.pitch_value.pitches import STATCAST_DIR
from v2.players import production
from v2.players import runs as player_runs
from v2.players.runs import FULL_SEASON_PA
from v2.xwar import build, gap

log = logging.getLogger(__name__)

OUT_DIR = STATCAST_DIR / "xwar"
# First seasons of the pairs every choice is made on; the later pairs are read once, for the record.
DEVELOPMENT = (2021, 2022, 2023)
CONFIRMATION = (2024, 2025)
# Plate appearances required in both seasons of a pair for the next-season fWAR read.
MIN_PA = 300
# Tracked balls required in both seasons of a pair for the gate.
MIN_BALLS = 200
# k is fitted on thinner seasons too: the shrink has to hold where it matters most.
FIT_MIN_BALLS = 50
# The search range for k, in balls: from almost no shrink to almost no credit.
K_BOUNDS = (1.0, 1e5)
BOOTSTRAP_DRAWS = 2000
SEED = 20261116
# Season file, player column, gap column, tracked-balls column.
SIDES = {
    "hitter": (batting_run.OUT_DIR / "hitter_seasons.parquet", "batter", "gap_runs", "tracked_balls"),
    "pitcher": (pitch_run.OUT_DIR / "pitcher_seasons.parquet", "pitcher", "home_run_gap", "in_play"),
}


def load(side: str) -> pd.DataFrame:
    """One row per player-season: season, player, gap, balls and rate (gap per ball)."""
    path, player, gap_column, balls = SIDES[side]
    frame = pd.read_parquet(path, columns=["season", player, gap_column, balls])
    frame = frame.rename(columns={player: "player", gap_column: "gap", balls: "balls"})
    return frame.assign(rate=frame["gap"] / frame["balls"].where(frame["balls"] > 0))


def pairs(seasons: pd.DataFrame, player: str) -> pd.DataFrame:
    """A player's consecutive seasons side by side; the later one's columns end in ``_next``."""
    nxt = seasons.assign(season=seasons["season"] - 1)
    return seasons.merge(nxt, on=["season", player], suffixes=("", "_next"))


def _draws(frame: pd.DataFrame, player: str, rng: np.random.Generator, statistic) -> np.ndarray:
    """``statistic`` on ``BOOTSTRAP_DRAWS`` resamples of the players, each with all of his pairs."""
    players = frame[player].unique()
    rows = {p: np.flatnonzero(frame[player].to_numpy() == p) for p in players}
    return np.array([statistic(frame.iloc[np.concatenate([rows[p] for p in rng.choice(players, len(players))])])
                     for _ in range(BOOTSTRAP_DRAWS)])


def _r(frame: pd.DataFrame, a: str, b: str) -> float:
    return float(np.corrcoef(frame[a], frame[b])[0, 1])


def _interval(draws: np.ndarray) -> list[float]:
    return [float(v) for v in np.percentile(draws, [2.5, 97.5])]


def repeat(frame: pd.DataFrame, player: str, a: str, b: str, rng: np.random.Generator) -> dict:
    """Correlation of ``a`` with ``b`` over the pairs, with an interval that resamples players."""
    return {"r": _r(frame, a, b), "interval": _interval(_draws(frame, player, rng, lambda f: _r(f, a, b))),
            "pairs": len(frame), "players": int(frame[player].nunique())}


def earned_rate(seasons: pd.DataFrame, shrink: gap.Shrink) -> pd.Series:
    """Each player-season's shrunk gap per ball, from his seasons to date."""
    return gap.earned(seasons, "player", "gap", "balls", shrink) / seasons["balls"].where(seasons["balls"] > 0)


def _loss(seasons: pd.DataFrame, shrink: gap.Shrink) -> float:
    """Squared miss of the shrunk rate on the next season's rate over the development pairs, each pair weighted
    by the balls behind its target."""
    paired = pairs(seasons.assign(predicted=earned_rate(seasons, shrink)), "player")
    fit = paired[paired["season"].isin(DEVELOPMENT) & (paired["balls"] >= FIT_MIN_BALLS) & (paired["balls_next"] >= FIT_MIN_BALLS)]
    return float((fit["balls_next"] * (fit["rate_next"] - fit["predicted"]) ** 2).sum())


def _best_k(seasons: pd.DataFrame, decay: float) -> tuple[float, float]:
    found = minimize_scalar(lambda k: _loss(seasons, gap.Shrink(k, decay)), bounds=K_BOUNDS, method="bounded")
    return float(found.x), float(found.fun)


def fit_shrink(seasons: pd.DataFrame) -> gap.Shrink:
    """The phantom balls and the season weight that together best predict the next season's rate."""
    decay = float(minimize_scalar(lambda d: _best_k(seasons, d)[1], bounds=(0.0, 1.0), method="bounded").x)
    return gap.Shrink(k=_best_k(seasons, decay)[0], decay=decay)


def development(side: str) -> dict:
    rng = np.random.default_rng(SEED)
    seasons = load(side)
    # Nothing after the development pairs' second seasons is read here.
    seasons = seasons[seasons["season"] <= max(DEVELOPMENT) + 1]
    paired = pairs(seasons, "player")
    gated = paired[paired["season"].isin(DEVELOPMENT) & (paired["balls"] >= MIN_BALLS) & (paired["balls_next"] >= MIN_BALLS)]
    gate = repeat(gated, "player", "rate", "rate_next", rng)
    gate["passed"] = gate["interval"][0] > 0

    shrink = fit_shrink(seasons)
    # The two simpler readings of his record, each with its own best k.
    rivals = {"latest_only": 0.0, "all_alike": 1.0}
    best = {name: _best_k(seasons, decay) for name, decay in rivals.items()}
    rates = {"fitted": earned_rate(seasons, shrink), **{name: earned_rate(seasons, gap.Shrink(best[name][0], decay)) for name, decay in rivals.items()}}
    gated = gated.merge(seasons[["season", "player"]].assign(**rates), on=["season", "player"])
    lead = lambda name: (lambda f: _r(f, "fitted", "rate_next") - _r(f, name, "rate_next"))  # noqa: E731
    return {"gate": gate, "min_balls": MIN_BALLS, "fit_min_balls": FIT_MIN_BALLS,
            "shrink": {"k": shrink.k, "decay": shrink.decay, "loss": _loss(seasons, shrink), "r_next": _r(gated, "fitted", "rate_next")},
            "rivals": {name: {"k": best[name][0], "loss": best[name][1], "r_next": _r(gated, name, "rate_next"),
                              "fitted_lead": lead(name)(gated), "interval": _interval(_draws(gated, "player", rng, lead(name)))}
                       for name in rivals}}


def hitter_war(shrink: gap.Shrink | None) -> pd.DataFrame:
    """Per hitter-season, per 600 PA: hitting fWAR, fWAR with the whole contact gap removed (``expected``) and
    xWAR under ``shrink``."""
    side = build.hitter_side(pd.read_parquet(SIDES["hitter"][0]), shrink)
    war = production.load_war(int(side["season"].max()))
    hitting = war[war["role"].eq("bat")].groupby(build.KEYS)["war"].sum()
    side = side.join(hitting, on=build.KEYS).dropna(subset=["war"])
    per_win = side["season"].map({s: player_runs.runs_per_win(int(s)) for s in side["season"].unique()})
    per_600 = FULL_SEASON_PA / side["pa"].where(side["pa"] > 0)
    return pd.DataFrame({
        "season": side["season"], "player": side["player_id"], "pa": side["pa"], "fwar": side["war"] * per_600,
        "expected": (side["war"] - side["realized_gap_runs"] / per_win) * per_600,
        "xwar": (side["war"] - (side["realized_gap_runs"] - side["earned_gap_runs"]) / per_win) * per_600,
    })


def confirmation() -> dict:
    """The single read of the pairs no choice was made on."""
    rng = np.random.default_rng(SEED + 1)
    out = {}
    for side in SIDES:
        paired = pairs(load(side), "player")
        held = paired[paired["season"].isin(CONFIRMATION) & (paired["balls"] >= MIN_BALLS) & (paired["balls_next"] >= MIN_BALLS)]
        out[side] = {"gate": repeat(held, "player", "rate", "rate_next", rng), "min_balls": MIN_BALLS}
    paired = pairs(hitter_war(gap.HITTER), "player")
    held = paired[paired["season"].isin(CONFIRMATION) & (paired["pa"] >= MIN_PA) & (paired["pa_next"] >= MIN_PA)]
    measures = ("fwar", "expected", "xwar")
    lead = lambda a, b: (lambda f: _r(f, a, "fwar_next") - _r(f, b, "fwar_next"))  # noqa: E731
    out["next_season_fwar"] = {
        "min_pa": MIN_PA, "pairs": len(held), "players": int(held["player"].nunique()),
        "r": {name: repeat(held, "player", name, "fwar_next", rng) for name in measures},
        "lead": {f"{a}_over_{b}": {"r": lead(a, b)(held), "interval": _interval(_draws(held, "player", rng, lead(a, b)))}
                 for a, b in (("expected", "fwar"), ("xwar", "fwar"), ("xwar", "expected"))},
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true", help="also read the 2024 and 2025 pairs (once, with the pinned shrink)")
    args = parser.parse_args()
    setup_logging()
    path = OUT_DIR / "validation.json"
    out = {"development": {side: development(side) for side in SIDES}}
    if args.confirm:
        out["confirmation"] = confirmation()
        for side in SIDES:
            gate = out["confirmation"][side]["gate"]
            log.info(f"confirmation, {side}: {gate['pairs']} pairs, repeat {gate['r']:+.3f} [{gate['interval'][0]:+.3f}, {gate['interval'][1]:+.3f}]")
        block = out["confirmation"]["next_season_fwar"]
        log.info(f"confirmation, next-season hitting fWAR per 600 PA, {block['pairs']} pairs at {MIN_PA}+ PA: "
                 + ", ".join(f"{name} {r['r']:+.3f}" for name, r in block["r"].items()) + "; "
                 + "; ".join(f"{name} {r['r']:+.3f} [{r['interval'][0]:+.3f}, {r['interval'][1]:+.3f}]" for name, r in block["lead"].items()))
    elif path.exists() and "confirmation" in (kept := json.loads(path.read_text())):
        out["confirmation"] = kept["confirmation"]
    for side, block in out["development"].items():
        gate, shrink, rivals = block["gate"], block["shrink"], block["rivals"]
        log.info(f"{side}: {gate['pairs']} development pairs at {MIN_BALLS}+ balls, repeat {gate['r']:+.3f} "
                 f"[{gate['interval'][0]:+.3f}, {gate['interval'][1]:+.3f}], gate {'passed' if gate['passed'] else 'failed'}; "
                 f"k {shrink['k']:.1f}, season weight {shrink['decay']:.3f}, r with next season {shrink['r_next']:+.3f}; "
                 + "; ".join(f"lead over {name} {r['fitted_lead']:+.3f} [{r['interval'][0]:+.3f}, {r['interval'][1]:+.3f}]" for name, r in rivals.items()))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    log.info(f"wrote {path}")


if __name__ == "__main__":
    main()
