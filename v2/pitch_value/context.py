"""Context taken out of the pitch values: the park, the hitter, and the season's scoring.

Every adjustment is additive per pitch, in runs, and is measured on the expected-contact values, so what is
left for the park is what exit velocity and launch angle do not already remove (strikeouts, walks and how
hard the ball is hit there).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

N_PARKS = 30
# Hitter and pitcher effects are fitted in turn. The largest change in any hitter's effect is 0.22 runs per 100 pitches
# on the second pass, 0.05 on the third and 0.02 on the fourth, against a spread of 0.56 across hitters.
BACKFIT_PASSES = 3


@dataclass(frozen=True)
class Context:
    parks: pd.DataFrame      # park, seasons, pitches, raw, effect (runs per pitch, shrunk)
    batters: pd.Series       # (season, batter) -> runs per pitch, shrunk
    environment: pd.Series   # season -> runs per pitch relative to all seasons


def park(pitches: pd.DataFrame) -> pd.Series:
    """The park a pitch was thrown in. Home team, except where a team's home moved."""
    team, season = pitches["home_team"], pitches["season"]
    name = team.astype(object).copy()
    name[team.eq("ATH") & season.ge(2025)] = "ATH (Sacramento)"
    name[team.eq("TB") & season.eq(2025)] = "TB (Steinbrenner Field)"
    name[team.eq("TOR") & pitches["game_date"].lt("2021-07-30")] = "TOR (Dunedin, Buffalo)"
    return name


def _shrunk_means(value: np.ndarray, group: np.ndarray) -> tuple[pd.Series, float]:
    """Group means pulled toward zero by k phantom pitches, k = pitch variance over the variance between groups."""
    frame = pd.DataFrame({"g": group, "v": value})
    stats = frame.groupby("g")["v"].agg(["sum", "size", "var"])
    within = float((stats["var"] * (stats["size"] - 1)).sum() / (stats["size"] - 1).sum())
    mean = stats["sum"] / stats["size"]
    weight = stats["size"]
    grand = float((mean * weight).sum() / weight.sum())
    between = float((weight * (mean - grand) ** 2).sum() / weight.sum()) - within * len(stats) / weight.sum()
    if between <= 0:
        raise ValueError("no variance between groups beyond sampling noise")
    k = within / between
    return stats["sum"] / (stats["size"] + k), k


def park_effects(pitches: pd.DataFrame, value: pd.Series) -> pd.DataFrame:
    """Runs per pitch a park adds, from each home team's games there against its road games that season."""
    frame = pd.DataFrame({"season": pitches["season"], "home": pitches["home_team"], "park": park(pitches),
                          "value": value})
    home = frame.groupby(["season", "home", "park"])["value"].agg(["mean", "size"]).reset_index()
    # A team's road games: every game it played away from home, both teams pitching.
    road = value.groupby([pitches["season"], pitches["away_team"]]).mean().rename("road")
    home = home.join(road, on=["season", "home"])
    # The road sample is the other parks, whose effects sum to minus this park's.
    home["raw"] = (home["mean"] - home["road"]) * (N_PARKS - 1) / N_PARKS

    by_park = home.groupby("park")
    seasons = by_park.size()
    raw = by_park.apply(lambda g: np.average(g["raw"], weights=g["size"]), include_groups=False)
    # Season-to-season scatter within a park is the noise in one season's estimate.
    repeat = home[home["park"].map(seasons) > 1]
    noise = float(repeat.groupby("park")["raw"].var().mean())
    between = float(raw.var()) - float((noise / seasons).mean())
    if between <= 0:
        raise ValueError("park effects are indistinguishable from season-to-season noise")
    keep = between / (between + noise / seasons)
    return pd.DataFrame({"seasons": seasons, "pitches": by_park["size"].sum(), "raw": raw, "effect": raw * keep,
                         "kept": keep}).sort_values("effect").reset_index()


def batter_effects(pitches: pd.DataFrame, value: pd.Series) -> pd.Series:
    """Runs per pitch each hitter adds in a season, net of the pitchers he faced.

    Hitters and pitchers are fitted in turn: a hitter's effect is his shrunk mean once the pitchers' effects
    are removed, and the reverse, so a hitter is not marked down for a division full of good arms.
    """
    v = value.to_numpy()
    batter = (pitches["season"].to_numpy() * 10_000_000 + pitches["batter"].to_numpy())
    pitcher = (pitches["season"].to_numpy() * 10_000_000 + pitches["pitcher"].to_numpy())
    pitcher_effect = np.zeros(len(v))
    for _ in range(BACKFIT_PASSES):
        batters, _ = _shrunk_means(v - pitcher_effect, batter)
        batter_effect = batters.reindex(batter).to_numpy()
        pitchers, _ = _shrunk_means(v - batter_effect, pitcher)
        pitcher_effect = pitchers.reindex(pitcher).to_numpy()
    batters.index = pd.MultiIndex.from_arrays([batters.index // 10_000_000, batters.index % 10_000_000],
                                              names=["season", "batter"])
    return batters


def fit(pitches: pd.DataFrame, value: pd.Series) -> Context:
    parks = park_effects(pitches, value)
    without_park = value - park(pitches).map(parks.set_index("park")["effect"]).to_numpy()
    runs = pitches.groupby("season")["runs"].mean()
    return Context(parks=parks, batters=batter_effects(pitches, without_park),
                   environment=runs / pitches["runs"].mean())


def adjust(pitches: pd.DataFrame, value: pd.Series, context: Context) -> pd.Series:
    """Park and hitter removed, each season centred on its league average and put on a common run scale."""
    park_effect = park(pitches).map(context.parks.set_index("park")["effect"])
    batter_effect = context.batters.reindex(pd.MultiIndex.from_frame(pitches[["season", "batter"]])).to_numpy()
    if park_effect.isna().any() or np.isnan(batter_effect).any():
        raise ValueError("a pitch has no fitted park or hitter effect")
    out = value - park_effect.to_numpy() - batter_effect
    out = out - out.groupby(pitches["season"]).transform("mean")
    return (out / pitches["season"].map(context.environment)).rename("value")
