"""Runs: the common currency, and the baselines that turn runs into value.

Linear weights come from Statcast's own run-expectancy deltas each season.
Replacement level and the positional adjustment are measured from the
population, with the published values kept beside them for reference.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from backend.data.game_types import regular_season_pitches
from v2.data.pa_dataset import EVENT_TO_OUTCOME, NON_PA_EVENTS, OUTCOMES

log = logging.getLogger(__name__)

STATCAST_DIR = Path(__file__).resolve().parents[2] / "cache"
FULL_SEASON_PA = 600.0
STARTER_INNINGS = 180.0
RELIEVER_INNINGS = 65.0
GAMES_PER_SEASON = 162
# Pythagenpat exponent: runs per game for both teams raised to this power.
PYTHAGENPAT_POWER = 0.287
# The published positional scale in runs per 150 games, kept for reference and as
# the fallback when the multi-position fit rests on too few player-seasons.
PUBLISHED_POSITIONAL_RUNS = {
    "C": 12.5, "SS": 7.5, "2B": 2.5, "CF": 2.5, "3B": 2.5, "RF": -7.5, "LF": -7.5, "1B": -12.5, "DH": -17.5,
}
MIN_MULTI_POSITION_SEASONS = 40
MIN_INNINGS_AT_POSITION = 200.0
INNINGS_PER_150_GAMES = 1350.0
# Outs above average to runs, the ratio Savant's own runs-prevented column uses.
RUNS_PER_OAA = 0.875
# Replacement level in use: the conventional offsets, minus 20 runs per 600 PA for
# hitters (on top of the positional scale), minus 18 per 180 IP for starters, minus 5
# per 65 IP for relievers. The measured fringe is reported beside them; it is biased
# upward because the fringe players who kept playing are the ones who produced.
CONVENTIONAL_REPLACEMENT = {"hitter": -20.0, "SP": -18.0, "RP": -5.0}
CONVENTIONAL_REPLACEMENT_RUNS = CONVENTIONAL_REPLACEMENT["hitter"]
# The pool a team can get without a trade: players outside these ranks by playing time.
REPLACEMENT_RANK = {"hitter": 30, "SP": 150, "RP": 210}
# Reliever leverage multiplier bounds.
LEVERAGE_CLIP = (0.5, 2.0)


def _pa_frame(season: int) -> pd.DataFrame:
    path = STATCAST_DIR / f"statcast_{season}.parquet"
    frame = regular_season_pitches(pd.read_parquet(path, columns=["game_type", "game_date", "events", "delta_run_exp", "home_score", "away_score", "post_home_score", "post_away_score", "game_pk", "inning", "pitcher"]), season)
    pa = frame[frame["events"].notna() & ~frame["events"].isin(NON_PA_EVENTS)].copy()
    pa["outcome"] = pa["events"].map(EVENT_TO_OUTCOME)
    return pa.dropna(subset=["outcome"])


def linear_weights(season: int) -> dict[str, float]:
    """Mean run-expectancy change per outcome, centred so a league-average PA is zero."""
    pa = _pa_frame(season)
    pa["delta_run_exp"] = pd.to_numeric(pa["delta_run_exp"], errors="coerce")
    pa = pa.dropna(subset=["delta_run_exp"])
    means = pa.groupby("outcome")["delta_run_exp"].mean()
    centre = pa["delta_run_exp"].mean()
    return {k: float(means.get(k, 0.0) - centre) for k in OUTCOMES}


def pa_per_inning(season: int) -> float:
    pa = _pa_frame(season)
    outs = pa["outcome"].eq("OUT").sum() + pa["outcome"].eq("K").sum()
    return float(len(pa) / (outs / 3.0))


def batting_runs(rates: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Runs above a league-average hitter per 600 PA, indexed like ``rates``."""
    w = np.array([weights[k] for k in OUTCOMES])
    return pd.Series(rates[list(OUTCOMES)].values @ w * FULL_SEASON_PA, index=rates.index)


def pitching_runs(
    rates: pd.DataFrame, weights: dict[str, float], role: pd.Series, pa_per_ip: float,
    leverage: pd.Series | None = None,
) -> pd.Series:
    """Runs prevented versus a league-average pitcher per 180 IP (SP) or 65 IP (RP).

    Relievers are scaled by their leverage multiplier so that a closer's run counts for more.
    """
    w = np.array([weights[k] for k in OUTCOMES])
    per_pa = rates[list(OUTCOMES)].values @ w
    innings = np.where(role.reindex(rates.index).eq("SP"), STARTER_INNINGS, RELIEVER_INNINGS)
    runs = pd.Series(-per_pa * pa_per_ip * innings, index=rates.index)
    if leverage is not None:
        mult = leverage.reindex(rates.index).fillna(1.0).clip(*LEVERAGE_CLIP)
        runs = runs.where(role.reindex(rates.index).eq("SP"), runs * mult)
    return runs


def runs_per_win(season: int) -> float:
    """Marginal runs per win at the league-average team, from the Pythagenpat derivative."""
    frame = regular_season_pitches(pd.read_parquet(
        STATCAST_DIR / f"statcast_{season}.parquet",
        columns=["game_type", "game_date", "game_pk", "post_home_score", "post_away_score"],
    ), season)
    finals = frame.groupby("game_pk")[["post_home_score", "post_away_score"]].max()
    rpg = float(finals.sum(axis=1).mean())  # both teams, per game
    exponent = rpg ** PYTHAGENPAT_POWER
    rs = ra = rpg / 2.0

    def win_share(scored: float) -> float:
        return scored ** exponent / (scored ** exponent + ra ** exponent)

    eps = 0.01
    # One extra run per game over a season is 162 runs and slope × 162 wins.
    slope = (win_share(rs + eps) - win_share(rs - eps)) / (2 * eps)
    return float(1.0 / slope)


def positional_adjustment(positions: pd.DataFrame, oaa_by_position: pd.DataFrame) -> tuple[dict[str, float], int]:
    """Runs per 150 games by position from players who played two positions in a season.

    ``positions`` has player_id, season, position, innings. ``oaa_by_position`` has
    player_id, season, position, outs_above_average (one row per position played).
    Each pair of positions a player held for at least 200 innings in one season gives
    one equation: his OAA rate at A minus at B equals adjustment(B) minus adjustment(A),
    since a harder position makes the same fielder look worse. Solved by least squares,
    anchored so the innings-weighted mean over positions is zero. Returns the scale and
    the number of player-seasons behind it; the published scale when fewer than 40.
    """
    merged = positions.merge(oaa_by_position, on=["player_id", "season", "position"], how="inner")
    merged = merged[merged["innings"] >= MIN_INNINGS_AT_POSITION].copy()
    merged["rate"] = merged["outs_above_average"] / merged["innings"] * INNINGS_PER_150_GAMES * RUNS_PER_OAA
    fielded = [p for p in PUBLISHED_POSITIONAL_RUNS if p != "DH"]
    index = {p: i for i, p in enumerate(fielded)}
    rows, targets, weights = [], [], []
    seasons = 0
    for (_, _), group in merged.groupby(["player_id", "season"]):
        if len(group) < 2:
            continue
        seasons += 1
        recs = group.to_dict("records")
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                row = np.zeros(len(fielded))
                row[index[b["position"]]] += 1.0
                row[index[a["position"]]] -= 1.0
                rows.append(row)
                targets.append(a["rate"] - b["rate"])
                weights.append(min(a["innings"], b["innings"]))
    if seasons < MIN_MULTI_POSITION_SEASONS:
        log.info(f"positional adjustment: {seasons} multi-position seasons, using the published scale")
        return dict(PUBLISHED_POSITIONAL_RUNS), seasons
    a = np.array(rows) * np.sqrt(np.array(weights))[:, None]
    y = np.array(targets) * np.sqrt(np.array(weights))
    # Anchor: innings-weighted mean of the fielded positions is zero.
    share = positions[positions["position"].isin(fielded)].groupby("position")["innings"].sum().reindex(fielded).fillna(0)
    anchor = (share / share.sum()).values[None, :] * np.sqrt(weights).sum()
    a = np.vstack([a, anchor])
    y = np.append(y, 0.0)
    solution, *_ = np.linalg.lstsq(a, y, rcond=None)
    scale = {p: float(solution[index[p]]) for p in fielded}
    scale["DH"] = PUBLISHED_POSITIONAL_RUNS["DH"]  # a DH never fields; the published gap to 1B stands
    return scale, seasons


def replacement_level(projected: pd.Series, playing_time: pd.Series, group_of: pd.Series) -> dict[str, float]:
    """Playing-time-weighted projected runs of the pool outside the regulars at each group.

    Hitters: outside the top 30 by playing time at the position. SP: outside the top 150. RP: outside the top 210.
    """
    frame = pd.DataFrame({"runs": projected, "pt": playing_time, "group": group_of}).dropna()
    out: dict[str, float] = {}
    for group, g in frame.groupby("group"):
        rank = REPLACEMENT_RANK.get(group, REPLACEMENT_RANK["hitter"])
        pool = g.sort_values("pt", ascending=False).iloc[rank:]
        if pool.empty or pool["pt"].sum() <= 0:
            pool = g
        out[group] = float(np.average(pool["runs"], weights=pool["pt"].clip(lower=1.0)))
    return out
