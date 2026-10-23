"""Matured MLB outcomes for each prospect snapshot, from the player model's cached fWAR and season lines.

A snapshot is a player at the end of season S. Its window is the seven MLB seasons S+1 to S+7:

- ``mlb``: at least one window season with 100 plate appearances or 40 innings
- ``eta``: the first such season, or missing
- ``regular``: at least one window season of ``REGULAR_WAR`` or more
- ``peak_war``: the best mean of three consecutive window seasons' WAR (missing seasons count zero),
  zero for a player who never reaches the majors in the window
- ``control_war``: WAR over the six seasons from ``eta``; missing when that span runs past the last
  observed season (censored), zero when the player never reaches the majors in a complete window

Per-season WAR is the sum of batting and pitching fWAR for the player. ``CONTROL_SEASONS`` is the
one definition of the control window; the trade value prices the same six seasons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from v2.players import bio
from v2.players.production import load_war

WINDOW_SEASONS = 7
CONTROL_SEASONS = 6
PEAK_SPAN = 3
REGULAR_WAR = 2.0
MLB_PA = 100
MLB_IP = 40.0
FIRST_SNAPSHOT = 2005


def mlb_seasons(last_season: int) -> pd.DataFrame:
    """One row per player and MLB season since FIRST_SNAPSHOT: total WAR and whether it qualifies as an MLB season."""
    war = load_war(last_season)
    war = war[war["season"] >= FIRST_SNAPSHOT].groupby(["player_id", "season"])["war"].sum().rename("war").reset_index()
    seasons = list(range(FIRST_SNAPSHOT, last_season + 1))
    hit = bio.league_season_lines(seasons, "hitting")[["player_id", "season", "plateAppearances"]]
    pit = bio.league_season_lines(seasons, "pitching")[["player_id", "season", "outs"]]
    lines = hit.merge(pit, on=["player_id", "season"], how="outer").fillna(0.0)
    lines["qualifies"] = (lines["plateAppearances"] >= MLB_PA) | (lines["outs"] / 3.0 >= MLB_IP)
    out = war.merge(lines[["player_id", "season", "qualifies"]], on=["player_id", "season"], how="outer")
    out["war"] = out["war"].fillna(0.0)
    out["qualifies"] = out["qualifies"].fillna(False).astype(bool)
    out["player_id"] = out["player_id"].astype("int64")
    out["season"] = out["season"].astype("int64")
    return out


def _peak(wars: np.ndarray) -> float:
    if len(wars) < PEAK_SPAN:
        wars = np.concatenate([wars, np.zeros(PEAK_SPAN - len(wars))])
    windows = np.lib.stride_tricks.sliding_window_view(wars, PEAK_SPAN).mean(axis=1)
    return float(max(windows.max(), 0.0))


def outcomes(snapshots: pd.DataFrame, last_season: int) -> pd.DataFrame:
    """Targets for (player_id, season) snapshot rows; windows that run past ``last_season`` are censored."""
    mlb = mlb_seasons(last_season)
    by_player = {pid: g.set_index("season").sort_index() for pid, g in mlb.groupby("player_id")}
    rows = []
    for pid, s in snapshots[["player_id", "season"]].drop_duplicates().itertuples(index=False):
        lo, hi = s + 1, s + WINDOW_SEASONS
        complete = hi <= last_season
        g = by_player.get(pid)
        w = g.loc[lo:min(hi, last_season)] if g is not None else None
        if w is None or w.empty:
            rows.append({"player_id": pid, "season": s, "mlb": False, "eta": np.nan, "regular": False,
                         "peak_war": 0.0 if complete else np.nan, "control_war": 0.0 if complete else np.nan,
                         "window_complete": complete})
            continue
        qual = w.index[w["qualifies"]]
        eta = int(qual[0]) if len(qual) else np.nan
        wars = w["war"].reindex(range(lo, min(hi, last_season) + 1), fill_value=0.0).to_numpy()
        reached = bool(len(qual))
        regular = bool((w["war"] >= REGULAR_WAR).any())
        if reached:
            span = range(eta, eta + CONTROL_SEASONS)
            control = float(g["war"].reindex(span, fill_value=0.0).sum()) if eta + CONTROL_SEASONS - 1 <= last_season else np.nan
        else:
            control = 0.0 if complete else np.nan
        peak = _peak(wars) if reached else 0.0
        if not complete and not reached:
            peak = np.nan
        rows.append({"player_id": pid, "season": s, "mlb": reached, "eta": eta, "regular": regular,
                     "peak_war": peak, "control_war": control, "window_complete": complete})
    return pd.DataFrame(rows)
