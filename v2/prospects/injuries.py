"""Injured-list history from Stats API transactions: how often and how long a prospect has been hurt.

One request per level and season returns every transaction; the injured-list moves are parsed from
their descriptions ("placed RHP X on the 60-day injured list", "activated ... from the ...", "transferred
... to the 60-day injured list"). Per player and season: placements, an estimate of the days spent on the
list (placement to the next activation, or to season end), and whether any stint was long (60-day,
full-season, or transferred to one). Cached per season under ``cache/prospects/injuries_{season}.parquet``.
"""
from __future__ import annotations

import logging
import re
from datetime import date

import pandas as pd

from v2.prospects.feeds import CACHE_DIR, STATSAPI, fetch_json

log = logging.getLogger(__name__)

SPORTS = (1, 11, 12, 13, 14, 15, 16)
LONG_LISTS = ("60-day", "full-season")
SEASON_END = (9, 30)
PLACED = re.compile(r"placed .* on the ([\w-]+) injured list", re.I)
TRANSFERRED = re.compile(r"transferred .* to the ([\w-]+) injured list", re.I)
ACTIVATED = re.compile(r"(activated|reinstated) .* from the [\w-]+ injured list", re.I)
FEATURES = ["il_stints", "il_days", "il_long"]


def path(season: int):
    return CACHE_DIR / f"injuries_{season}.parquet"


def fetch(season: int) -> pd.DataFrame:
    """Injured-list moves of one season across MLB and the affiliated levels: player_id, date, move, list."""
    rows = []
    for sport in SPORTS:
        try:
            payload = fetch_json(f"{STATSAPI}/transactions", {"sportId": sport, "startDate": f"{season}-01-01", "endDate": f"{season}-12-31"},
                                 timeout=120)
        except FileNotFoundError:  # a level that did not exist that season
            continue
        for t in payload.get("transactions", []):
            desc, pid, day = t.get("description", ""), t.get("person", {}).get("id"), t.get("effectiveDate") or t.get("date")
            if not pid or not day or "injured list" not in desc.lower():
                continue
            for move, rx in (("placed", PLACED), ("transferred", TRANSFERRED), ("activated", ACTIVATED)):
                m = rx.search(desc)
                if m:
                    rows.append({"player_id": int(pid), "date": pd.Timestamp(day[:10]), "move": move,
                                 "list": m.group(1).lower() if move != "activated" else None})
                    break
    out = pd.DataFrame(rows, columns=["player_id", "date", "move", "list"]).drop_duplicates()
    out.to_parquet(path(season), index=False)
    log.info(f"injuries[{season}]: {len(out):,} injured-list moves")
    return out


def load(season: int, refresh: bool = False) -> pd.DataFrame:
    return fetch(season) if refresh or not path(season).exists() else pd.read_parquet(path(season))


def season_summary(moves: pd.DataFrame, season: int) -> pd.DataFrame:
    """Per player: placements, estimated days on the list, and whether any stint was long."""
    if moves.empty:
        return pd.DataFrame(columns=FEATURES)
    end = pd.Timestamp(date(season, *SEASON_END))
    out = {}
    for pid, g in moves.sort_values("date").groupby("player_id"):
        stints, days, long_ = 0, 0.0, False
        start = None
        for r in g.itertuples(index=False):
            if r.move == "placed":
                stints += 1
                start = start or r.date
                long_ |= r.list in LONG_LISTS
            elif r.move == "transferred":
                long_ |= r.list in LONG_LISTS
            elif r.move == "activated" and start is not None:
                days += max((r.date - start).days, 0)
                start = None
        if start is not None:
            days += max((end - start).days, 0)
        out[pid] = (stints, days, float(long_))
    return pd.DataFrame.from_dict(out, orient="index", columns=FEATURES)


def window_features(season: int, weights: dict[int, float], refresh_current: bool = False) -> pd.DataFrame:
    """Window-weighted injured-list history ending at ``season``, indexed by player_id; players never on a list are absent."""
    frames = []
    for back, w in weights.items():
        s = season - back
        summary = season_summary(load(s, refresh=refresh_current and back == 0), s)
        frames.append(summary.mul([w, w, 1.0]).assign(w=w))
    allf = pd.concat(frames)
    g = allf.groupby(level=0)
    out = pd.DataFrame({"il_stints": g["il_stints"].sum() / sum(weights.values()), "il_days": g["il_days"].sum() / sum(weights.values()),
                        "il_long": g["il_long"].max()})
    out.index.name = "player_id"
    return out
