"""MLB Pipeline top-100 lists, transcribed by hand from the public archive pages, matched to MLBAM ids.

Files ``cache/prospects/lists/pipeline_top100_{year}.txt`` hold ``rank|mlbam_id|name|position|team`` per
line; the id is present only where the page carried a current headshot. The rest are matched by
name (accent-insensitive) against the people cache, restricted to players who appeared in the
minors around the list year, with the Stats API name search as the fallback. The list published
in year Y reflects the end of season Y-1, so it is compared with snapshot season Y-1.
"""
from __future__ import annotations

import logging
import unicodedata

import pandas as pd

from v2.prospects.feeds import CACHE_DIR, STATSAPI, fetch_json

log = logging.getLogger(__name__)

LISTS_DIR = CACHE_DIR / "lists"
SEARCH = f"{STATSAPI}/people/search"


def _fold(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return " ".join(s.replace(".", "").replace("'", "").split())


def available_years() -> list[int]:
    return sorted(int(p.stem.split("_")[-1]) for p in LISTS_DIR.glob("pipeline_top100_*.txt"))


def read_list(year: int) -> pd.DataFrame:
    rows = []
    for line in (LISTS_DIR / f"pipeline_top100_{year}.txt").read_text().splitlines():
        if not line.strip():
            continue
        rank, pid, name, pos, team = line.split("|")
        rows.append({"list_year": year, "rank": int(rank), "player_id": int(pid) if pid else None, "name": name, "position": pos, "team": team})
    return pd.DataFrame(rows)


def _search(name: str) -> int | None:
    try:
        payload = fetch_json(SEARCH, {"names": name, "limit": 5, "sportIds": "1,11,12,13,14,16"}, timeout=30)
    except FileNotFoundError:
        return None
    people = [p for p in payload.get("people", []) if _fold(p.get("fullName", "")) == _fold(name)]
    return int(people[0]["id"]) if len(people) == 1 else None


def resolve(lists: pd.DataFrame, bio: pd.DataFrame, active: pd.DataFrame) -> pd.DataFrame:
    """Fill missing ids by name among players active in the minors in the list year or the year before."""
    bio = bio.assign(key=bio["full_name"].fillna("").map(_fold))
    out = lists.copy()
    for i, row in out[out["player_id"].isna()].iterrows():
        key = _fold(row["name"])
        pool = active[(active["season"] >= row["list_year"] - 1) & (active["season"] <= row["list_year"])]["player_id"]
        hits = bio[bio["key"].eq(key) & bio["player_id"].isin(pool)]["player_id"].unique()
        if len(hits) == 0:
            # Drop a suffix such as Jr., II, III and retry.
            base = " ".join(k for k in key.split() if k not in {"jr", "ii", "iii", "iv"})
            hits = bio[bio["key"].eq(base) & bio["player_id"].isin(pool)]["player_id"].unique()
        if len(hits) == 1:
            out.at[i, "player_id"] = int(hits[0])
        elif len(hits) == 0:
            found = _search(row["name"])
            if found is not None:
                out.at[i, "player_id"] = found
    matched = out["player_id"].notna().mean()
    log.info(f"lists: matched {matched:.0%} of {len(out)} entries to ids")
    out["player_id"] = out["player_id"].astype("Int64")
    out["season"] = out["list_year"] - 1
    return out


def load(bio: pd.DataFrame, active: pd.DataFrame, years: list[int] | None = None) -> pd.DataFrame:
    years = years or available_years()
    lists = pd.concat([read_list(y) for y in years], ignore_index=True)
    return resolve(lists, bio, active)
