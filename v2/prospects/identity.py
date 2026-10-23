"""Who the prospects are: bio, draft, roster history and rookie eligibility from the Stats API.

People are hydrated once (draft, education, roster entries) and cached under ``cache/prospects/``.
Rookie eligibility follows MLB's rule: fewer than 130 MLB at-bats, fewer than 50 MLB innings, and
no more than 45 days on an MLB active roster outside September, all before the snapshot date.
MLB at-bats and innings come from the player model's cached league season lines (read-only);
active-roster days come from roster entries on MLB clubs with status Active.
"""
from __future__ import annotations

import logging
import re
from datetime import date
from functools import cache

import pandas as pd

from v2.players import bio
from v2.prospects.feeds import CACHE_DIR, STATSAPI, fetch_json

log = logging.getLogger(__name__)

PEOPLE_BATCH = 100
PEOPLE_PATH = CACHE_DIR / "people.parquet"
ROSTER_PATH = CACHE_DIR / "roster_entries.parquet"
HYDRATE = "draft,education,rosterEntries"
ROOKIE_AB = 130
ROOKIE_IP = 50.0
ROOKIE_DAYS = 45
CHECKPOINT_BATCHES = 20
SEPTEMBER = 9
PEOPLE_COLUMNS = ["player_id", "full_name", "birth_date", "bats", "throws", "primary_position", "birth_country", "height", "weight",
                  "mlb_debut", "draft_year", "draft_round", "draft_pick", "signing_bonus_musd", "school_type", "draft_school"]
ROSTER_COLUMNS = ["player_id", "team_id", "start", "end", "status"]
HS_PATTERN = re.compile(r"\bHS\b|HIGH SCHOOL|\(HS\)|\bH\.S\.")


def _get(path: str, **params) -> dict:
    return fetch_json(f"{STATSAPI}/{path}", params, timeout=90)


@cache
def mlb_team_ids() -> frozenset[int]:
    return frozenset(t["id"] for t in _get("teams", sportId=1)["teams"])


def _school_type(person: dict, draft: dict | None) -> str:
    """Where the signed draft pick came from; the latest draft is the one that started the pro career."""
    edu = person.get("education", {}) or {}
    if draft is None:
        return "international" if person.get("birthCountry") not in {"USA", "Canada", "Puerto Rico"} else "undrafted"
    name = ((draft.get("school") or {}).get("name", "") or "").upper()
    if HS_PATTERN.search(name):
        return "high_school"
    if name or edu.get("colleges"):
        return "college"
    return "high_school" if edu.get("highschools") else "unknown"


def _rows(person: dict) -> tuple[dict, list[dict]]:
    drafts = [d for d in person.get("drafts", []) if d.get("isDrafted", True)]
    draft = max(drafts, key=lambda d: int(d.get("year", 0))) if drafts else None
    bonus = draft.get("signingBonus") if draft else None
    bio_row = {
        "player_id": person["id"], "full_name": person.get("fullName"), "birth_date": person.get("birthDate"),
        "bats": person.get("batSide", {}).get("code"), "throws": person.get("pitchHand", {}).get("code"),
        "primary_position": person.get("primaryPosition", {}).get("abbreviation"), "birth_country": person.get("birthCountry"),
        "height": person.get("height"), "weight": person.get("weight"), "mlb_debut": person.get("mlbDebutDate"),
        "draft_year": int(draft["year"]) if draft else None,
        "draft_round": draft.get("pickRound") if draft else None,
        "draft_pick": draft.get("pickNumber") if draft else None,
        "signing_bonus_musd": float(bonus) / 1e6 if bonus not in (None, "") else None,
        "school_type": _school_type(person, draft), "draft_school": (draft.get("school") or {}).get("name") if draft else None,
    }
    roster = [{"player_id": person["id"], "team_id": r.get("team", {}).get("id"), "start": r.get("startDate"), "end": r.get("endDate"),
               "status": (r.get("status") or {}).get("code")} for r in person.get("rosterEntries", []) if r.get("startDate")]
    return bio_row, roster


def _people_batch(chunk: list[int]) -> list[dict]:
    """One hydrated people request; a 404 (one retired id poisons the batch) falls back to single lookups."""
    try:
        return _get("people", personIds=",".join(map(str, chunk)), hydrate=HYDRATE).get("people", [])
    except FileNotFoundError:
        pass
    out = []
    for pid in chunk:
        try:
            out.extend(_get("people", personIds=str(pid), hydrate=HYDRATE).get("people", []))
        except FileNotFoundError:
            log.warning(f"people: {pid} not found")
    return out


def people(player_ids: list[int], *, refresh_ids: set[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bio rows and roster entries for the ids, fetching only ids missing from the cache (or in ``refresh_ids``)."""
    cached = pd.read_parquet(PEOPLE_PATH) if PEOPLE_PATH.exists() else pd.DataFrame(columns=PEOPLE_COLUMNS)
    roster = pd.read_parquet(ROSTER_PATH) if ROSTER_PATH.exists() else pd.DataFrame(columns=ROSTER_COLUMNS)
    refresh_ids = refresh_ids or set()
    missing = sorted((set(player_ids) - set(cached["player_id"].astype(int))) | (refresh_ids & set(player_ids)))
    if missing:
        cached = cached[~cached["player_id"].isin(missing)]
        roster = roster[~roster["player_id"].isin(missing)]
        bio_rows, roster_rows = [], []
        batches = list(range(0, len(missing), PEOPLE_BATCH))
        for i, start in enumerate(batches, 1):
            chunk = missing[start:start + PEOPLE_BATCH]
            for person in _people_batch(chunk):
                b, r = _rows(person)
                bio_rows.append(b)
                roster_rows.extend(r)
            # An id the API no longer serves gets a marker row so it is not refetched every run.
            seen = {b["player_id"] for b in bio_rows}
            bio_rows.extend({"player_id": pid} for pid in chunk if pid not in seen)
            log.info(f"people: hydrated {min(start + PEOPLE_BATCH, len(missing))}/{len(missing)}")
            if i % CHECKPOINT_BATCHES == 0 or i == len(batches):
                # Checkpoint so a dropped connection keeps the batches already fetched.
                cached = pd.concat([cached, pd.DataFrame(bio_rows, columns=PEOPLE_COLUMNS)], ignore_index=True)
                roster = pd.concat([roster, pd.DataFrame(roster_rows, columns=ROSTER_COLUMNS)], ignore_index=True)
                bio_rows, roster_rows = [], []
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cached.to_parquet(PEOPLE_PATH, index=False)
                roster.to_parquet(ROSTER_PATH, index=False)
    cached["player_id"] = cached["player_id"].astype("int64")
    roster["player_id"] = roster["player_id"].astype("int64")
    return cached[cached["player_id"].isin(player_ids)], roster[roster["player_id"].isin(player_ids)]


def active_mlb_days(roster: pd.DataFrame, as_of: date, mlb_ids: set[int]) -> pd.Series:
    """Days on an MLB active roster before September of each season, through ``as_of``, per player."""
    r = roster[roster["team_id"].isin(mlb_ids) & roster["status"].eq("A")].copy()
    if r.empty:
        return pd.Series(dtype="float64", name="mlb_days")
    r["start"] = pd.to_datetime(r["start"])
    r["end"] = pd.to_datetime(r["end"]).fillna(pd.Timestamp(as_of))
    r["end"] = r["end"].clip(upper=pd.Timestamp(as_of))
    r = r[r["start"] <= r["end"]]
    days = []
    for season in range(int(r["start"].dt.year.min()), as_of.year + 1):
        lo, hi = pd.Timestamp(season, 3, 15), pd.Timestamp(season, SEPTEMBER, 1)
        s = r["start"].clip(lower=lo)
        e = r["end"].clip(upper=hi)
        d = (e - s).dt.days.clip(lower=0)
        days.append(d.groupby(r["player_id"]).sum())
    return pd.concat(days).groupby(level=0).sum().rename("mlb_days")


def mlb_career_to(season: int) -> pd.DataFrame:
    """Cumulative MLB at-bats and innings per player through ``season`` from the player model's league caches."""
    seasons = list(range(2005, season + 1))
    hit = bio.league_season_lines(seasons, "hitting")[["player_id", "atBats"]]
    pit = bio.league_season_lines(seasons, "pitching")[["player_id", "outs"]]
    ab = hit.groupby("player_id")["atBats"].sum().rename("mlb_ab")
    ip = (pit.groupby("player_id")["outs"].sum() / 3.0).rename("mlb_ip")
    return pd.concat([ab, ip], axis=1).fillna(0.0)


def rookie_status(player_ids: list[int], as_of: date, roster: pd.DataFrame) -> pd.DataFrame:
    """Per player: MLB at-bats, innings, active days and rookie eligibility as of the date."""
    career = mlb_career_to(as_of.year).reindex(player_ids).fillna(0.0)
    days = active_mlb_days(roster, as_of, mlb_team_ids()).reindex(player_ids).fillna(0.0)
    out = career.assign(mlb_days=days)
    out["rookie_eligible"] = (out["mlb_ab"] < ROOKIE_AB) & (out["mlb_ip"] < ROOKIE_IP) & (out["mlb_days"] <= ROOKIE_DAYS)
    out.index.name = "player_id"
    return out.reset_index()


def on_forty_man(roster: pd.DataFrame, as_of: date, mlb_ids: frozenset[int]) -> pd.Series:
    """Whether a player has an open Active entry with an MLB club as of the date (a 40-man proxy)."""
    r = roster[(pd.to_datetime(roster["start"]) <= pd.Timestamp(as_of)) & roster["end"].isna() & roster["status"].eq("A")]
    return r.groupby("player_id")["team_id"].agg(lambda t: bool(t.isin(mlb_ids).any())).rename("on_40_man")


def mlb_linked(roster: pd.DataFrame, mlb_ids: frozenset[int]) -> set[int]:
    """Players whose cached roster history has an open entry with an MLB club: the ones whose eligibility can still move."""
    r = roster[roster["end"].isna() & roster["team_id"].isin(mlb_ids)]
    return set(r["player_id"].astype(int))
