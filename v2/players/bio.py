"""Who is in the population, what they play, and how often they were hurt.

MLB Stats API for rosters, people, transactions, and career season lines;
the Statcast cache for positions actually played. Everything keyed by MLBAM id.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from backend.data.game_types import regular_season_pitches
from backend.team_mappings import TEAM_ID_BY_CODE

log = logging.getLogger(__name__)

STATSAPI = "https://statsapi.mlb.com/api/v1"
CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "players"
STATCAST_DIR = Path(__file__).resolve().parents[2] / "cache"
PEOPLE_BATCH = 100
TIMEOUT = 60
SURGERY_SOURCE = "https://docs.google.com/spreadsheets/d/1gQujXQQGOVNaiuwSN680Hq-FDVsCwvN-3AazykOBON0/export?format=csv&gid=0"
# Statcast uses standard defensive numbering: third base is 5, shortstop is 6.
FIELD_POSITIONS = {2: "C", 3: "1B", 4: "2B", 5: "3B", 6: "SS", 7: "LF", 8: "CF", 9: "RF"}
POSITION_GROUPS = ("C", "1B", "2B", "SS", "3B", "LF", "CF", "RF", "DH", "SP", "RP")
# A pitcher is a starter when at least this share of his appearances opened the game.
STARTER_SHARE = 0.5
# A secondary position needs this share of the primary position's innings.
SECONDARY_SHARE = 0.2
CODE_BY_TEAM_ID = {team_id: code for code, team_id in TEAM_ID_BY_CODE.items()}


def _get(path: str, **params) -> dict:
    response = requests.get(f"{STATSAPI}/{path}", params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def season_dates(season: int) -> tuple[date, date]:
    payload = _get("seasons", sportId=1, season=season)["seasons"][0]
    return date.fromisoformat(payload["regularSeasonStartDate"]), date.fromisoformat(payload["regularSeasonEndDate"])


def _people(ids: list[int], hydrate: str | None = None) -> list[dict]:
    out: list[dict] = []
    for start in range(0, len(ids), PEOPLE_BATCH):
        chunk = ids[start:start + PEOPLE_BATCH]
        params = {"personIds": ",".join(str(i) for i in chunk)}
        if hydrate:
            params["hydrate"] = hydrate
        out.extend(_get("people", **params).get("people", []))
    return out


def _transactions(start: date, end: date) -> list[dict]:
    """All MLB transactions in a date range, fetched month by month and cached per past season."""
    out: list[dict] = []
    cursor = start
    while cursor <= end:
        stop = min(end, cursor + timedelta(days=30))
        out.extend(_get("transactions", sportId=1, startDate=str(cursor), endDate=str(stop)).get("transactions", []))
        cursor = stop + timedelta(days=1)
    return out


_TRANSACTIONS: dict[tuple[int, date], pd.DataFrame] = {}


def _season_transactions(season: int, as_of: date) -> pd.DataFrame:
    key = (season, as_of)
    if key in _TRANSACTIONS:
        return _TRANSACTIONS[key]
    path = CACHE_DIR / f"transactions_{season}.parquet"
    end = min(as_of, date(season, 12, 31))
    if path.exists() and season < as_of.year:
        _TRANSACTIONS[key] = pd.read_parquet(path)
        return _TRANSACTIONS[key]
    rows = [
        {
            "player_id": t.get("person", {}).get("id"),
            "date": t.get("date"),
            "type": t.get("typeDesc"),
            "to_team_id": t.get("toTeam", {}).get("id"),
            "description": t.get("description", ""),
        }
        for t in _transactions(date(season, 1, 1), end)
    ]
    frame = pd.DataFrame(rows, columns=["player_id", "date", "type", "to_team_id", "description"])
    frame = frame.dropna(subset=["player_id"]).astype({"player_id": "int64"})
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    if season < as_of.year:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    _TRANSACTIONS[key] = frame
    return frame


def _age_on(birth: pd.Series, when: date) -> pd.Series:
    born = pd.to_datetime(birth)
    return ((pd.Timestamp(when) - born).dt.days / 365.25).floordiv(1).astype("Int64")


def population(season: int, as_of: date) -> pd.DataFrame:
    """Every 40-man or 60-day-IL player plus pending free agents, with bio fields.

    ``age`` is the age on June 30 of the next season to be played.
    """
    rows: list[dict] = []
    for code, team_id in TEAM_ID_BY_CODE.items():
        for entry in _get(f"teams/{team_id}/roster", rosterType="40Man", season=season).get("roster", []):
            rows.append({
                "player_id": entry["person"]["id"], "team": code, "team_id": team_id,
                "status": "il_60" if entry["status"]["code"] == "D60" else "forty_man",
            })
    roster = pd.DataFrame(rows).drop_duplicates("player_id")
    _, season_end = season_dates(season)
    if as_of > season_end:
        tx = _season_transactions(season, as_of)
        tx = tx[tx["date"] > season_end].sort_values("date")
        declared = set(tx.loc[tx["type"] == "Declared Free Agency", "player_id"])
        signed = set(tx.loc[tx["type"].isin(["Signed as Free Agent", "Signed"]), "player_id"])
        pending = declared - signed - set(roster["player_id"])
        roster = pd.concat([
            roster,
            pd.DataFrame({"player_id": sorted(pending), "team": "", "team_id": pd.NA, "status": "free_agent"}),
        ], ignore_index=True)
    people = _people(roster["player_id"].astype(int).tolist())
    bio = pd.DataFrame([
        {
            "player_id": p["id"], "player_name": p.get("fullName"), "birth_date": p.get("birthDate"),
            "bats": p.get("batSide", {}).get("code"), "throws": p.get("pitchHand", {}).get("code"),
            "mlb_debut": p.get("mlbDebutDate"), "listed_position": p.get("primaryPosition", {}).get("abbreviation"),
        }
        for p in people
    ])
    out = roster.merge(bio, on="player_id", how="left")
    next_season = season if as_of <= season_end else season + 1
    out["age"] = _age_on(out["birth_date"], date(next_season, 6, 30))
    out["player_id"] = out["player_id"].astype("int64")
    return out


def _load_pitches(season: int, columns: list[str]) -> pd.DataFrame:
    path = STATCAST_DIR / f"statcast_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run python -m v2.data.build_cache --years {season}")
    needed = list(dict.fromkeys([*columns, "game_type", "game_date"]))
    return regular_season_pitches(pd.read_parquet(path, columns=needed), season)


def _pitcher_roles(pitches: pd.DataFrame) -> pd.DataFrame:
    """SP when at least STARTER_SHARE of a pitcher's games began with him in inning 1."""
    first = pitches[pitches["inning"] == 1]
    side = np.where(first["inning_topbot"] == "Top", first["home_team"], first["away_team"])
    first = first.assign(side=side)
    starters = first.groupby(["game_pk", "side"])["pitcher"].agg(lambda s: s.mode().iloc[0]).reset_index()
    starts = starters.groupby("pitcher").size()
    games = pitches.groupby("pitcher")["game_pk"].nunique()
    share = (starts.reindex(games.index).fillna(0) / games)
    return pd.DataFrame({
        "player_id": games.index.astype("int64"), "position": np.where(share >= STARTER_SHARE, "SP", "RP"),
        "innings": (pitches.groupby("pitcher").size() / 15.0).values,  # about 15 pitches an inning
    })


def positions(seasons: list[int], before: date | None = None) -> pd.DataFrame:
    """Innings fielded by position per player-season, from who stood where on every pitch.

    Batters with plate appearances but no fielding innings are DH. Pitchers are SP or RP.
    ``before`` keeps only games played before that date, for as-of evaluations.
    """
    fielder_cols = [f"fielder_{k}" for k in FIELD_POSITIONS]
    frames = []
    for season in seasons:
        pitches = _load_pitches(
            season, ["game_pk", "inning", "inning_topbot", "home_team", "away_team", "batter", "pitcher", "events", *fielder_cols],
        )
        if before is not None:
            pitches = pitches[pd.to_datetime(pitches["game_date"]).dt.date < before]
        half = pitches[["game_pk", "inning", "inning_topbot", *fielder_cols]].drop_duplicates()
        long = half.melt(id_vars=["game_pk", "inning", "inning_topbot"], value_vars=fielder_cols,
                         var_name="slot", value_name="player_id").dropna(subset=["player_id"])
        long["position"] = long["slot"].str.slice(8).astype(int).map(FIELD_POSITIONS)
        fielded = long.groupby(["player_id", "position"]).size().rename("innings").reset_index()
        fielded["player_id"] = fielded["player_id"].astype("int64")
        # A DH game: the batter had a plate appearance but neither fielded nor pitched in that game.
        batter_games = pitches.loc[pitches["events"].notna(), ["game_pk", "batter"]].dropna().drop_duplicates()
        batter_games = batter_games.rename(columns={"batter": "player_id"}).astype({"player_id": "int64"})
        fielded_games = long[["game_pk", "player_id"]].drop_duplicates().astype({"player_id": "int64"})
        pitched_games = pitches[["game_pk", "pitcher"]].dropna().drop_duplicates().rename(columns={"pitcher": "player_id"}).astype({"player_id": "int64"})
        busy = pd.concat([fielded_games, pitched_games]).drop_duplicates()
        dh_games = batter_games.merge(busy, on=["game_pk", "player_id"], how="left", indicator=True)
        dh_games = dh_games[dh_games["_merge"] == "left_only"]
        dh = dh_games.groupby("player_id").size().mul(9.0).rename("innings").reset_index()
        dh["position"] = "DH"
        roles = _pitcher_roles(pitches)
        frame = pd.concat([fielded, dh, roles], ignore_index=True)
        frame["season"] = season
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["player_id"] = out["player_id"].astype("int64")
    return out[["player_id", "season", "position", "innings"]]


def primary_position(pos: pd.DataFrame, window_weights: dict[int, float]) -> pd.DataFrame:
    """Primary position by weighted innings, with secondary positions, one row per player."""
    weighted = pos.assign(w=pos["innings"] * pos["season"].map(window_weights).fillna(0.0))
    totals = weighted.groupby(["player_id", "position"])["w"].sum().reset_index()
    totals = totals[totals["w"] > 0]
    order = totals.sort_values(["player_id", "w"], ascending=[True, False])
    primary = order.drop_duplicates("player_id").rename(columns={"position": "position_group", "w": "primary_w"})
    merged = order.merge(primary[["player_id", "position_group", "primary_w"]], on="player_id")
    secondary = merged[(merged["position"] != merged["position_group"]) & (merged["w"] >= SECONDARY_SHARE * merged["primary_w"])]
    secondary = secondary.groupby("player_id")["position"].agg(list).rename("secondary_positions")
    out = primary[["player_id", "position_group"]].merge(secondary, on="player_id", how="left")
    out["secondary_positions"] = out["secondary_positions"].apply(lambda v: v if isinstance(v, list) else [])
    return out


def il_days(seasons: list[int], as_of: date) -> pd.DataFrame:
    """Days on the injured list per player-season from placement and activation transactions."""
    out = []
    for season in seasons:
        tx = _season_transactions(season, as_of)
        _, season_end = season_dates(season)
        text = tx["description"].str.lower()
        placed = tx[text.str.contains("placed") & text.str.contains("injured list")]
        activated = tx[text.str.contains("activated") & text.str.contains("injured list")]
        for player_id, group in placed.groupby("player_id"):
            ends = activated.loc[activated["player_id"] == player_id, "date"].sort_values().tolist()
            starts = sorted(group["date"])
            days = 0
            for i, start in enumerate(starts):
                # A stint ends at the next activation, the next placement, or the season's end.
                limit = min(season_end, as_of)
                if i + 1 < len(starts):
                    limit = min(limit, starts[i + 1])
                end = next((e for e in ends if e > start), limit)
                days += max(0, (min(end, limit) - start).days)
            out.append({"player_id": int(player_id), "season": season, "il_days": days})
    return pd.DataFrame(out, columns=["player_id", "season", "il_days"])


def season_lines(player_ids: list[int], group: str) -> pd.DataFrame:
    """Career season-by-season stat lines, cached; only ids missing from the cache are fetched."""
    path = CACHE_DIR / f"season_lines_{group}.parquet"
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["player_id", "season"])
    missing = sorted(set(player_ids) - set(cached["player_id"].astype(int)))
    if missing:
        rows = []
        for person in _people(missing, hydrate=f"stats(group=[{group}],type=[yearByYear],sportId=1)"):
            for block in person.get("stats", []):
                for split in block.get("splits", []):
                    stat = dict(split.get("stat", {}))
                    stat.update({"player_id": person["id"], "season": int(split["season"]), "age": split.get("stat", {}).get("age")})
                    rows.append(stat)
        fresh = pd.DataFrame(rows)
        # A fetched player with no MLB seasons still gets a marker row so he is not refetched.
        seen = {r["player_id"] for r in rows}
        markers = pd.DataFrame({"player_id": [i for i in missing if i not in seen], "season": 0})
        cached = pd.concat([cached, fresh, markers], ignore_index=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached.to_parquet(path, index=False)
        log.info(f"season_lines[{group}]: fetched {len(missing)} players")
    cached["player_id"] = cached["player_id"].astype("int64")
    return cached[cached["season"] > 0]


def league_season_lines(seasons: list[int], group: str, *, refresh: bool = False) -> pd.DataFrame:
    """Complete MLB seasons, including players no longer in today's population.

    Career histories fetched for current players are not a historical league
    sample. Cache each complete regular season independently and reject truncation.
    """
    if group not in {"hitting", "pitching"}:
        raise ValueError("Unsupported season-line group")
    frames = []
    for season in seasons:
        path = CACHE_DIR / f"league_{group}_{season}.parquet"
        if path.exists() and not refresh:
            frame = pd.read_parquet(path)
        else:
            blocks = _get("stats", stats="season", group=group, season=season,
                          sportIds=1, playerPool="ALL", limit=10000, gameType="R")["stats"]
            block, = blocks
            splits = block["splits"]
            if len(splits) != block["totalSplits"] or not splits:
                raise ValueError(f"Incomplete MLB {group} season {season}")
            frame = pd.DataFrame([{**s["stat"], "player_id": s["player"]["id"],
                                   "season": int(s["season"])} for s in splits])
            if frame["player_id"].duplicated().any() or not frame["season"].eq(season).all():
                raise ValueError(f"Ambiguous MLB {group} totals for {season}")
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def ages(season: int) -> pd.Series:
    """Seasonal age in ``season`` by player_id for everyone with an MLB line that season or the one before."""
    parts = []
    for s, shift in ((season, 0), (season - 1, 1)):
        for group in ("hitting", "pitching"):
            lines = league_season_lines([s], group)
            parts.append(pd.Series(lines["age"].to_numpy(float) + shift, index=lines["player_id"].to_numpy()))
    return pd.concat(parts).groupby(level=0).first()


def offseason_roster(season: int) -> pd.DataFrame:
    """Dated January 1 reserve roster, before the target season's outcomes.

    Offseason reserve lists include injured players. Unlike observed MLB playing
    time, membership does not condition on surviving or returning during the year.
    """
    path = CACHE_DIR / f"roster_jan1_{season}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    rows = []
    for team_id in TEAM_ID_BY_CODE.values():
        roster = _get(f"teams/{team_id}/roster", rosterType="40Man", date=f"{season}-01-01")["roster"]
        if not 15 <= len(roster) <= 45:
            raise ValueError(f"Unexpected historical roster coverage: {team_id}, {season}")
        rows.extend({"player_id": r["person"]["id"], "team_id": team_id,
                     "status": r["status"]["code"], "season": season} for r in roster)
    frame = pd.DataFrame(rows)
    # Historical transfers can appear on both clubs. Membership is the union;
    # never count a player twice when constructing a projection population.
    frame = frame.drop_duplicates(["player_id", "team_id", "status"])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def october_roster(season: int) -> pd.DataFrame:
    """Reserve and injured membership before November free agency and trades.

    MLB's dated 40Man response includes 60-day IL players, so its historical
    count can exceed 40. January membership would use post-signing information.
    """
    if season < 2009 or date(season, 10, 1) > date.today():
        raise ValueError("October roster is outside the verified historical coverage")
    path = CACHE_DIR / f"roster_oct1_{season}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    rows = []
    for team_id in sorted(set(TEAM_ID_BY_CODE.values())):
        roster = _get(f"teams/{team_id}/roster", rosterType="40Man", date=f"{season}-10-01")["roster"]
        if not 20 <= len(roster) <= 70:
            raise ValueError(f"Unexpected October roster coverage: {team_id}, {season}")
        rows.extend({"player_id": r["person"]["id"], "team_id": team_id,
                     "status": r["status"]["code"], "note": r.get("note", ""), "season": season} for r in roster)
    frame = pd.DataFrame(rows).drop_duplicates(["player_id", "team_id", "status"])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def surgery_dates(*, refresh: bool = False) -> pd.DataFrame:
    """Documented TJ operations, never future return dates or recovery outcomes.

    Jon Roegele's public history is also cited by MLB. Some historical dates are
    approximate. Absence from this list does not establish that a player is healthy.
    """
    path = CACHE_DIR / "tommy_john_surgeries.csv"
    if path.exists() and not refresh:
        raw = path.read_text()
    else:
        response = requests.get(SURGERY_SOURCE, timeout=TIMEOUT)
        response.raise_for_status()
        raw = response.text
    f = pd.read_csv(StringIO(raw), skiprows=1, usecols=["mlbamid", "TJ Surgery Date"])
    f["player_id"] = pd.to_numeric(f.pop("mlbamid"), errors="raise")
    f["date"] = pd.to_datetime(f.pop("TJ Surgery Date"), format="%m/%d/%Y", errors="raise")
    if len(f) < 1000 or f.date.isna().any() or f.date.gt(pd.Timestamp(date.today())).any():
        raise ValueError("Incomplete or future-dated surgery source")
    f = f.dropna(subset=["player_id"]).astype({"player_id": int})
    if f.duplicated(["player_id", "date"]).any():
        raise ValueError("Duplicate surgery event")
    if refresh or not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(".tmp")
        pending.write_text(raw)
        pending.replace(path)
    return f
