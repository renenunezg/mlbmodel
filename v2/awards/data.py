"""Season lines, ballots, and winners behind the award boards.

Everything is cached under cache/awards so a refit touches no network. Ballots
come from Baseball-Reference (Rawlings publishes no votes, so Gold Gloves have
winners only). Regular-season fWAR comes from the Stats API per league, so a
player traded across leagues is judged on the league that votes on him.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from v2.players import savant
from v2.players.contracts import BBREF, CRAWL_DELAY_SECONDS, USER_AGENT, chadwick_map

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "awards"
STATSAPI = "https://statsapi.mlb.com/api/v1"
LEAGUES = {"AL": 103, "NL": 104}
BALLOT_AWARDS = {"MVP": "MVP", "CYA": "CY", "ROY": "ROY"}
POSITION_CODES = {"P": 1, "C": 2, "1B": 3, "2B": 4, "3B": 5, "SS": 6, "LF": 7, "CF": 8, "RF": 9}
_INT_STATS = ("gamesPlayed", "plateAppearances", "homeRuns", "rbi", "stolenBases", "gamesStarted", "gamesPitched",
              "outs", "wins", "saves", "holds", "strikeOuts", "battersFaced", "earnedRuns", "errors", "chances",
              "assists", "putOuts", "doublePlays", "games")


def _get(name: str, path: str, **params) -> dict:
    """One Stats API response, cached for good: every caller asks for a finished season."""
    target = CACHE_DIR / f"{name}.json"
    if target.exists():
        return json.loads(target.read_text())
    response = requests.get(f"{STATSAPI}/{path}", params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload))
    return payload


def _splits(payload: dict, what: str) -> list[dict]:
    block, = payload["stats"]
    splits = block["splits"]
    if len(splits) != block["totalSplits"]:
        raise ValueError(f"Truncated {what}")
    return splits


def _frame(splits: list[dict]) -> pd.DataFrame:
    rows = [{**s["stat"], "player_id": s["player"]["id"], "player_name": s["player"]["fullName"],
             "team_id": s.get("team", {}).get("id"), "team": s.get("team", {}).get("name")} for s in splits]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for column in _INT_STATS:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame


def league_lines(season: int, league: str, group: str, pool: str = "ALL") -> pd.DataFrame:
    name = f"lines_{group}_{pool.lower()}_{league}_{season}"
    splits = _splits(_get(name, "stats", stats="season", group=group, season=season, sportIds=1, leagueIds=LEAGUES[league],
                          playerPool=pool, limit=5000, gameType="R"), name)
    frame = _frame(splits)
    if frame.player_id.duplicated().any():
        raise ValueError(f"Duplicate players in {name}")
    return frame


def league_war(season: int, league: str, group: str) -> pd.DataFrame:
    name = f"war_{group}_{league}_{season}"
    splits = _splits(_get(name, "stats", stats="sabermetrics", group=group, season=season, sportIds=1,
                          leagueIds=LEAGUES[league], playerPool="ALL", limit=5000, gameType="R"), name)
    frame = pd.DataFrame([{"player_id": s["player"]["id"], "war": s["stat"].get("war"),
                           "ra9_war": s["stat"].get("ra9War"), "wrc_plus": s["stat"].get("wRcPlus")} for s in splits])
    if frame.player_id.duplicated().any():
        raise ValueError(f"Duplicate players in {name}")
    return frame


def team_context(season: int, league: str) -> pd.DataFrame:
    """Win percentage and whether the team reached the postseason, which voters know when ballots close."""
    standings = _get(f"standings_{league}_{season}", "standings", leagueId=LEAGUES[league], season=season)
    teams = [t for record in standings["records"] for t in record["teamRecords"]]
    postseason = _get(f"postseason_{season}", "schedule/postseason", season=season, sportId=1)
    entered = {g["teams"][side]["team"]["id"] for d in postseason["dates"] for g in d["games"] for side in ("home", "away")}
    if not entered or len(teams) < 14:  # 14 AL and 16 NL clubs before 2013.
        raise ValueError(f"Incomplete team context for {league} {season}")
    return pd.DataFrame([{"team_id": t["team"]["id"], "win_pct": t["wins"] / (t["wins"] + t["losses"]),
                          "playoff": float(t["team"]["id"] in entered)} for t in teams])


def fielding_lines(season: int, league: str, position: str) -> pd.DataFrame:
    # The endpoint ignores its position parameter and returns one row per player, club, and position.
    name = f"fielding_{league}_{season}"
    splits = _splits(_get(name, "stats", stats="season", group="fielding", season=season, sportIds=1,
                          leagueIds=LEAGUES[league], playerPool="ALL", limit=5000, gameType="R"), name)
    frame = _frame([s for s in splits if s["position"]["abbreviation"] == position])
    if frame.empty:
        raise ValueError(f"No {position} fielders in {league} {season}")
    # Innings arrive as "1234.2": two outs, not two tenths.
    whole = frame["innings"].astype(float).astype(int)
    frame["innings"] = whole + (frame["innings"].astype(float) - whole) * 10 / 3
    # A player traded within the league has one row per club at the position.
    return (frame.groupby("player_id", as_index=False, sort=False)
            .agg(player_name=("player_name", "first"), team=("team", "last"), innings=("innings", "sum"),
                 errors=("errors", "sum"), chances=("chances", "sum")))


def recipients(award_id: str, season: int) -> pd.DataFrame:
    """Winners of one award in one season; callers decide which seasons they may read."""
    payload = _get(f"recipients_{award_id}_{season}", f"awards/{award_id}/recipients", season=season)
    return pd.DataFrame([{"player_id": a["player"]["id"], "player_name": a["player"]["nameFirstLast"],
                          "position": a["player"]["primaryPosition"]["abbreviation"]} for a in payload["awards"]],
                        columns=["player_id", "player_name", "position"])


def ballots(season: int) -> pd.DataFrame:
    """Vote points per player from one season's voting page, with one row per vote getter."""
    path = CACHE_DIR / f"bbref_awards_{season}.html"
    if path.exists():
        html = path.read_text(encoding="utf-8")
    else:
        time.sleep(CRAWL_DELAY_SECONDS)
        response = requests.get(f"{BBREF}/awards/awards_{season}.shtml", headers={"User-Agent": USER_AGENT}, timeout=60)
        response.raise_for_status()
        html = response.text
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
    soup = BeautifulSoup(html.replace("<!--", "").replace("-->", ""), "html.parser")
    mapping = chadwick_map()
    rows = []
    for league in LEAGUES:
        for source, award in BALLOT_AWARDS.items():
            table = soup.find("table", id=f"{league}_{source}_voting")
            if table is None:
                raise ValueError(f"No {league} {source} ballot for {season}")
            for row in table.find("tbody").find_all("tr"):
                player = row.find(attrs={"data-stat": "player"})
                points = row.find(attrs={"data-stat": "points_won"})
                if player is None or points is None or not player.get("data-append-csv"):
                    continue
                rows.append({"season": season, "league": league, "award": award, "bbref_id": player["data-append-csv"],
                             "player_id": mapping.get(player["data-append-csv"]), "points": float(points.get_text() or 0)})
    frame = pd.DataFrame(rows)
    frame["share"] = frame.points / frame.groupby(["league", "award"]).points.transform("sum")
    return frame


def outs_above_average(season: int, position: str) -> pd.DataFrame:
    board = savant.load_board("oaa", season, pos=str(POSITION_CODES[position]))
    board = board.rename(columns={"outs_above_average": "oaa", "fielding_runs_prevented": "frv"})
    return board.groupby("player_id", as_index=False)[["oaa", "frv"]].sum()


def catcher_runs(season: int) -> pd.DataFrame:
    framing = savant.load_board("catcher_framing", season).rename(columns={"rv_tot": "framing"})[["player_id", "framing"]]
    blocking = savant.load_board("catcher_blocking", season)[["player_id", "catcher_blocking_runs"]]
    throwing = savant.load_board("catcher_throwing", season)[["player_id", "catcher_stealing_runs"]]
    return (framing.merge(blocking, on="player_id", how="outer").merge(throwing, on="player_id", how="outer")
            .rename(columns={"catcher_blocking_runs": "blocking", "catcher_stealing_runs": "throwing"}).fillna(0.0)
            .groupby("player_id", as_index=False).sum())
