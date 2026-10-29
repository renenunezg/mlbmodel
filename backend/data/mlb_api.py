"""MLB Stats API fetchers: schedule, scores, probable starters, lineups, batting splits."""

import logging
from datetime import date, timedelta

import pandas as pd
import requests

from backend.data.game_types import INACTIVE_GAME_STATUSES, PREDICTION_GAME_TYPES
from backend.log import setup_logging
from backend.team_mappings import TEAM_ID_BY_CODE, normalize_team

log = logging.getLogger(__name__)

BASE_URL = "https://statsapi.mlb.com/api/v1"

# Cache for pitcher handedness lookups (pitcher_id -> 'L'/'R')
_handedness_cache: dict[int, str] = {}


def _schedule_status(game: dict) -> str:
    status = game.get("status", {})
    detailed = status.get("detailedState", "")
    # These games can still have an abstract Preview state and a start time.
    if detailed in INACTIVE_GAME_STATUSES:
        return detailed
    if game.get("ifNecessary") == "Y" and status.get("abstractGameState") == "Preview":
        return "If Necessary"
    return status.get("abstractGameState", "Scheduled")


def _batch_fetch_handedness(pitcher_ids: list[int]) -> dict[int, str]:
    ids_to_fetch = [pid for pid in pitcher_ids if pid and pid not in _handedness_cache]
    if not ids_to_fetch:
        return _handedness_cache

    # MLB API supports comma-separated person IDs
    id_str = ",".join(str(pid) for pid in ids_to_fetch)
    try:
        resp = requests.get(f"{BASE_URL}/people", params={"personIds": id_str}, timeout=15)
        resp.raise_for_status()
        for person in resp.json().get("people", []):
            pid = person.get("id")
            hand = person.get("pitchHand", {}).get("code")
            if pid and hand:
                _handedness_cache[pid] = hand
    except Exception as e:
        log.warning(f"Batch handedness fetch failed: {e}")

    return _handedness_cache


def _team_code(side: dict, game_pk: int) -> str:
    """A schedule row without a team code must stop the load, not store ''."""
    code = (side.get("team") or {}).get("abbreviation")
    if not code:
        raise ValueError(f"Schedule game {game_pk} has no team abbreviation; check the hydrate=team payload")
    return code


def fetch_schedule(game_date: date = None) -> pd.DataFrame:
    """Regular-season and playoff schedule, including probable pitchers and game type."""
    if game_date is None:
        game_date = date.today()

    date_str = game_date.strftime("%Y-%m-%d")

    params = {
        "sportId": 1,
        "date": date_str,
        "gameTypes": ",".join(PREDICTION_GAME_TYPES),
        "hydrate": "probablePitcher(note),venue,team",
    }
    resp = requests.get(f"{BASE_URL}/schedule", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            if game.get("gameType") not in PREDICTION_GAME_TYPES:
                continue
            home = game.get("teams", {}).get("home", {})
            away = game.get("teams", {}).get("away", {})

            home_pitcher = home.get("probablePitcher", {})
            away_pitcher = away.get("probablePitcher", {})

            rows.append({
                "game_pk": game["gamePk"],
                "game_type": game["gameType"],
                "game_date": game.get("officialDate") or date_entry["date"],
                "start_time": game.get("rescheduleDate") or game.get("gameDate"),
                "home_team": _team_code(home, game["gamePk"]),
                "away_team": _team_code(away, game["gamePk"]),
                "home_score": home.get("score"),
                "away_score": away.get("score"),
                "status": _schedule_status(game),
                "venue": game.get("venue", {}).get("name", ""),
                "home_pitcher_name": home_pitcher.get("fullName"),
                "home_pitcher_id": home_pitcher.get("id"),
                "away_pitcher_name": away_pitcher.get("fullName"),
                "away_pitcher_id": away_pitcher.get("id"),
            })

    df = pd.DataFrame(rows)

    if not df.empty:
        # Normalize team abbreviations
        df["home_team"] = df["home_team"].apply(normalize_team)
        df["away_team"] = df["away_team"].apply(normalize_team)
        df = df[df.home_team.isin(TEAM_ID_BY_CODE) & df.away_team.isin(TEAM_ID_BY_CODE)].copy()

        # Batch fetch handedness for all pitchers
        all_pitcher_ids = (
            df["home_pitcher_id"].dropna().astype(int).tolist() +
            df["away_pitcher_id"].dropna().astype(int).tolist()
        )
        _batch_fetch_handedness(list(set(all_pitcher_ids)))
        df["home_pitcher_hand"] = df["home_pitcher_id"].map(_handedness_cache)
        df["away_pitcher_hand"] = df["away_pitcher_id"].map(_handedness_cache)

    return df


def fetch_probable_starters(game_date: date = None, days_ahead: int = 7) -> pd.DataFrame:
    """Fetch probable starters for upcoming games.

    Returns DataFrame with columns:
        game_pk, game_date, team, pitcher_name, pitcher_id, handedness, is_home
    """
    if game_date is None:
        game_date = date.today()

    end_date = game_date + timedelta(days=days_ahead)

    params = {
        "sportId": 1,
        "startDate": game_date.strftime("%Y-%m-%d"),
        "endDate": end_date.strftime("%Y-%m-%d"),
        "hydrate": "probablePitcher(note),team",
        "gameTypes": ",".join(PREDICTION_GAME_TYPES),
    }
    resp = requests.get(f"{BASE_URL}/schedule", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            if game.get("gameType") not in PREDICTION_GAME_TYPES:
                continue
            game_pk = game["gamePk"]
            gd = game.get("officialDate") or date_entry["date"]

            for side, is_home in [("home", True), ("away", False)]:
                team_data = game.get("teams", {}).get(side, {})
                pitcher = team_data.get("probablePitcher", {})

                team = normalize_team(team_data.get("team", {}).get("abbreviation", ""))
                if team not in TEAM_ID_BY_CODE:
                    continue
                rows.append({
                    "game_pk": game_pk, "game_date": gd, "team": team,
                    "pitcher_name": pitcher.get("fullName"),
                    "pitcher_id": pitcher.get("id") if pitcher.get("fullName") else None,
                    "is_home": is_home,
                })

    df = pd.DataFrame(rows)

    # Batch fetch handedness
    if not df.empty:
        pitcher_ids = df["pitcher_id"].dropna().astype(int).tolist()
        _batch_fetch_handedness(list(set(pitcher_ids)))
        df["handedness"] = df["pitcher_id"].map(_handedness_cache)

    return df


def fetch_lineup(game_pk: int) -> dict[str, list[int]]:
    """Posted batting orders {home: [pid1..pid9], away: [...]}. Empty lists pre-posting."""
    url = f"{BASE_URL}/game/{game_pk}/boxscore"
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            last_exc = e
    else:
        raise RuntimeError(f"fetch_lineup({game_pk}) failed: {last_exc}")

    out: dict[str, list[int]] = {"home": [], "away": []}
    teams = data.get("teams", {})
    for side in ("home", "away"):
        order = teams.get(side, {}).get("battingOrder", []) or []
        out[side] = [int(pid) for pid in order]
    return out


_active_roster_cache: dict[tuple[int, date], list[int]] = {}


def fetch_active_pitchers(team_id: int, game_date: date | None = None) -> list[int]:
    """Active 26-man-roster pitcher ids for a team. Cached in memory per process."""
    game_date = game_date or date.today()
    key = (team_id, game_date)
    if key in _active_roster_cache:
        return _active_roster_cache[key]

    try:
        resp = requests.get(
            f"{BASE_URL}/teams/{team_id}/roster",
            params={"rosterType": "active", "date": game_date.isoformat()},
            timeout=10,
        )
        resp.raise_for_status()
        roster = resp.json().get("roster", [])
    except Exception as e:
        log.warning(f"fetch_active_pitchers({team_id}) failed: {e}")
        return []

    ids = [
        int(p["person"]["id"])
        for p in roster
        if p.get("position", {}).get("abbreviation") in ("P", "TWP")
    ]
    _active_roster_cache[key] = ids
    return ids


def fetch_game_pitchers(game_pk: int) -> dict[str, list[int]]:
    """Game-specific eligible pitchers, including two-way players, from the boxscore.

    A generic active roster can include players omitted from a playoff series.
    The bullpen list is an eligibility pool, not a confirmed relief workload.
    """
    resp = requests.get(f"{BASE_URL}/game/{game_pk}/boxscore", timeout=10)
    resp.raise_for_status()
    teams = resp.json().get("teams", {})
    result = {}
    for side in ("home", "away"):
        team = teams.get(side, {})
        players = team.get("players", {})
        ids = dict.fromkeys([*team.get("pitchers", []), *team.get("bullpen", [])])
        roster = set([*ids, *team.get("batters", []), *team.get("bench", [])])
        ordinary_pitchers = sum(players.get(f"ID{pid}", {}).get("position", {}).get("abbreviation") == "P"
                                for pid in ids)
        # Pregame boxscores can retain September's 28-player roster until the
        # series roster is posted. Do not guess which players will be omitted.
        if not 25 <= len(roster) <= 26 or ordinary_pitchers > 13:
            log.warning(f"Game {game_pk} {side}: playoff roster is not confirmed ({len(roster)} players)")
            result[side] = []
            continue
        result[side] = [int(pid) for pid in ids if players.get(f"ID{pid}", {}).get(
            "position", {}).get("abbreviation") in ("P", "TWP")]
    return result


if __name__ == "__main__":
    setup_logging()
    print("=== Today's Schedule ===")
    sched = fetch_schedule()
    if not sched.empty:
        print(sched[["game_pk", "game_date", "away_team", "home_team", "status", "venue"]].to_string(index=False))
        print(f"\nTotal games: {len(sched)}")
    else:
        print("No games today.")

    print("\n=== Probable Starters (next 7 days) ===")
    starters = fetch_probable_starters()
    if not starters.empty:
        print(starters.to_string(index=False))
    else:
        print("No probable starters announced.")

