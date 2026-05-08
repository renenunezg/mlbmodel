"""
MLB Stats API data fetcher.

Replaces: scrape_season_scores.py, scrape_probable_starters.py,
          scrape_team_batting_vs_lhp.py, scrape_team_batting_vs_rhp.py

Uses the MLB-StatsAPI package for clean access to:
- Game schedule and scores (game_pk as universal ID)
- Probable starters with handedness
- Team batting splits (vs LHP / vs RHP)
"""

import statsapi
import pandas as pd
import requests
from datetime import date, timedelta
from backend.team_mappings import normalize_team

BASE_URL = "https://statsapi.mlb.com/api/v1"

# Cache for pitcher handedness lookups (pitcher_id -> 'L'/'R')
_handedness_cache: dict[int, str] = {}


def _fetch_pitcher_handedness(pitcher_id: int) -> str | None:
    """Fetch a pitcher's throwing hand from the MLB People API."""
    if pitcher_id in _handedness_cache:
        return _handedness_cache[pitcher_id]

    try:
        resp = requests.get(f"{BASE_URL}/people/{pitcher_id}", timeout=5)
        resp.raise_for_status()
        person = resp.json().get("people", [{}])[0]
        hand = person.get("pitchHand", {}).get("code")
        if hand:
            _handedness_cache[pitcher_id] = hand
        return hand
    except Exception as e:
        print(f"  Handedness fetch failed for pitcher {pitcher_id}: {e}")
        return None


def _batch_fetch_handedness(pitcher_ids: list[int]) -> dict[int, str]:
    """Fetch handedness for multiple pitchers in one API call."""
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
        print(f"Batch handedness fetch failed: {e}")

    return _handedness_cache


def fetch_schedule(game_date: date = None) -> pd.DataFrame:
    """Fetch the game schedule for a given date.

    Returns DataFrame with columns:
        game_pk, game_date, home_team, away_team, home_score, away_score,
        status, venue, home_pitcher_name, home_pitcher_id, home_pitcher_hand,
        away_pitcher_name, away_pitcher_id, away_pitcher_hand
    """
    if game_date is None:
        game_date = date.today()

    date_str = game_date.strftime("%Y-%m-%d")

    params = {
        "sportId": 1,
        "date": date_str,
        "gameType": "R",  # Regular season only (excludes spring training)
        "hydrate": "probablePitcher(note),venue,team",
    }
    resp = requests.get(f"{BASE_URL}/schedule", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            home = game.get("teams", {}).get("home", {})
            away = game.get("teams", {}).get("away", {})

            home_pitcher = home.get("probablePitcher", {})
            away_pitcher = away.get("probablePitcher", {})

            rows.append({
                "game_pk": game["gamePk"],
                "game_date": date_entry["date"],
                "start_time": game.get("gameDate"),  # ISO 8601 UTC timestamp
                "home_team": home.get("team", {}).get("abbreviation", ""),
                "away_team": away.get("team", {}).get("abbreviation", ""),
                "home_score": home.get("score"),
                "away_score": away.get("score"),
                "status": game.get("status", {}).get("abstractGameState", "Scheduled"),
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

        # Batch fetch handedness for all pitchers
        all_pitcher_ids = (
            df["home_pitcher_id"].dropna().astype(int).tolist() +
            df["away_pitcher_id"].dropna().astype(int).tolist()
        )
        _batch_fetch_handedness(list(set(all_pitcher_ids)))
        df["home_pitcher_hand"] = df["home_pitcher_id"].map(_handedness_cache)
        df["away_pitcher_hand"] = df["away_pitcher_id"].map(_handedness_cache)

    return df


def fetch_schedule_range(start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch schedule for a date range. Useful for loading historical data.

    Includes probable pitcher info when available.
    """
    params = {
        "sportId": 1,
        "startDate": start_date.strftime("%Y-%m-%d"),
        "endDate": end_date.strftime("%Y-%m-%d"),
        "hydrate": "probablePitcher(note),venue,team",
        "gameType": "R",  # Regular season only
    }
    resp = requests.get(f"{BASE_URL}/schedule", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            home = game.get("teams", {}).get("home", {})
            away = game.get("teams", {}).get("away", {})
            home_pitcher = home.get("probablePitcher", {})
            away_pitcher = away.get("probablePitcher", {})

            rows.append({
                "game_pk": game["gamePk"],
                "game_date": date_entry["date"],
                "start_time": game.get("gameDate"),
                "home_team": home.get("team", {}).get("abbreviation", ""),
                "away_team": away.get("team", {}).get("abbreviation", ""),
                "home_score": home.get("score"),
                "away_score": away.get("score"),
                "status": game.get("status", {}).get("abstractGameState", "Scheduled"),
                "venue": game.get("venue", {}).get("name", ""),
                "home_pitcher_name": home_pitcher.get("fullName"),
                "home_pitcher_id": home_pitcher.get("id"),
                "away_pitcher_name": away_pitcher.get("fullName"),
                "away_pitcher_id": away_pitcher.get("id"),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["home_team"] = df["home_team"].apply(normalize_team)
        df["away_team"] = df["away_team"].apply(normalize_team)

        # Batch fetch handedness for all pitchers
        all_pitcher_ids = (
            df["home_pitcher_id"].dropna().astype(int).tolist() +
            df["away_pitcher_id"].dropna().astype(int).tolist()
        )
        if all_pitcher_ids:
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
    }
    resp = requests.get(f"{BASE_URL}/schedule", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            game_pk = game["gamePk"]
            gd = date_entry["date"]

            for side, is_home in [("home", True), ("away", False)]:
                team_data = game.get("teams", {}).get(side, {})
                pitcher = team_data.get("probablePitcher", {})

                if pitcher.get("fullName"):
                    rows.append({
                        "game_pk": game_pk,
                        "game_date": gd,
                        "team": normalize_team(team_data.get("team", {}).get("abbreviation", "")),
                        "pitcher_name": pitcher["fullName"],
                        "pitcher_id": pitcher.get("id"),
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
    """Fetch posted batting orders for a game.

    Reads the boxscore endpoint, returns `{"home": [9 batter_ids], "away": [...]}`.
    Lists are empty if the lineup hasn't posted yet (pre-game) or if the game
    isn't found. Player IDs are returned as ints in slot order 1-9.
    """
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


def fetch_batting_splits(season: int = None, split: str = "vs_rhp") -> pd.DataFrame:
    """Fetch team batting splits from MLB Stats API.

    Args:
        season: MLB season year (defaults to current year)
        split: 'vs_rhp' or 'vs_lhp'

    Returns DataFrame with columns:
        team, season, split, pa, wrc_plus, woba, ops, slg, obp, iso, babip, k_pct, bb_pct
    """
    if season is None:
        season = date.today().year

    sit_code = "vr" if split == "vs_rhp" else "vl"

    params = {
        "stats": "season",
        "group": "hitting",
        "sportIds": 1,
        "season": season,
        "sitCodes": sit_code,
        "fields": (
            "records,teamName,stat,plateAppearances,onBasePercentage,"
            "sluggingPercentage,ops,iso,babip,strikeOuts,baseOnBalls,"
            "atBats,hits,doubles,triples,homeRuns,stolenBases"
        ),
    }

    try:
        stats_data = statsapi.get(
            "teams_stats",
            {
                "stats": "season",
                "group": "hitting",
                "sportIds": 1,
                "season": season,
                "sitCodes": sit_code,
            },
        )
    except Exception as e:
        print(f"  statsapi.get failed, falling back to direct request: {e}")
        resp = requests.get(f"{BASE_URL}/teams/stats", params=params, timeout=15)
        resp.raise_for_status()
        stats_data = resp.json()

    rows = []
    for split_group in stats_data.get("stats", []):
        for team_entry in split_group.get("splits", []):
            stat = team_entry.get("stat", {})
            team_info = team_entry.get("team", {})

            # Compute derived metrics
            pa = int(stat.get("plateAppearances", 0))
            ab = int(stat.get("atBats", 0))
            bb = int(stat.get("baseOnBalls", 0))
            so = int(stat.get("strikeOuts", 0))
            hits = int(stat.get("hits", 0))
            doubles = int(stat.get("doubles", 0))
            triples = int(stat.get("triples", 0))
            hr = int(stat.get("homeRuns", 0))
            singles = hits - doubles - triples - hr

            obp = float(stat.get("obp", 0))
            slg = float(stat.get("slg", 0))
            ops_val = float(stat.get("ops", 0))

            # ISO = SLG - AVG
            avg = hits / ab if ab > 0 else 0
            iso = slg - avg

            # BABIP = (H - HR) / (AB - SO - HR + SF)
            sf = int(stat.get("sacFlies", 0))
            babip_denom = ab - so - hr + sf
            babip = (hits - hr) / babip_denom if babip_denom > 0 else 0

            # K% and BB%
            k_pct = (so / pa * 100) if pa > 0 else 0
            bb_pct = (bb / pa * 100) if pa > 0 else 0

            rows.append({
                "team": normalize_team(team_info.get("abbreviation", team_info.get("name", ""))),
                "season": season,
                "split": split,
                "pa": pa,
                "wrc_plus": None,  # Not directly available from MLB API
                "woba": None,  # Not directly available from MLB API
                "ops": round(ops_val, 3),
                "slg": round(slg, 3),
                "obp": round(obp, 3),
                "iso": round(iso, 3),
                "babip": round(babip, 3),
                "k_pct": round(k_pct, 1),
                "bb_pct": round(bb_pct, 1),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
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

    print("\n=== Team Batting vs RHP ===")
    batting = fetch_batting_splits(split="vs_rhp")
    if not batting.empty:
        print(batting.head(10).to_string(index=False))
    else:
        print("No batting data available (season may not have started).")
