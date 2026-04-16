"""
The Odds API data fetcher.

Replaces: scrape_odds.py

Fetches betting odds from the-odds-api.com.
Free tier: 500 requests/month. Each call with multiple markets costs ~3 credits.
"""

import os
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"
SPORT = "baseball_mlb"

# Map The Odds API team names to our abbreviations
ODDS_TEAM_MAP = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "ATH",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP",
    "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN",
    # The Odds API sometimes returns just the mascot name
    "Athletics": "ATH",
}


def _get_api_key() -> str:
    key = os.getenv("ODDS_API_KEY")
    if not key:
        raise ValueError(
            "ODDS_API_KEY not set. Get a free key at https://the-odds-api.com "
            "and add it to your .env file."
        )
    return key


def _normalize_team(name: str) -> str:
    return ODDS_TEAM_MAP.get(name, name)


def fetch_odds(books: list[str] = None) -> pd.DataFrame:
    """Fetch current MLB odds from The Odds API.

    Args:
        books: List of bookmaker keys. Defaults to major US books.
               Options: 'draftkings', 'fanduel', 'betmgm', 'pointsbet', 'bet365'

    Returns DataFrame with columns:
        game_id (from odds API), game_pk (to be matched later),
        team, book, moneyline, spread, spread_odds,
        total, total_over_odds, total_under_odds, scraped_at

    Note: The Odds API game IDs are NOT MLB game_pk values.
          You must match games by team + date to link to game_pk.
    """
    api_key = _get_api_key()

    if books is None:
        books = ["draftkings"]

    markets = "h2h,spreads,totals"
    bookmakers_str = ",".join(books)

    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": markets,
        "bookmakers": bookmakers_str,
        "oddsFormat": "american",
    }

    resp = requests.get(f"{ODDS_API_BASE}/{SPORT}/odds", params=params, timeout=15)
    resp.raise_for_status()

    # Log remaining API credits
    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")
    print(f"Odds API credits — used: {used}, remaining: {remaining}")

    events = resp.json()
    rows = []

    for event in events:
        event_id = event["id"]
        home_team = _normalize_team(event["home_team"])
        away_team = _normalize_team(event["away_team"])
        commence = event.get("commence_time", "")

        for bookmaker in event.get("bookmakers", []):
            book_key = bookmaker["key"]

            # Initialize per-team data
            team_data = {
                home_team: {"book": book_key, "moneyline": None, "spread": None,
                            "spread_odds": None, "total": None,
                            "total_over_odds": None, "total_under_odds": None},
                away_team: {"book": book_key, "moneyline": None, "spread": None,
                            "spread_odds": None, "total": None,
                            "total_over_odds": None, "total_under_odds": None},
            }

            for market in bookmaker.get("markets", []):
                market_key = market["key"]

                for outcome in market.get("outcomes", []):
                    team = _normalize_team(outcome.get("name", ""))
                    price = outcome.get("price", 0)
                    point = outcome.get("point")

                    if market_key == "h2h" and team in team_data:
                        team_data[team]["moneyline"] = price

                    elif market_key == "spreads" and team in team_data:
                        team_data[team]["spread"] = point
                        team_data[team]["spread_odds"] = price

                    elif market_key == "totals":
                        side = outcome.get("name", "").lower()
                        # Totals apply to both teams (same line)
                        for t in team_data:
                            team_data[t]["total"] = point
                            if side == "over":
                                team_data[t]["total_over_odds"] = price
                            elif side == "under":
                                team_data[t]["total_under_odds"] = price

            for team, data in team_data.items():
                rows.append({
                    "odds_event_id": event_id,
                    "commence_time": commence,
                    "team": team,
                    **data,
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                })

    df = pd.DataFrame(rows)

    if not df.empty:
        # Parse commence_time to extract game_date for matching
        df["game_date"] = pd.to_datetime(df["commence_time"]).dt.date

    return df


def check_remaining_credits() -> dict:
    """Check remaining API credits. Costs one h2h-only call on The Odds API."""
    api_key = _get_api_key()
    resp = requests.get(
        f"{ODDS_API_BASE}/{SPORT}/odds",
        params={"apiKey": api_key, "regions": "us", "markets": "h2h"},
        timeout=10,
    )
    return {
        "used": resp.headers.get("x-requests-used", "?"),
        "remaining": resp.headers.get("x-requests-remaining", "?"),
    }


if __name__ == "__main__":
    print("=== MLB Odds ===")
    try:
        odds = fetch_odds()
        if not odds.empty:
            display_cols = ["team", "book", "moneyline", "spread", "spread_odds", "total", "game_date"]
            available = [c for c in display_cols if c in odds.columns]
            print(odds[available].head(20).to_string(index=False))
            print(f"\nTotal rows: {len(odds)}")
        else:
            print("No odds data available.")
    except ValueError as e:
        print(f"Configuration error: {e}")
    except requests.RequestException as e:
        print(f"API error: {e}")
