"""DraftKings pregame odds from ESPN's public scoreboard feed.

Fallback for when The Odds API monthly quota is exhausted. Selected with
ODDS_SOURCE=espn; unset that variable to return to The Odds API. ESPN carries
one book, so the market consensus is DraftKings alone while this is active.
"""

import logging
from datetime import UTC, date, datetime

import pandas as pd
import requests

from backend.data.odds_api import ODDS_TEAM_MAP

log = logging.getLogger(__name__)

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
# ESPN names the sportsbook; map it to the book keys the odds table already uses.
ESPN_BOOKS = {"DraftKings": "draftkings"}


def _price(quote: dict) -> int | None:
    raw = str(quote.get("odds", "")).strip().upper()
    if raw == "EVEN":
        return 100
    try:
        return int(raw)
    except ValueError:
        return None


def _line(quote: dict) -> float | None:
    # Totals arrive as "o7.5" / "u7.5", run lines as "+1.5" / "-1.5".
    try:
        return float(str(quote.get("line", "")).strip().lstrip("ou"))
    except ValueError:
        return None


def fetch_espn_odds(target_date: date) -> pd.DataFrame:
    """Fetch current pregame odds for one date in the fetch_odds row shape."""
    resp = requests.get(
        ESPN_SCOREBOARD,
        params={"dates": target_date.strftime("%Y%m%d")},
        timeout=15,
    )
    resp.raise_for_status()
    retrieved_at = datetime.now(UTC).isoformat()

    rows = []
    for event in resp.json().get("events", []):
        # Started games carry live or closing lines, never a pregame offer.
        if event["status"]["type"]["state"] != "pre":
            continue
        competition = event["competitions"][0]
        teams = {
            c["homeAway"]: ODDS_TEAM_MAP.get(c["team"]["displayName"], c["team"]["displayName"])
            for c in competition["competitors"]
        }
        for odds in competition.get("odds") or []:
            provider = odds.get("provider", {}).get("name")
            book = ESPN_BOOKS.get(provider)
            if book is None:
                log.warning(f"Skipping ESPN odds from unmapped provider {provider!r}")
                continue
            total = odds.get("total", {})
            over = total.get("over", {}).get("close", {})
            under = total.get("under", {}).get("close", {})
            for side, team in teams.items():
                moneyline = odds.get("moneyline", {}).get(side, {}).get("close", {})
                run_line = odds.get("pointSpread", {}).get(side, {}).get("close", {})
                rows.append({
                    "odds_event_id": event["id"],
                    "commence_time": event["date"],
                    "team": team,
                    "book": book,
                    "moneyline": _price(moneyline),
                    "spread": _line(run_line),
                    "spread_odds": _price(run_line),
                    "total": _line(over),
                    "total_over_odds": _price(over),
                    "total_under_odds": _price(under),
                    "scraped_at": retrieved_at,
                })
    return pd.DataFrame(rows)
