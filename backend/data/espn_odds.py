"""DraftKings pregame odds from ESPN's public scoreboard feed.

Fallback for when The Odds API monthly quota is exhausted. Selected with
ODDS_SOURCE=espn; unset that variable to return to The Odds API. ESPN carries
one book, so the market consensus is DraftKings alone while this is active.
The live worker also reads the same scoreboard for closing lines (free, one
call covers the slate) when its Odds API budget skips a request.
"""

import logging
from datetime import UTC, date, datetime

import pandas as pd
import requests

from backend.data.odds_api import ODDS_TEAM_MAP

log = logging.getLogger(__name__)

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
# ESPN's provider id is stable while its display name is not ("DraftKings" and
# "Draft Kings" both appear). Map the id to the book key the odds table uses.
ESPN_BOOKS = {"100": "draftkings"}
CLOSING_SOURCE = "espn:draftkings"


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


def _fetch_scoreboard(target_date: date) -> dict:
    resp = requests.get(ESPN_SCOREBOARD, params={"dates": target_date.strftime("%Y%m%d")}, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _pregame_draftkings(payload: dict):
    """Yield (event, {homeAway: team}, DraftKings odds) for every unstarted event."""
    for event in payload.get("events", []):
        # Started games carry live or closing lines, never a pregame offer.
        if event["status"]["type"]["state"] != "pre":
            continue
        competition = event["competitions"][0]
        teams = {
            c["homeAway"]: ODDS_TEAM_MAP.get(c["team"]["displayName"], c["team"]["displayName"])
            for c in competition["competitors"]
        }
        for odds in competition.get("odds") or []:
            provider = odds.get("provider", {})
            if ESPN_BOOKS.get(str(provider.get("id"))) is None:
                log.warning(f"Skipping ESPN odds from unmapped provider {provider.get('name')!r}")
                continue
            yield event, teams, odds


def _close(odds: dict, market: str, side: str) -> dict:
    return odds.get(market, {}).get(side, {}).get("close", {})


def fetch_espn_odds(target_date: date) -> pd.DataFrame:
    """Fetch current pregame odds for one date in the fetch_odds row shape."""
    payload = _fetch_scoreboard(target_date)
    retrieved_at = datetime.now(UTC).isoformat()
    rows = []
    for event, teams, odds in _pregame_draftkings(payload):
        over, under = _close(odds, "total", "over"), _close(odds, "total", "under")
        for side, team in teams.items():
            moneyline, run_line = _close(odds, "moneyline", side), _close(odds, "pointSpread", side)
            rows.append({
                "odds_event_id": event["id"],
                "commence_time": event["date"],
                "team": team,
                "book": "draftkings",
                "moneyline": _price(moneyline),
                "spread": _line(run_line),
                "spread_odds": _price(run_line),
                "total": _line(over),
                "total_over_odds": _price(over),
                "total_under_odds": _price(under),
                "scraped_at": retrieved_at,
            })
    return pd.DataFrame(rows)


def espn_closing_quotes(payload: dict, fetched_at: str) -> list[dict]:
    """One closing-line quote per unstarted event, in the closing_lines row shape.

    A market enters only when both of its sides are quoted (a moneyline needs both
    prices; a run line or total needs its point and both prices); an event with no
    complete market is skipped. observed_at is the fetch time: ESPN does not say
    when DraftKings last moved.
    """
    quotes = []
    for event, teams, odds in _pregame_draftkings(payload):
        home_ml, away_ml = _price(_close(odds, "moneyline", "home")), _price(_close(odds, "moneyline", "away"))
        home_rl, away_rl = _close(odds, "pointSpread", "home"), _close(odds, "pointSpread", "away")
        over, under = _close(odds, "total", "over"), _close(odds, "total", "under")
        quote = {"home_team": teams.get("home"), "away_team": teams.get("away"), "commence_time": event["date"],
                 "source": CLOSING_SOURCE, "fetched_at": fetched_at, "observed_at": fetched_at,
                 "n_offers": 1, "price_book": "draftkings",
                 "home_moneyline": None, "away_moneyline": None,
                 "home_spread": None, "spread_home_price": None, "spread_away_price": None,
                 "total": None, "over_price": None, "under_price": None}
        if home_ml is not None and away_ml is not None:
            quote.update(home_moneyline=float(home_ml), away_moneyline=float(away_ml))
        if _line(home_rl) is not None and _price(home_rl) is not None and _price(away_rl) is not None:
            quote.update(home_spread=_line(home_rl), spread_home_price=float(_price(home_rl)),
                         spread_away_price=float(_price(away_rl)))
        if _line(over) is not None and _price(over) is not None and _price(under) is not None:
            quote.update(total=_line(over), over_price=float(_price(over)), under_price=float(_price(under)))
        if quote["home_moneyline"] is None and quote["home_spread"] is None and quote["total"] is None:
            continue
        quotes.append(quote)
    return quotes


def fetch_espn_closing_quotes(target_date: date) -> list[dict]:
    """One free scoreboard call: the current DraftKings quote for every unstarted game on target_date."""
    payload = _fetch_scoreboard(target_date)
    return espn_closing_quotes(payload, datetime.now(UTC).isoformat())
