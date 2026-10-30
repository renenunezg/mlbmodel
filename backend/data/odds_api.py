"""Betting odds from the-odds-api.com.

Free tier: 500 requests/month. Each call with multiple markets costs ~3 credits.
"""

import json
import logging
import os
import statistics
import tempfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

log = logging.getLogger(__name__)

load_dotenv()

ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"
SPORT = "baseball_mlb"
DEFAULT_BOOKS = ("draftkings", "fanduel", "betmgm")
DEFAULT_STATE_PATH = Path("cache/odds_api_state.json")
MARKETS = ("h2h", "spreads", "totals")
# Odds are pulled once per day, so a quote stays usable until first pitch.
# Anything older than a day predates today's pull.
MAX_QUOTE_AGE = timedelta(hours=24)

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


def _state_path() -> Path:
    return Path(os.getenv("ODDS_API_STATE_PATH", str(DEFAULT_STATE_PATH)))


def _read_state() -> dict:
    path = _state_path()
    try:
        state = json.loads(path.read_text())
        return state if isinstance(state, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _write_state(state: dict) -> None:
    """Atomically persist the latest quota headers."""
    path = _state_path()
    tmp_name = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as tmp:
            json.dump(state, tmp, indent=2, sort_keys=True)
            tmp.write("\n")
            tmp_name = tmp.name
        os.replace(tmp_name, path)
    except OSError as exc:
        log.warning(f"could not persist Odds API state at {path}: {exc}")
    finally:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass


def _header_value(headers, name: str) -> int | str | None:
    value = headers.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _capture_quota_state(resp: requests.Response, retrieved_at: str) -> dict:
    # The state file also carries the closing-line call counter; keep it.
    state = _read_state()
    state.update({
        "x-requests-last": _header_value(resp.headers, "x-requests-last"),
        "x-requests-used": _header_value(resp.headers, "x-requests-used"),
        "x-requests-remaining": _header_value(resp.headers, "x-requests-remaining"),
        "retrieved_at": retrieved_at,
    })
    _write_state(state)
    return state


def latest_quota_state() -> dict | None:
    """Return the latest persisted normal-response quota headers, if known."""
    quota = _read_state()
    return quota or None


def closing_calls_today(day: date) -> int:
    """Closing-line requests already charged on this slate day, across worker handoffs."""
    calls = _read_state().get("closing_calls") or {}
    return int(calls.get("count", 0)) if calls.get("date") == day.isoformat() else 0


def record_closing_call(day: date) -> None:
    state = _read_state()
    state["closing_calls"] = {"date": day.isoformat(), "count": closing_calls_today(day) + 1}
    _write_state(state)


def fetch_odds_events(upcoming_game_pks: tuple[int, ...], books: list[str] | None) -> tuple[list[dict], str]:
    """One paid request: every upcoming MLB event with its bookmakers' h2h, spreads and totals.

    books=None asks for every US bookmaker; the credit cost (markets x regions) is the same.
    Returns the raw events and the request time as an ISO string.
    """
    # Re-validate at the only paid-request boundary. This makes an accidental
    # direct caller fail closed before requests.get can consume quota.
    if not upcoming_game_pks:
        raise ValueError("Refusing Odds API request without upcoming unstarted MLB games")

    params = {
        "apiKey": _get_api_key(),
        "regions": "us",
        "markets": ",".join(MARKETS),
        "oddsFormat": "american",
    }
    if books:
        params["bookmakers"] = ",".join(books)

    resp = requests.get(f"{ODDS_API_BASE}/{SPORT}/odds", params=params, timeout=15)
    resp.raise_for_status()

    retrieved_at = datetime.now(UTC).isoformat()
    quota = _capture_quota_state(resp, retrieved_at)
    log.info(
        "Odds API quota - "
        f"last: {quota['x-requests-last']}, "
        f"used: {quota['x-requests-used']}, "
        f"remaining: {quota['x-requests-remaining']}, "
        f"retrieved_at: {quota['retrieved_at']}"
    )
    return resp.json(), retrieved_at


def fetch_odds(
    upcoming_game_pks: tuple[int, ...],
    books: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch current MLB odds from The Odds API.

    Args:
        upcoming_game_pks: Non-empty upcoming MLB game IDs proven by the schedule DB.
        books: Bookmaker keys. Defaults to DraftKings, FanDuel, and BetMGM.

    Returns DataFrame with columns:
        game_id (from odds API), game_pk (to be matched later),
        team, book, moneyline, spread, spread_odds,
        total, total_over_odds, total_under_odds, scraped_at

    Note: The Odds API game IDs are NOT MLB game_pk values.
          You must match games by team + start time to link to game_pk.
    """
    if books is None:
        books = list(DEFAULT_BOOKS)
    events, retrieved_at = fetch_odds_events(upcoming_game_pks, books)
    quota = latest_quota_state()
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
                    "scraped_at": retrieved_at,
                })

    df = pd.DataFrame(rows)

    if not df.empty:
        # Parse commence_time to extract game_date for matching
        df["game_date"] = pd.to_datetime(df["commence_time"]).dt.date

    # DataFrame.attrs exposes the headers to normal callers without adding
    # provider metadata columns to the stable odds-table contract.
    df.attrs["quota"] = quota
    return df


CLOSING_SOURCE = "odds_api:median"
PREFERRED_CLOSING_BOOK = "draftkings"


def _market_offers(bookmaker: dict, home: str, away: str) -> dict:
    """One bookmaker's three markets keyed by point, with the newest update time seen."""
    offers = {"h2h": None, "spreads": {}, "totals": {}, "last_update": bookmaker.get("last_update")}
    for market in bookmaker.get("markets", []):
        updated = market.get("last_update") or bookmaker.get("last_update")
        if updated and (offers["last_update"] is None or updated > offers["last_update"]):
            offers["last_update"] = updated
        by_name = {_normalize_team(str(o.get("name", ""))): o for o in market.get("outcomes", [])}
        if market["key"] == "h2h" and home in by_name and away in by_name:
            offers["h2h"] = (by_name[home]["price"], by_name[away]["price"])
        elif market["key"] == "spreads" and home in by_name and away in by_name:
            if by_name[home].get("point") is not None:
                offers["spreads"][float(by_name[home]["point"])] = (by_name[home]["price"], by_name[away]["price"])
        elif market["key"] == "totals":
            over, under = by_name.get("Over"), by_name.get("Under")
            if over and under and over.get("point") is not None:
                offers["totals"][float(over["point"])] = (over["price"], under["price"])
    return offers


def _choose(candidates: dict[str, dict]) -> str | None:
    """DraftKings when it quotes the point, otherwise the most recently updated bookmaker."""
    if not candidates:
        return None
    if PREFERRED_CLOSING_BOOK in candidates:
        return PREFERRED_CLOSING_BOOK
    return max(candidates, key=lambda book: (candidates[book]["last_update"] or "", book))


def median_closing_quote(event: dict, fetched_at: str) -> dict | None:
    """The closing row for one Odds API event: median points across bookmakers, one book's prices at each.

    The run line and total are the median of the points offered (the lower middle of an even count, so the
    point is one a book actually offered). Prices at each point come from DraftKings when it offers that point,
    otherwise from the bookmaker with the latest last_update there; the moneyline pair likewise. observed_at and
    price_book describe the moneyline's bookmaker, falling back to the run line's, then the total's.
    None when no bookmaker quotes any of the three markets.
    """
    home, away = _normalize_team(event["home_team"]), _normalize_team(event["away_team"])
    books = {b["key"]: _market_offers(b, home, away) for b in event.get("bookmakers", [])}
    books = {k: v for k, v in books.items() if v["h2h"] or v["spreads"] or v["totals"]}
    if not books:
        return None
    quote = {"home_team": home, "away_team": away, "commence_time": event.get("commence_time"),
             "source": CLOSING_SOURCE, "fetched_at": fetched_at, "n_offers": len(books),
             "home_spread": None, "spread_home_price": None, "spread_away_price": None,
             "total": None, "over_price": None, "under_price": None,
             "home_moneyline": None, "away_moneyline": None, "price_book": None, "observed_at": None}
    chosen = []
    moneyline = _choose({k: v for k, v in books.items() if v["h2h"]})
    if moneyline:
        quote["home_moneyline"], quote["away_moneyline"] = (float(p) for p in books[moneyline]["h2h"])
        chosen.append(moneyline)
    for market, point_key, price_keys in (("spreads", "home_spread", ("spread_home_price", "spread_away_price")),
                                          ("totals", "total", ("over_price", "under_price"))):
        points = [point for offers in books.values() for point in offers[market]]
        if not points:
            continue
        point = statistics.median_low(points)
        book = _choose({k: v for k, v in books.items() if point in v[market]})
        quote[point_key] = point
        quote[price_keys[0]], quote[price_keys[1]] = (float(p) for p in books[book][market][point])
        chosen.append(book)
    quote["price_book"] = chosen[0]
    quote["observed_at"] = books[chosen[0]]["last_update"] or fetched_at
    return quote
