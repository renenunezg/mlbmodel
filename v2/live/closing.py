"""Capture one closing line per game while the live worker runs.

The worker already polls the slate every 30 seconds around first pitch. While a game is
still unstarted this module holds its latest pregame quote in memory and, at the first
poll where the game is live or past its scheduled start, writes exactly one row to
mlb.closing_lines. Two feeds:

* The Odds API, one request per first-pitch cluster (games starting within ten minutes of
  the cluster's first start) about three minutes before that start, inside a credit budget:
  skipped when the persisted quota would fall below ODDS_API_CLOSING_RESERVE or the slate
  day already spent ODDS_API_CLOSING_DAILY_CALLS requests, which go to the largest clusters.
  One request covers the whole slate; its offers are kept for the cluster's games only.
* ESPN's free scoreboard (DraftKings), at most once a minute while any game is inside the
  thirty minutes before its start. It is the fallback row and the only feed on skipped
  clusters.

Nothing here touches mlb.odds, so the morning quote history stays what it is.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from backend.data.odds_api import ODDS_TEAM_MAP

log = logging.getLogger(__name__)

ESPN_WINDOW = timedelta(minutes=30)
ESPN_INTERVAL = timedelta(seconds=60)
ODDS_API_LEAD = timedelta(minutes=3)
CLUSTER_SPAN = timedelta(minutes=10)
# Matches the database trigger: a row more than ten minutes after first pitch is refused.
WRITE_GRACE = timedelta(minutes=10)
MATCH_TOLERANCE = timedelta(minutes=90)
ODDS_API_REQUEST_CREDITS = 3
SLATE_TZ = ZoneInfo("America/Los_Angeles")
ESPN_TZ = ZoneInfo("America/New_York")
# Statuses that end a game without a first pitch; the held quote is dropped, never written.
INACTIVE_STATES = {"Postponed", "Cancelled", "Canceled", "Suspended", "If Necessary"}

ROW_FIELDS = ("home_spread", "spread_home_price", "spread_away_price", "total", "over_price", "under_price",
              "home_moneyline", "away_moneyline", "n_offers", "price_book", "source", "observed_at", "fetched_at")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return parsed


@dataclass(frozen=True)
class TrackedGame:
    game_pk: int
    season: int
    start: datetime
    home: str
    away: str
    abstract_state: str
    detailed_state: str

    @classmethod
    def from_schedule(cls, game: dict) -> TrackedGame:
        teams = game["teams"]
        return cls(
            game_pk=int(game["gamePk"]), season=int(game["season"]),
            start=datetime.fromisoformat(game["gameDate"].replace("Z", "+00:00")),
            home=ODDS_TEAM_MAP.get(teams["home"]["team"]["name"], teams["home"]["team"]["name"]),
            away=ODDS_TEAM_MAP.get(teams["away"]["team"]["name"], teams["away"]["team"]["name"]),
            abstract_state=game["status"]["abstractGameState"],
            detailed_state=game["status"].get("detailedState", ""),
        )

    @property
    def inactive(self) -> bool:
        return self.detailed_state in INACTIVE_STATES

    def pregame(self, now: datetime) -> bool:
        return self.abstract_state == "Preview" and not self.inactive and now < self.start

    def started(self, now: datetime) -> bool:
        return self.abstract_state in {"Live", "Final"} or now >= self.start

    @property
    def slate_day(self) -> date:
        return self.start.astimezone(SLATE_TZ).date()


def tracked_games(schedule: Iterable[dict]) -> list[TrackedGame]:
    return [TrackedGame.from_schedule(g) for g in schedule]


def match_quotes(games: Iterable[TrackedGame], quotes: Iterable[dict]) -> dict[int, dict]:
    """Attach each provider quote to the tracked game with the same teams and the nearest start."""
    matched: dict[int, tuple[timedelta, dict]] = {}
    for quote in quotes:
        commence = quote.get("commence_time")
        if not commence:
            continue
        commence = datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
        for game in games:
            if (game.home, game.away) != (quote.get("home_team"), quote.get("away_team")):
                continue
            gap = abs(game.start - commence)
            if gap <= MATCH_TOLERANCE and (game.game_pk not in matched or gap < matched[game.game_pk][0]):
                matched[game.game_pk] = (gap, quote)
    return {pk: quote for pk, (_, quote) in matched.items()}


def start_clusters(games: Iterable[TrackedGame]) -> list[tuple[datetime, frozenset[int]]]:
    """Group games starting within CLUSTER_SPAN of a cluster's first start; (first start, game_pks) per cluster."""
    clusters: list[tuple[datetime, set[int]]] = []
    for game in sorted(games, key=lambda g: (g.start, g.game_pk)):
        if clusters and game.start <= clusters[-1][0] + CLUSTER_SPAN:
            clusters[-1][1].add(game.game_pk)
        else:
            clusters.append((game.start, {game.game_pk}))
    return [(start, frozenset(pks)) for start, pks in clusters]


def budgeted_clusters(games: Iterable[TrackedGame], daily_calls: int) -> list[tuple[datetime, frozenset[int]]]:
    """The slate day's clusters that may spend a request: the largest first, earliest on ties."""
    ranked = sorted(start_clusters(games), key=lambda c: (-len(c[1]), c[0]))
    return ranked[:daily_calls]


class ClosingLineCapture:
    """Pure capture state; the fetchers and the writer are injected so the logic needs no network or database."""

    def __init__(self, *, espn_fetch: Callable[[date], list[dict]],
                 odds_api_fetch: Callable[[tuple[int, ...]], tuple[list[dict], str]],
                 quote_from_event: Callable[[dict, str], dict | None],
                 quota_state: Callable[[], dict | None], calls_today: Callable[[date], int],
                 record_call: Callable[[date], None], reserve: int | None = None, daily_calls: int | None = None):
        self._espn_fetch = espn_fetch
        self._odds_api_fetch = odds_api_fetch
        self._quote_from_event = quote_from_event
        self._quota_state = quota_state
        self._calls_today = calls_today
        self._record_call = record_call
        self.reserve = _env_int("ODDS_API_CLOSING_RESERVE", 60) if reserve is None else reserve
        self.daily_calls = _env_int("ODDS_API_CLOSING_DAILY_CALLS", 2) if daily_calls is None else daily_calls
        self.held: dict[int, dict] = {}
        self.written: set[int] = set()
        self.fired: set[datetime] = set()
        self.last_espn_at: datetime | None = None

    def observe(self, games: list[TrackedGame], now: datetime) -> None:
        """Refresh held pregame quotes. Provider failures are logged; the live feed must keep publishing."""
        try:
            self._observe_odds_api(games, now)
        except Exception:
            log.exception("Odds API closing request failed")
        try:
            self._observe_espn(games, now)
        except Exception:
            log.exception("ESPN closing scoreboard failed")

    def _observe_odds_api(self, games: list[TrackedGame], now: datetime) -> None:
        pregame = {g.game_pk: g for g in games if g.pregame(now)}
        if not pregame:
            return
        slate_day = now.astimezone(SLATE_TZ).date()
        todays = [g for g in games if g.slate_day == slate_day and not g.inactive]
        for first_start, pks in budgeted_clusters(todays, self.daily_calls):
            due = first_start - ODDS_API_LEAD <= now < first_start
            waiting = tuple(sorted(pk for pk in pks if pk in pregame))
            if first_start in self.fired or not due or not waiting:
                continue
            self.fired.add(first_start)
            if not self._budget_allows(slate_day):
                continue
            events, fetched_at = self._odds_api_fetch(waiting)
            self._record_call(slate_day)
            quotes = [q for q in (self._quote_from_event(e, fetched_at) for e in events) if q]
            matched = match_quotes([pregame[pk] for pk in waiting], quotes)
            for pk, quote in matched.items():
                self.held[pk] = quote
            log.info("Odds API closing request for the %s cluster (%d credits): %d of %d games quoted",
                     first_start.isoformat(), ODDS_API_REQUEST_CREDITS, len(matched), len(waiting))

    def _budget_allows(self, slate_day: date) -> bool:
        spent = self._calls_today(slate_day)
        if spent >= self.daily_calls:
            log.info("Skipping Odds API closing request: %d of %d daily calls already spent", spent, self.daily_calls)
            return False
        quota = self._quota_state() or {}
        remaining = quota.get("x-requests-remaining")
        if isinstance(remaining, int) and remaining - ODDS_API_REQUEST_CREDITS < self.reserve:
            log.info("Skipping Odds API closing request: remaining=%d, cost=%d, reserve=%d",
                     remaining, ODDS_API_REQUEST_CREDITS, self.reserve)
            return False
        return True

    def _observe_espn(self, games: list[TrackedGame], now: datetime) -> None:
        in_window = [g for g in games if g.pregame(now) and g.start - ESPN_WINDOW <= now]
        if not in_window or (self.last_espn_at is not None and now - self.last_espn_at < ESPN_INTERVAL):
            return
        self.last_espn_at = now
        target_date = min(g.start for g in in_window).astimezone(ESPN_TZ).date()
        matched = match_quotes(in_window, self._espn_fetch(target_date))
        for pk, quote in matched.items():
            # An Odds API quote outranks the ESPN fallback once held.
            if self.held.get(pk, {}).get("source") in (None, quote["source"]):
                self.held[pk] = quote

    def due_rows(self, games: list[TrackedGame], now: datetime) -> list[dict]:
        """closing_lines rows for games that just started and still hold a quote; drops what can never be written."""
        rows = []
        for game in games:
            quote = self.held.get(game.game_pk)
            if quote is None or game.game_pk in self.written:
                continue
            if game.inactive:
                log.info("Dropping the held closing quote for %s: %s", game.game_pk, game.detailed_state)
                del self.held[game.game_pk]
            elif now > game.start + WRITE_GRACE:
                log.warning("Closing line for %s missed its write window; dropped", game.game_pk)
                del self.held[game.game_pk]
            elif game.started(now):
                rows.append({"game_pk": game.game_pk, "season": game.season, "start_time": game.start.isoformat(),
                             **{field: quote[field] for field in ROW_FIELDS}})
        return rows

    def acknowledge(self, game_pks: Iterable[int]) -> None:
        for pk in game_pks:
            self.written.add(pk)
            self.held.pop(pk, None)

    def prune(self, games: list[TrackedGame]) -> None:
        tracked = {g.game_pk for g in games}
        self.held = {pk: q for pk, q in self.held.items() if pk in tracked}
        self.written &= tracked
        self.fired = {start for start in self.fired if start > datetime.now(UTC) - timedelta(days=1)}


def write_closing_lines(rows: list[dict]) -> set[int]:
    """Insert each row once; a refused or failed row is logged and left for the next poll inside the window."""
    from sqlalchemy import text

    from backend.db import engine

    written = set()
    for row in rows:
        try:
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO closing_lines (game_pk, season, start_time, source, observed_at, fetched_at,
                        home_spread, spread_home_price, spread_away_price, total, over_price, under_price,
                        home_moneyline, away_moneyline, n_offers, price_book)
                    VALUES (:game_pk, :season, :start_time, :source, :observed_at, :fetched_at,
                        :home_spread, :spread_home_price, :spread_away_price, :total, :over_price, :under_price,
                        :home_moneyline, :away_moneyline, :n_offers, :price_book)
                    ON CONFLICT (game_pk) DO NOTHING
                """), row)
            written.add(row["game_pk"])
        except Exception:
            log.exception("Could not write the closing line for %s", row["game_pk"])
    return written


def production_capture() -> ClosingLineCapture:
    from backend.data import odds_api
    from backend.data.espn_odds import fetch_espn_closing_quotes

    return ClosingLineCapture(
        espn_fetch=fetch_espn_closing_quotes,
        odds_api_fetch=lambda pks: odds_api.fetch_odds_events(pks, books=None),
        quote_from_event=odds_api.median_closing_quote,
        quota_state=odds_api.latest_quota_state,
        calls_today=odds_api.closing_calls_today,
        record_call=odds_api.record_closing_call,
    )
