"""Closing-line capture and benchmark: provider parsing, the one-row-per-game window, the de-vig and pairing."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from backend.data.espn_odds import espn_closing_quotes
from backend.data.odds_api import median_closing_quote
from backend.evaluation import _closing_comparison, _market_comparison
from backend.simulation import paired_home_prob
from v2.live.closing import ClosingLineCapture, tracked_games

FIRST_PITCH = datetime(2026, 7, 4, 23, 10, tzinfo=UTC)


def _espn_quote(odds, line=None):
    return {"close": {"odds": odds, **({"line": line} if line else {})}}


def _espn_event(event_id, home, away, date, state="pre", **markets):
    return {"id": event_id, "date": date, "status": {"type": {"state": state}}, "competitions": [{
        "competitors": [{"homeAway": "home", "team": {"displayName": home}},
                        {"homeAway": "away", "team": {"displayName": away}}],
        "odds": [{"provider": {"id": "100", "name": "Draft Kings"}, **markets}],
    }]}


def test_espn_quote_needs_both_sides_of_a_market():
    payload = {"events": [
        _espn_event("1", "Los Angeles Dodgers", "San Diego Padres", "2026-07-04T23:10Z",
                    moneyline={"home": _espn_quote("-150"), "away": _espn_quote("EVEN")},
                    pointSpread={"home": _espn_quote("+120", "-1.5"), "away": _espn_quote("-140", "+1.5")},
                    total={"over": _espn_quote("-110", "o8.5"), "under": _espn_quote("-110", "u8.5")}),
        _espn_event("2", "Boston Red Sox", "New York Yankees", "2026-07-04T23:10Z",
                    moneyline={"home": _espn_quote("-120")},
                    total={"over": _espn_quote("-105", "o9"), "under": _espn_quote("-115", "u9")}),
        _espn_event("3", "Chicago Cubs", "Milwaukee Brewers", "2026-07-04T18:20Z", state="in",
                    moneyline={"home": _espn_quote("-200"), "away": _espn_quote("+170")}),
    ]}
    quotes = espn_closing_quotes(payload, "2026-07-04T22:58:00+00:00")
    assert [(q["home_team"], q["away_team"]) for q in quotes] == [("LAD", "SDP"), ("BOS", "NYY")]
    full, partial = quotes
    assert (full["home_moneyline"], full["away_moneyline"]) == (-150.0, 100.0)
    assert (full["home_spread"], full["spread_home_price"], full["spread_away_price"]) == (-1.5, 120.0, -140.0)
    assert (full["total"], full["over_price"], full["under_price"]) == (8.5, -110.0, -110.0)
    assert full["source"] == "espn:draftkings" and full["observed_at"] == full["fetched_at"]
    # One moneyline side is not a moneyline; the complete total still enters.
    assert partial["home_moneyline"] is None and partial["away_moneyline"] is None
    assert partial["total"] == 9.0


def _book(key, updated, h2h=None, spread=None, total=None, teams=("Los Angeles Dodgers", "San Diego Padres")):
    home, away = teams
    markets = []
    if h2h:
        markets.append({"key": "h2h", "last_update": updated, "outcomes": [
            {"name": home, "price": h2h[0]}, {"name": away, "price": h2h[1]}]})
    if spread:
        markets.append({"key": "spreads", "last_update": updated, "outcomes": [
            {"name": home, "price": spread[1], "point": spread[0]},
            {"name": away, "price": spread[2], "point": -spread[0]}]})
    if total:
        markets.append({"key": "totals", "last_update": updated, "outcomes": [
            {"name": "Over", "price": total[1], "point": total[0]}, {"name": "Under", "price": total[2], "point": total[0]}]})
    return {"key": key, "last_update": updated, "markets": markets}


def test_odds_api_median_points_take_draftkings_or_the_freshest_book():
    event = {"home_team": "Los Angeles Dodgers", "away_team": "San Diego Padres",
             "commence_time": "2026-07-04T23:10:00Z", "bookmakers": [
                 _book("draftkings", "2026-07-04T23:05:00Z", h2h=(-150, 130), spread=(-2.5, 150, -175), total=(8.5, -110, -110)),
                 _book("fanduel", "2026-07-04T23:01:00Z", h2h=(-148, 126), spread=(-1.5, 110, -130), total=(8.5, -108, -112)),
                 _book("betmgm", "2026-07-04T23:06:00Z", h2h=(-155, 130), spread=(-1.5, 115, -135), total=(9.0, -105, -115)),
             ]}
    quote = median_closing_quote(event, "2026-07-04T23:07:00+00:00")
    assert quote["source"] == "odds_api:median" and quote["n_offers"] == 3
    assert (quote["home_moneyline"], quote["away_moneyline"]) == (-150.0, 130.0)
    # DraftKings does not offer the median run line, so the freshest book at -1.5 prices it.
    assert (quote["home_spread"], quote["spread_home_price"], quote["spread_away_price"]) == (-1.5, 115.0, -135.0)
    assert (quote["total"], quote["over_price"], quote["under_price"]) == (8.5, -110.0, -110.0)
    assert quote["price_book"] == "draftkings" and quote["observed_at"] == "2026-07-04T23:05:00Z"
    assert median_closing_quote({**event, "bookmakers": []}, "x") is None


def _schedule(game_pk, start, home, away, state="Preview", detailed="Pre-Game"):
    return {"gamePk": game_pk, "season": "2026", "gameDate": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": {"abstractGameState": state, "detailedState": detailed},
            "teams": {"home": {"team": {"name": home}}, "away": {"team": {"name": away}}}}


def _odds_event(home, away, start, home_ml):
    return {"home_team": home, "away_team": away, "commence_time": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "bookmakers": [_book("draftkings", "2026-07-04T23:06:00Z", h2h=(home_ml, 100 - home_ml), teams=(home, away))]}


def test_capture_writes_one_row_per_game_from_the_budgeted_feed():
    espn_calls, odds_calls, recorded = [], [], []
    slate = [
        _schedule(1, FIRST_PITCH, "Los Angeles Dodgers", "San Diego Padres"),
        _schedule(2, FIRST_PITCH + timedelta(minutes=5), "Boston Red Sox", "New York Yankees"),
        _schedule(3, FIRST_PITCH + timedelta(hours=3), "Seattle Mariners", "Houston Astros"),
        _schedule(4, FIRST_PITCH - timedelta(hours=6), "Chicago Cubs", "Milwaukee Brewers", "Final", "Final"),
    ]

    def espn_fetch(target_date):
        espn_calls.append(target_date)
        return espn_closing_quotes({"events": [
            _espn_event("a", "Los Angeles Dodgers", "San Diego Padres", "2026-07-04T23:10Z",
                        moneyline={"home": _espn_quote("-140"), "away": _espn_quote("+120")}),
            _espn_event("b", "Boston Red Sox", "New York Yankees", "2026-07-04T23:15Z",
                        moneyline={"home": _espn_quote("-110"), "away": _espn_quote("-110")}),
            _espn_event("c", "Seattle Mariners", "Houston Astros", "2026-07-05T02:10Z",
                        moneyline={"home": _espn_quote("-105"), "away": _espn_quote("-115")}),
        ]}, "espn-fetch")

    def odds_fetch(pks):
        odds_calls.append(pks)
        return [_odds_event("Los Angeles Dodgers", "San Diego Padres", FIRST_PITCH, -150),
                _odds_event("Boston Red Sox", "New York Yankees", FIRST_PITCH + timedelta(minutes=5), -120),
                _odds_event("Seattle Mariners", "Houston Astros", FIRST_PITCH + timedelta(hours=3), -130)], "odds-fetch"

    spent = {"count": 0}
    capture = ClosingLineCapture(
        espn_fetch=espn_fetch, odds_api_fetch=odds_fetch, quote_from_event=median_closing_quote,
        quota_state=lambda: {"x-requests-remaining": 100}, calls_today=lambda day: spent["count"],
        record_call=lambda day: recorded.append(day) or spent.__setitem__("count", spent["count"] + 1),
        reserve=60, daily_calls=1)
    games = tracked_games(slate)

    # Forty minutes out nothing is inside the ESPN window; at thirty the slate is fetched once a minute.
    capture.observe(games, FIRST_PITCH - timedelta(minutes=40))
    assert espn_calls == [] and capture.held == {}
    capture.observe(games, FIRST_PITCH - timedelta(minutes=25))
    capture.observe(games, FIRST_PITCH - timedelta(minutes=24, seconds=30))
    assert len(espn_calls) == 1 and set(capture.held) == {1, 2}
    assert capture.held[1]["source"] == "espn:draftkings"
    assert capture.due_rows(games, FIRST_PITCH - timedelta(minutes=24)) == []

    # The 23:10 cluster (two games) is the largest, so it spends the day's one request at T-3.
    capture.observe(games, FIRST_PITCH - timedelta(minutes=4))
    assert odds_calls == []
    capture.observe(games, FIRST_PITCH - timedelta(minutes=3))
    capture.observe(games, FIRST_PITCH - timedelta(minutes=2, seconds=30))
    assert odds_calls == [(1, 2)] and len(recorded) == 1
    assert capture.held[1]["source"] == capture.held[2]["source"] == "odds_api:median"
    assert capture.held[1]["home_moneyline"] == -150.0

    # A later ESPN refresh does not displace the Odds API quote.
    capture.observe(games, FIRST_PITCH - timedelta(minutes=1))
    assert capture.held[1]["source"] == "odds_api:median"

    # First poll at or past first pitch writes game 1 only; game 2 starts five minutes later.
    rows = capture.due_rows(games, FIRST_PITCH + timedelta(seconds=20))
    assert [r["game_pk"] for r in rows] == [1]
    assert rows[0]["source"] == "odds_api:median" and rows[0]["start_time"] == FIRST_PITCH.isoformat()
    assert rows[0]["season"] == 2026 and rows[0]["fetched_at"] == "odds-fetch"
    capture.acknowledge([1])
    assert capture.due_rows(games, FIRST_PITCH + timedelta(minutes=1)) == []
    live = tracked_games([{**slate[1], "status": {"abstractGameState": "Live", "detailedState": "In Progress"}}, *slate[2:]])
    rows = capture.due_rows(live, FIRST_PITCH + timedelta(minutes=5, seconds=10))
    assert [r["game_pk"] for r in rows] == [2]
    capture.acknowledge([2])

    # The 02:10 game's cluster is over budget, so its row comes from ESPN.
    late = FIRST_PITCH + timedelta(hours=3)
    capture.observe(live, late - timedelta(minutes=10))
    capture.observe(live, late - timedelta(minutes=3))
    assert len(odds_calls) == 1
    rows = capture.due_rows(live, late + timedelta(seconds=5))
    assert [(r["game_pk"], r["source"], r["home_moneyline"]) for r in rows] == [(3, "espn:draftkings", -105.0)]

    # A quote the worker could not write inside the trigger's ten minutes is dropped, never written late.
    capture.held[3] = rows[0]
    assert capture.due_rows(live, late + timedelta(minutes=11)) == [] and 3 not in capture.held


def test_closing_benchmark_pairs_only_games_with_a_captured_close():
    window = pd.DataFrame([
        (1, "LAD", "LAD", 0.60, 1, -150.0), (1, "SDP", "LAD", 0.40, 0, 135.0),
        (2, "BOS", "BOS", 0.55, 0, -120.0), (2, "NYY", "BOS", 0.45, 1, 110.0),
        (3, "SEA", "SEA", 0.50, 1, np.nan), (3, "HOU", "SEA", 0.50, 0, np.nan),
    ], columns=["game_pk", "team", "home_team", "win_prob", "actual_win", "moneyline"])
    closing = pd.DataFrame([
        (1, "odds_api:median", -160.0, 140.0),
        (3, "espn:draftkings", -105.0, -115.0),
        (9, "espn:draftkings", -200.0, 170.0),
    ], columns=["game_pk", "source", "home_moneyline", "away_moneyline"])

    assert paired_home_prob(-150, 130) == pytest.approx(0.6 / (0.6 + 100 / 230))
    morning = _market_comparison(window)
    assert morning["paired_games"] == 2
    expected = np.mean([(paired_home_prob(-150, 135) - 1) ** 2, (paired_home_prob(-120, 110) - 0) ** 2])
    assert morning["market_brier"] == pytest.approx(expected)

    close = _closing_comparison(window, closing)
    assert close["closing_paired_games"] == 2
    assert close["closing_games_by_source"] == {"odds_api:median": 1, "espn:draftkings": 1}
    expected = np.mean([(paired_home_prob(-160, 140) - 1) ** 2, (paired_home_prob(-105, -115) - 1) ** 2])
    assert close["closing_market_brier"] == pytest.approx(expected)
    assert close["closing_paired_brier"] == pytest.approx(np.mean([0.4 ** 2, 0.5 ** 2]))
    assert _closing_comparison(window, closing.iloc[:0]) == {"closing_paired_games": 0, "closing_games_by_source": {}}
