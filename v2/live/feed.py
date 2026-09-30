"""Read one coherent MLB feed snapshot; never modify pregame predictions."""
from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime

import requests

from v2.live.model import GameState, WinExpectancy

POSTSEASON = {"F", "D", "L", "W"}


def fetch_feed(game_pk: int) -> dict:
    if isinstance(game_pk, bool) or not isinstance(game_pk, int) or game_pk <= 0:
        raise ValueError("A positive MLB game ID is required")
    response = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live", timeout=15)
    response.raise_for_status()
    feed = response.json()
    if feed.get("gamePk") != game_pk:
        raise ValueError("Feed game ID does not match requested game")
    return feed


def _integer(value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Missing or invalid feed state")
    return value


def _after_play(inning, top, outs, bases, home, away, automatic_runner):
    if inning >= 9:
        if not top and home > away:
            return None, 1.
        if outs == 3 and ((top and home > away) or (not top and home != away)):
            return None, float(home > away)
    if outs == 3:
        inning += int(not top)
        top = not top
        bases = 2 if automatic_runner and inning >= 10 else 0
        outs = 0
    return GameState(inning, top, outs, bases, home, away, automatic_runner=automatic_runner), None


def score_feed(feed: dict, model: WinExpectancy, *, fetched_at: datetime | None = None) -> dict:
    fetched_at = fetched_at or datetime.now(UTC)
    game = feed.get("gameData", {})
    status = game.get("status", {})
    live = feed.get("liveData", {})
    ls = live.get("linescore", {})
    teams = game.get("teams", {})
    kind = game.get("game", {}).get("type")
    result = {"game_pk": feed.get("gamePk"), "status": status.get("detailedState"),
              "abstract_state": status.get("abstractGameState"),
              "home_team": teams.get("home", {}).get("abbreviation"),
              "away_team": teams.get("away", {}).get("abbreviation"),
              "home_name": teams.get("home", {}).get("name"),
              "away_name": teams.get("away", {}).get("name"),
              "fetched_at": fetched_at.isoformat(), "source_timestamp": None,
              "model": model.metadata, "home_win_probability": None,
              "away_win_probability": None, "state": None, "history": []}
    try:
        stamp = datetime.strptime(feed["metaData"]["timeStamp"], "%Y%m%d_%H%M%S").replace(tzinfo=UTC)
        result["source_timestamp"] = stamp.isoformat()
        result["source_age_seconds"] = max(0, (fetched_at - stamp).total_seconds())
    except (KeyError, ValueError, TypeError):
        result["unavailable_reason"] = "Missing source timestamp"
        return result
    if kind not in {"R", *POSTSEASON} or ls.get("scheduledInnings") != 9:
        result["unavailable_reason"] = "Unsupported game rules"
        return result
    automatic_runner = kind == "R"
    result["automatic_runner"] = automatic_runner
    if status.get("abstractGameState") not in {"Live", "Final"}:
        result["unavailable_reason"] = "Game has not started"
        return result
    try:
        home = _integer(ls["teams"]["home"]["runs"])
        away = _integer(ls["teams"]["away"]["runs"])
        result.update(home_score=home, away_score=away)
        if status.get("abstractGameState") == "Final":
            if home == away:
                raise ValueError("Tied final has no binary winner")
            probability = float(home > away)
        else:
            inning = _integer(ls["currentInning"])
            phase = ls["inningState"]
            if phase not in {"Top", "Bottom", "Middle", "End"}:
                raise ValueError("Unsupported inning phase")
            top = phase in {"Top", "Middle"}
            outs = _integer(ls["outs"])
            offense = ls["offense"]
            bases = sum((1 << i) for i, base in enumerate(("first", "second", "third"))
                        if offense.get(base, {}).get("id"))
            if phase in {"Middle", "End"} and outs != 3:
                raise ValueError("Inconsistent inning transition")
            state, probability = _after_play(inning, top, outs, bases, home, away, automatic_runner)
            if state is not None:
                balls, strikes = _integer(ls["balls"]), _integer(ls["strikes"])
                current = live.get("plays", {}).get("currentPlay", {})
                if outs == 3 or current.get("about", {}).get("isComplete"):
                    balls = strikes = 0
                state = GameState(**{**asdict(state), "balls": balls, "strikes": strikes})
                probability = model.predict(state)
                result["state"] = asdict(state)
            result["batter"] = offense.get("batter", {}).get("fullName")
            result["pitcher"] = ls.get("defense", {}).get("pitcher", {}).get("fullName")
        result["home_win_probability"] = probability
        result["away_win_probability"] = 1 - probability
        result["stale"] = status.get("abstractGameState") == "Live" and result["source_age_seconds"] > 120
    except (KeyError, ValueError, TypeError) as exc:
        result["unavailable_reason"] = str(exc)
        return result
    result["history"], result["history_omitted"] = _history(feed, model, automatic_runner)
    if (status.get("abstractGameState") == "Final"
            and result["history"][-1]["home_win_probability"] != result["home_win_probability"]):
        # Officially called/shortened games can finish before the normal ninth-inning boundary.
        result["history"].append({"id": "final", "inning": ls.get("currentInning", 9),
                                  "top": ls.get("isTopInning", False), "label": "Final",
                                  "description": "Official final score", "home_score": home, "away_score": away,
                                  "home_win_probability": result["home_win_probability"]})
    return result


def _history(feed, model, automatic_runner):
    start = GameState(1, True, 0, 0, 0, 0, automatic_runner=automatic_runner)
    points = [{"id": "start", "inning": 1, "top": True, "label": "Start",
               "description": "League-average game-state baseline", "home_score": 0, "away_score": 0,
               "home_win_probability": model.predict(start)}]
    omitted = 0
    for play in feed.get("liveData", {}).get("plays", {}).get("allPlays", []):
        about = play.get("about", {})
        if not about.get("isComplete"):
            continue
        try:
            inning, outs = _integer(about["inning"]), _integer(play["count"]["outs"])
            top = about["isTopInning"]
            if not isinstance(top, bool):
                raise ValueError("Invalid half inning")
            home, away = _integer(play["result"]["homeScore"]), _integer(play["result"]["awayScore"])
            matchup = play["matchup"]
            bases = sum(1 << i for i, base in enumerate(("postOnFirst", "postOnSecond", "postOnThird"))
                        if matchup.get(base, {}).get("id"))
            state, probability = _after_play(inning, top, outs, bases, home, away, automatic_runner)
            if state is not None:
                probability = model.predict(state)
            points.append({"id": str(about["atBatIndex"]), "inning": inning, "top": top,
                           "label": f"{'T' if top else 'B'}{inning}",
                           "description": play["result"].get("description", "Completed play"),
                           "home_score": home, "away_score": away,
                           "home_win_probability": probability,
                           "event_at": about.get("endTime")})
        except (KeyError, ValueError, TypeError):
            omitted += 1
    return points, omitted
