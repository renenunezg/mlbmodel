"""Minor league game feeds from the MLB Stats API, parsed on arrival into compact per-level-season caches.

One live feed per regular-season game (gzip, 30 to 140 KB) yields four row kinds, written as chunk
parquet files under ``cache/prospects/feeds/{sport_id}/{season}/``:

- ``games``: one row per game with venue, teams, parent organizations, league and tracked share
- ``plays``: one row per play (plate appearance or runner event) with the call-code counts of its
  pitches, the stringer batted-ball description, and tracking fields where the last pitch had them
- ``pitches``: one row per pitch, only for games whose pitches carry tracking
- ``appearances``: one row per player per game from the boxscore (positions, batting order, counts)
- ``players``: the feed's bio rows (birth date, hand, primary position), deduplicated at read time

Levels are Stats API sport ids: 11 Triple-A, 12 Double-A, 13 High-A, 14 Single-A, 15 short-season
A (through 2020), 16 Rookie (complex leagues, DSL, and the former advanced rookie leagues), and 17
only for the Arizona Fall League (league 119). A run is resumable: games already cached are skipped,
and ``--minutes`` time-boxes it. The 2005-plus history is built locally and never committed.

Usage:
    python -m v2.prospects.feeds --years 2005 2006 --levels 11 12 13 14 15 16 17
    python -m v2.prospects.feeds --years 2026 --minutes 15
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from backend.log import setup_logging

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "prospects"
FEEDS_DIR = CACHE_DIR / "feeds"
SCHEDULE = "https://statsapi.mlb.com/api/v1/schedule"
FEED = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"
SPORT_IDS = (11, 12, 13, 14, 15, 16, 17)
AFL_LEAGUE_ID = 119
FIRST_SEASON = 2005
WORKERS = 6
REQUESTS_PER_SECOND = 8.0
ATTEMPTS = 4
CHUNK_GAMES = 250
KINDS = ("games", "plays", "pitches", "appearances", "players")
BALL_CODES = {"B", "*B", "I", "V", "P"}
CALLED_STRIKE_CODES = {"C"}
SWINGING_STRIKE_CODES = {"S", "W", "M", "Q"}
FOUL_CODES = {"F", "L", "R", "T", "O"}
IN_PLAY_CODES = {"X", "D", "E"}
MIN_TRACKED_SHARE = 0.8  # a game counts as tracked when this share of its pitches carries a start speed

_session = threading.local()
_throttle_lock = threading.Lock()
_next_slot = [0.0]


def _get(url: str, params: dict | None = None, timeout: int = 60) -> dict:
    session = getattr(_session, "s", None)
    if session is None:
        session = _session.s = requests.Session()
        session.headers["User-Agent"] = "Mozilla/5.0"
    for attempt in range(1, ATTEMPTS + 1):
        with _throttle_lock:
            now = time.monotonic()
            wait = max(0.0, _next_slot[0] - now)
            _next_slot[0] = max(now, _next_slot[0]) + 1.0 / REQUESTS_PER_SECOND
        if wait:
            time.sleep(wait)
        try:
            response = session.get(url, params=params, timeout=timeout)
            if response.status_code == 404:
                raise FileNotFoundError(url)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == ATTEMPTS:
                raise
            log.warning(f"{url}: {exc}; retry {attempt}")
            time.sleep(5.0 * attempt)
    raise RuntimeError("unreachable")


def schedule(sport_id: int, season: int) -> pd.DataFrame:
    """Final regular-season games of one level and season: game_pk, date, league_id, venue_id."""
    payload = _get(SCHEDULE, {"sportId": sport_id, "season": season, "gameType": "R", "hydrate": "team"}, timeout=120)
    rows = []
    for day in payload.get("dates", []):
        for g in day["games"]:
            if g["status"].get("abstractGameState") != "Final" or g["status"].get("codedGameState") in {"C", "D"}:
                continue
            league = g["teams"]["home"]["team"].get("league", {}).get("id")
            if sport_id == 17 and league != AFL_LEAGUE_ID:
                continue
            rows.append({"game_pk": g["gamePk"], "date": g["officialDate"], "league_id": league,
                         "venue_id": g.get("venue", {}).get("id")})
    return pd.DataFrame(rows, columns=["game_pk", "date", "league_id", "venue_id"]).drop_duplicates("game_pk")


def _pitch_row(e: dict, play: dict, game_pk: int) -> dict:
    d = e.get("details", {})
    pdata = e.get("pitchData", {})
    coords = pdata.get("coordinates", {})
    breaks = pdata.get("breaks", {})
    hit = e.get("hitData", {})
    m = play["matchup"]
    return {
        "game_pk": game_pk, "at_bat_index": play["about"]["atBatIndex"], "pitch_number": e.get("pitchNumber"),
        "batter": m["batter"]["id"], "pitcher": m["pitcher"]["id"], "bat_side": m.get("batSide", {}).get("code"),
        "pitch_hand": m.get("pitchHand", {}).get("code"), "code": d.get("code"), "pitch_type": (d.get("type") or {}).get("code"),
        "start_speed": pdata.get("startSpeed"), "pfx_x": coords.get("pfxX"), "pfx_z": coords.get("pfxZ"),
        "plate_x": coords.get("pX"), "plate_z": coords.get("pZ"), "x0": coords.get("x0"), "z0": coords.get("z0"),
        "spin_rate": breaks.get("spinRate"), "ivb": breaks.get("breakVerticalInduced"), "hb": breaks.get("breakHorizontal"),
        "extension": pdata.get("extension"), "zone": pdata.get("zone"), "sz_top": pdata.get("strikeZoneTop"),
        "sz_bot": pdata.get("strikeZoneBottom"), "launch_speed": hit.get("launchSpeed"), "launch_angle": hit.get("launchAngle"),
        "total_distance": hit.get("totalDistance"), "trajectory": hit.get("trajectory"), "hardness": hit.get("hardness"),
    }


def _play_row(play: dict, game_pk: int) -> dict:
    m = play["matchup"]
    r = play["result"]
    pitches = [e for e in play.get("playEvents", []) if e.get("isPitch")]
    codes = [e.get("details", {}).get("code") for e in pitches]
    last = pitches[-1] if pitches else {}
    hit = last.get("hitData", {})
    hc = hit.get("coordinates", {})
    pdata = last.get("pitchData", {})
    return {
        "game_pk": game_pk, "at_bat_index": play["about"]["atBatIndex"], "inning": play["about"]["inning"],
        "top": bool(play["about"]["isTopInning"]), "batter": m["batter"]["id"], "pitcher": m["pitcher"]["id"],
        "bat_side": m.get("batSide", {}).get("code"), "pitch_hand": m.get("pitchHand", {}).get("code"),
        "result_type": r.get("type"), "event_type": r.get("eventType"), "rbi": r.get("rbi"), "is_out": r.get("isOut"),
        "n_pitches": len(pitches),
        "balls": sum(c in BALL_CODES for c in codes), "called_strikes": sum(c in CALLED_STRIKE_CODES for c in codes),
        "swinging_strikes": sum(c in SWINGING_STRIKE_CODES for c in codes), "fouls": sum(c in FOUL_CODES for c in codes),
        "in_play": sum(c in IN_PLAY_CODES for c in codes), "tracked": pdata.get("startSpeed") is not None,
        "pitch_type": ((last.get("details") or {}).get("type") or {}).get("code"), "start_speed": pdata.get("startSpeed"),
        "trajectory": hit.get("trajectory"), "hardness": hit.get("hardness"), "location": hit.get("location"),
        "coord_x": hc.get("coordX"), "coord_y": hc.get("coordY"), "launch_speed": hit.get("launchSpeed"),
        "launch_angle": hit.get("launchAngle"), "total_distance": hit.get("totalDistance"),
    }


def _appearance_rows(box: dict, game_pk: int) -> list[dict]:
    rows = []
    for side in ("home", "away"):
        team = box["teams"][side]
        team_id = team["team"]["id"]
        org_id = team["team"].get("parentOrgId")
        for p in team.get("players", {}).values():
            stats = p.get("stats", {})
            bat = stats.get("batting", {})
            pit = stats.get("pitching", {})
            if not bat and not pit and not p.get("allPositions"):
                continue
            order = p.get("battingOrder")
            rows.append({
                "game_pk": game_pk, "player_id": p["person"]["id"], "team_id": team_id, "org_id": org_id, "home": side == "home",
                "positions": ",".join(pos["abbreviation"] for pos in p.get("allPositions", [])),
                "batting_order": int(order) if order else None, "substitute": bool(p.get("gameStatus", {}).get("isSubstitute")),
                "pa": bat.get("plateAppearances", 0), "ab": bat.get("atBats", 0), "sb": bat.get("stolenBases", 0),
                "cs": bat.get("caughtStealing", 0), "games_started": pit.get("gamesStarted", 0),
                "batters_faced": pit.get("battersFaced", 0), "outs": pit.get("outs", 0), "pitches_thrown": pit.get("numberOfPitches", 0),
                "games_finished": pit.get("gamesFinished", 0), "pitched": bool(pit),
            })
    return rows


def _player_rows(players: dict) -> list[dict]:
    return [{
        "player_id": p["id"], "full_name": p.get("fullName"), "birth_date": p.get("birthDate"),
        "primary_position": p.get("primaryPosition", {}).get("abbreviation"), "bats": p.get("batSide", {}).get("code"),
        "throws": p.get("pitchHand", {}).get("code"), "draft_year": p.get("draftYear"), "mlb_debut": p.get("mlbDebutDate"),
        "birth_country": p.get("birthCountry"), "height": p.get("height"), "weight": p.get("weight"),
    } for p in players.values() if p.get("isPlayer", True)]


def parse_feed(feed: dict, sport_id: int, season: int) -> dict[str, list[dict]]:
    gd, ld = feed["gameData"], feed["liveData"]
    game_pk = gd["game"]["pk"]
    plays = ld["plays"]["allPlays"]
    play_rows = [_play_row(p, game_pk) for p in plays]
    n_pitches = sum(r["n_pitches"] for r in play_rows)
    pitch_events = [(e, p) for p in plays for e in p.get("playEvents", []) if e.get("isPitch")]
    n_tracked = sum(e.get("pitchData", {}).get("startSpeed") is not None for e, _ in pitch_events)
    tracked_share = n_tracked / n_pitches if n_pitches else 0.0
    pitch_rows = [_pitch_row(e, p, game_pk) for e, p in pitch_events] if tracked_share >= MIN_TRACKED_SHARE else []
    box = ld["boxscore"]
    home, away = gd["teams"]["home"], gd["teams"]["away"]
    game = {
        "game_pk": game_pk, "season": season, "sport_id": sport_id, "league_id": home.get("league", {}).get("id"),
        "date": gd["datetime"].get("officialDate"), "venue_id": gd.get("venue", {}).get("id"),
        "home_team_id": home["id"], "away_team_id": away["id"], "home_org_id": home.get("parentOrgId"),
        "away_org_id": away.get("parentOrgId"), "game_type": gd["game"].get("type"), "double_header": gd["game"].get("doubleHeader"),
        "n_plays": len(plays), "n_pitches": n_pitches, "tracked_share": tracked_share,
    }
    return {"games": [game], "plays": play_rows, "pitches": pitch_rows,
            "appearances": _appearance_rows(box, game_pk), "players": _player_rows(gd.get("players", {}))}


def level_dir(sport_id: int, season: int) -> Path:
    return FEEDS_DIR / str(sport_id) / str(season)


def read_kind(sport_id: int, season: int, kind: str) -> pd.DataFrame:
    """All chunk files of one kind for a level-season, concatenated; empty frame without any."""
    files = sorted(level_dir(sport_id, season).glob(f"{kind}-*.parquet"))
    if not files:
        return pd.DataFrame()
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if kind == "players":
        frame = frame.drop_duplicates("player_id", keep="last")
    return frame


def cached_game_pks(sport_id: int, season: int, kind: str = "games") -> set[int]:
    games = read_kind(sport_id, season, kind)
    return set(games["game_pk"].astype(int)) if not games.empty else set()


def _write_chunk(sport_id: int, season: int, rows: dict[str, list[dict]], chunk: int, kinds: tuple[str, ...]) -> None:
    out = level_dir(sport_id, season)
    out.mkdir(parents=True, exist_ok=True)
    for kind in kinds:
        if rows[kind]:
            pd.DataFrame(rows[kind]).to_parquet(out / f"{kind}-{chunk:05d}.parquet", index=False)


def _next_chunk(sport_id: int, season: int, kind: str = "games") -> int:
    files = sorted(level_dir(sport_id, season).glob(f"{kind}-*.parquet"))
    return int(files[-1].stem.split("-")[1]) + 1 if files else 0


def build(sport_id: int, season: int, deadline: float | None = None, workers: int = WORKERS) -> int:
    """Fetch and cache every final regular-season game of a level-season not yet cached; returns games added."""
    kinds = KINDS
    marker = "games"
    sched = schedule(sport_id, season)
    todo = sorted(set(sched["game_pk"].astype(int)) - cached_game_pks(sport_id, season, marker))
    if not todo:
        log.info(f"feeds[{sport_id}/{season}]: complete ({len(sched)} games)")
        return 0
    log.info(f"feeds[{sport_id}/{season}]: {len(todo)} of {len(sched)} games to fetch")
    chunk = _next_chunk(sport_id, season, marker)
    added = 0
    buffer = {k: [] for k in kinds}
    buffered = 0

    def fetch(pk: int) -> dict[str, list[dict]] | None:
        try:
            return parse_feed(_get(FEED.format(pk=pk)), sport_id, season)
        except FileNotFoundError:
            log.warning(f"feeds[{sport_id}/{season}]: game {pk} has no feed")
            return {"games": [{"game_pk": pk, "season": season, "sport_id": sport_id, "n_plays": 0, "n_pitches": 0, "tracked_share": 0.0}],
                    "plays": [], "pitches": [], "appearances": [], "players": []}
        except Exception as exc:  # noqa: BLE001 - one bad game must not end a multi-hour backfill
            log.warning(f"feeds[{sport_id}/{season}]: game {pk} failed: {exc}")
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i in range(0, len(todo), CHUNK_GAMES):
            if deadline is not None and time.monotonic() > deadline:
                log.info(f"feeds[{sport_id}/{season}]: time box reached after {added} games")
                break
            for parsed in pool.map(fetch, todo[i:i + CHUNK_GAMES]):
                if parsed is None:
                    continue
                for k in kinds:
                    buffer[k].extend(parsed[k])
                buffered += 1
            if buffered:
                _write_chunk(sport_id, season, buffer, chunk, kinds)
                chunk += 1
                added += buffered
                log.info(f"feeds[{sport_id}/{season}]: {added}/{len(todo)} games cached")
                buffer = {k: [] for k in kinds}
                buffered = 0
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=[date.today().year])
    parser.add_argument("--levels", type=int, nargs="+", default=list(SPORT_IDS))
    parser.add_argument("--minutes", type=float, default=None, help="stop starting new chunks after this long")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    setup_logging()
    deadline = time.monotonic() + args.minutes * 60 if args.minutes else None
    for year in args.years:
        for sport_id in args.levels:
            if sport_id == 15 and year > 2020:
                continue
            if deadline is not None and time.monotonic() > deadline:
                return
            build(sport_id, year, deadline, args.workers)


if __name__ == "__main__":
    main()
