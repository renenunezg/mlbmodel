"""Baseball Savant leaderboard CSVs, one request per board per season, cached.

Every frame comes back with an int64 ``player_id`` column and a ``season``
column so the consumers never care which name the board itself used.
"""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "players"
USER_AGENT = "Mozilla/5.0"
TIMEOUT_SECONDS = 60

_SAVANT = "https://baseballsavant.mlb.com/leaderboard/"
BOARDS: dict[str, str] = {
    "oaa": _SAVANT + "outs_above_average?type=Fielder&startYear={season}&endYear={season}"
    "&split=yes&team=&range=year&min=1&pos={pos}&roles=&viz=hide&csv=true",
    "arm_strength": _SAVANT + "arm-strength?type=player&year={season}&csv=true",
    "catcher_blocking": _SAVANT + "catcher-blocking?game_type=All&n=q&season_end={season}"
    "&season_start={season}&split=no&team=&type=Cat&with_team=0&csv=true",
    "catcher_throwing": _SAVANT + "catcher-throwing?game_type=All&n=q&season_end={season}"
    "&season_start={season}&split=no&team=&type=Cat&with_team=0&csv=true",
    "catcher_framing": _SAVANT + "catcher-framing?year={season}&team=&min=q&csv=true",
    "baserunning": _SAVANT + "baserunning-run-value?game_type=All&season_start={season}"
    "&season_end={season}&csv=true",
    "sprint_speed": _SAVANT + "sprint_speed?year={season}&position=&team=&min=10&csv=true",
    "pitch_arsenal_stats": _SAVANT + "pitch-arsenal-stats?type=pitcher&pitchType=&year={season}"
    "&team=&min=10&csv=true",
    "expected_statistics_batter": _SAVANT + "expected_statistics?type=batter&year={season}"
    "&position=&team=&min=1&csv=true",
    "expected_statistics_pitcher": _SAVANT + "expected_statistics?type=pitcher&year={season}"
    "&position=&team=&min=1&csv=true",
}
# Boards that name the player id column something other than player_id.
_ID_COLUMNS = ("player_id", "id", "pitcher", "entity_id")


def _cache_path(board: str, season: int, pos: str) -> Path:
    suffix = f"_{pos.lower()}" if pos else ""
    return CACHE_DIR / f"savant_{board}_{season}{suffix}.csv"


def _fetch(url: str, attempts: int = 3) -> str:
    for attempt in range(1, attempts + 1):
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        body = response.content.decode("utf-8-sig")
        if not body.lstrip().startswith("<"):
            return body
        if attempt < attempts:
            time.sleep(2.0 * attempt)
    raise RuntimeError(f"Savant returned HTML instead of CSV for {url}")


def _normalize(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    frame = frame.rename(columns=lambda c: c.strip().lstrip("﻿").strip('"'))
    for name in _ID_COLUMNS:
        if name in frame.columns:
            frame = frame.rename(columns={name: "player_id"})
            break
    else:
        raise RuntimeError(f"Savant board has no player id column: {list(frame.columns)[:8]}")
    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="coerce")
    frame = frame.dropna(subset=["player_id"])
    frame["player_id"] = frame["player_id"].astype("int64")
    frame["season"] = season
    return frame


def load_board(board: str, season: int, *, pos: str = "", refresh: bool = False) -> pd.DataFrame:
    """Return one Savant leaderboard for a season, fetching it once and caching the CSV.

    ``pos`` restricts the OAA board to one position (SS, 2B, ...), which is how a
    player who played two positions gets a row for each.
    """
    if board not in BOARDS:
        raise ValueError(f"Unknown Savant board {board!r}; known: {sorted(BOARDS)}")
    path = _cache_path(board, season, pos)
    if path.exists() and not refresh:
        text = path.read_text(encoding="utf-8")
    else:
        text = _fetch(BOARDS[board].format(season=season, pos=pos))
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        log.info(f"savant: cached {board} {season} ({len(text):,} bytes)")
    return _normalize(pd.read_csv(io.StringIO(text)), season)
