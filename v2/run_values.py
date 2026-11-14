"""Refresh pitch and batting run value for the current season, on the player grades' schedule (players.yml).

Usage:
    python -m v2.run_values [--dry-run]

The current season is the latest one with regular-season pitches in the Statcast cache. Pitch value is refitted
whole (about a minute). Batting value reuses the hitter contact and pitch difficulty models of its last fit
(``cache/batting_value/league_models.joblib``) and fits them again only when they are missing or stop before the
last completed season, so an in-season run applies models fitted through the previous season and the first run
after a season ends refits them through it. Both tables are checked as a full run checks them (the batting parts
must sum and the league rates must match their zone, count and pitch-group averages), so a bad run stops here.

Nothing is published when the inputs are unchanged since the last publish (the offseason), or while the current
season is below a publisher's row floor (early in a season; the previous season stays up). The study rows keep
their validation, which only a full local run with ``validate --confirm`` writes; their season and coverage are
updated. ``--dry-run`` writes the local outputs and publishes nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from backend.data.game_types import regular_season_pitches
from backend.log import setup_logging
from v2.batting_value import publish as batting_publish
from v2.batting_value import run as batting_run
from v2.pitch_value import publish as pitch_publish
from v2.pitch_value import run as pitch_run
from v2.pitch_value.pitches import STATCAST_DIR
from v2.pitch_value.validate import LINES_FINAL
from v2.players import savant

log = logging.getLogger(__name__)

FIRST_SEASON = pitch_run.DEFAULT_SEASONS[0]
STATE = STATCAST_DIR / "run_values_state.json"
# The current season's Savant boards the batting tables read (names and sprint speed); refreshed every run.
BOARDS = ("expected_statistics_batter", "sprint_speed")
SOURCES = [Path(__file__), *sorted((Path(__file__).parent / "pitch_value").glob("*.py")),
           *sorted((Path(__file__).parent / "batting_value").glob("*.py"))]


def current_season(today: date) -> int:
    """The latest season with regular-season pitches in the Statcast cache."""
    for season in range(today.year, FIRST_SEASON - 1, -1):
        path = STATCAST_DIR / f"statcast_{season}.parquet"
        if path.exists() and len(regular_season_pitches(pd.read_parquet(path, columns=["game_type", "game_date"]), season)):
            return season
    raise FileNotFoundError(f"no Statcast cache with regular-season pitches from {FIRST_SEASON} on")


def completed_season(season: int, today: date) -> int:
    return season if today >= date(season, *LINES_FINAL) else season - 1


def fingerprint(season: int, models: dict) -> str:
    digest = hashlib.sha256(str(models["through"]).encode())
    paths = [STATCAST_DIR / f"statcast_{season}.parquet", *(savant.CACHE_DIR / f"savant_{b}_{season}.csv" for b in BOARDS), *SOURCES]
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def publish_pitch_value() -> None:
    seasons = pd.read_parquet(pitch_run.OUT_DIR / "pitcher_seasons.parquet")
    pitchers = pitch_publish.pitcher_rows(seasons)
    if len(pitchers) < pitch_publish.MIN_PITCHER_ROWS:
        log.info(f"pitch value: {len(pitchers)} pitchers so far, under {pitch_publish.MIN_PITCHER_ROWS}; not published yet")
        return
    types = pitch_publish.type_rows(pd.read_parquet(pitch_run.OUT_DIR / "pitch_types.parquet"), pitchers)
    pitch_publish.publish(pitch_publish.coverage(seasons), pitchers, types, keep_study=True)


def publish_batting_value() -> None:
    seasons = pd.read_parquet(batting_run.OUT_DIR / "hitter_seasons.parquet")
    hitters = batting_publish.hitter_rows(seasons)
    if len(hitters) < batting_publish.MIN_HITTER_ROWS:
        log.info(f"batting value: {len(hitters)} hitters so far, under {batting_publish.MIN_HITTER_ROWS}; not published yet")
        return
    splits = batting_publish.split_rows({name: pd.read_parquet(batting_run.OUT_DIR / f"{name}.parquet") for name in
                                         ("situations", "pitch_groups", "pitch_types", "zones", "sides", "heights")}, hitters)
    batting_publish.publish(batting_publish.coverage(seasons), hitters, splits, keep_study=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="write the local outputs and publish nothing")
    args = parser.parse_args()
    setup_logging()
    today = date.today()
    season = current_season(today)
    completed = completed_season(season, today)
    seasons = list(range(FIRST_SEASON, season + 1))
    for board in BOARDS:
        savant.load_board(board, season, refresh=True)

    path = batting_run.OUT_DIR / batting_run.MODELS_FILE
    models = batting_run.load_models() if path.exists() else None
    if models is not None and models["through"] < completed:
        log.info(f"batting value models stop at {models['through']}; refitting through {completed}")
        models = None
    if models is not None:
        state = json.loads(STATE.read_text()) if STATE.exists() else {}
        if not args.dry_run and state.get("fingerprint") == fingerprint(season, models):
            log.info(f"inputs unchanged since {state.get('published')}; nothing to refresh")
            return

    pitch_run.write(pitch_run.build(seasons))
    outputs = batting_run.build(seasons, models, through=completed)
    batting_run.write(outputs)
    models = outputs["models"] or models
    if args.dry_run:
        log.info(f"dry run: {season} refreshed locally (batting models through {models['through']}), nothing published")
        return
    publish_pitch_value()
    publish_batting_value()
    STATE.write_text(json.dumps({"fingerprint": fingerprint(season, models), "published": str(today), "season": season}))


if __name__ == "__main__":
    main()
