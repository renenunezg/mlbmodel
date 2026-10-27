"""Write the carry study row, the MLB hitter table and the minor-league table: one transaction, truncate then ordered
insert, as the prospect board does."""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sqlalchemy import text

from backend.db import engine

log = logging.getLogger(__name__)

HITTER_COLUMNS = ["player_id", "season", "player_name", "team", "bats", "pa", "ab", "h", "tb", "hr", "bb", "hbp", "sf", "so",
                  "swings", "whiffs", "n_air", "carry", "carry_raw", "physics", "backspin", "sidespin", "tilt", "attack",
                  "bat_speed", "lab_carry"]
MINOR_COLUMNS = ["player_id", "season", "level", "player_name", "team", "pa", "hr", "n_air", "carry", "carry_raw", "exit_velo"]
MIN_HITTER_ROWS = 600
MIN_MINOR_ROWS = 600


def _clean(value):
    if isinstance(value, (list, np.ndarray)):
        return [int(v) for v in value]
    if value is None or (isinstance(value, float) and value != value):
        return None
    return value.item() if hasattr(value, "item") else value


def _replace(conn, table: str, rows: list[dict], order: str) -> None:
    conn.execute(text(f"CREATE TEMP TABLE {table}_in (LIKE {table} INCLUDING DEFAULTS) ON COMMIT DROP"))
    conn.execute(text(f"INSERT INTO {table}_in SELECT * FROM jsonb_populate_recordset(NULL::{table}_in, CAST(:rows AS jsonb))"),
                 {"rows": json.dumps(rows)})
    conn.execute(text(f"TRUNCATE {table}"))
    conn.execute(text(f"INSERT INTO {table} SELECT * FROM {table}_in ORDER BY {order}"))


def publish(season: int, model_version: str, payload: dict, hitters: pd.DataFrame, minors: pd.DataFrame) -> None:
    if len(hitters) < MIN_HITTER_ROWS:
        raise ValueError(f"refusing to publish {len(hitters)} carry_hitters rows (under {MIN_HITTER_ROWS})")
    if len(minors) < MIN_MINOR_ROWS:
        raise ValueError(f"refusing to publish {len(minors)} carry_minors rows (under {MIN_MINOR_ROWS})")
    # jsonb_populate_recordset does not apply column defaults, so every row carries its timestamp.
    now = datetime.now(UTC).isoformat()
    rows = [{**{k: _clean(v) for k, v in rec.items()}, "updated_at": now} for rec in hitters[HITTER_COLUMNS].to_dict("records")]
    minor_rows = [{**{k: _clean(v) for k, v in rec.items()}, "updated_at": now} for rec in minors[MINOR_COLUMNS].to_dict("records")]
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM carry_study WHERE season = :s"), {"s": season})
        conn.execute(text("INSERT INTO carry_study (season, model_version, payload) VALUES (:s, :v, CAST(:p AS jsonb))"),
                     {"s": season, "v": model_version, "p": json.dumps(payload)})
        _replace(conn, "carry_hitters", rows, "season, player_id")
        _replace(conn, "carry_minors", minor_rows, "level, season, player_id")
    log.info(f"published carry study {season} ({model_version}), {len(rows)} hitter rows and {len(minor_rows)} minor-league rows")
