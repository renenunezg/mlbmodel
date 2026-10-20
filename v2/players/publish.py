"""Publish the three player tables: one truncate and one ordered insert each, in one transaction.

TRUNCATE gives each table a fresh heap, the insert fills it in primary-key order, and CLUSTER
straightens player_values, so the site's ordered read touches only as many pages as its rows
need. The site's revalidate trigger fires per statement and table (TRUNCATE included), so a
run costs at most six revalidations, and none when scoring skipped the write.

TRUNCATE and CLUSTER hold an exclusive lock until commit, so site reads wait out the publish.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import text

from backend.db import engine

log = logging.getLogger(__name__)

VALUE_COLUMNS = [
    "player_id", "player_name", "team", "team_id", "level", "position_group", "secondary_positions", "age", "bats", "throws",
    "status", "as_of", "model_version", "season", "grade", "grade_low", "grade_high", "grade_percentile", "grade_reason",
    "projected_war", "trade_value_musd", "trade_value_low", "trade_value_high", "market_value_musd", "impact_war", "impact_runs",
    "playing_time", "service_time", "contract_status", "fa_year", "control_years", "salary_current_musd",
    "owed_through_control_musd", "headshot_url",
]
DETAIL_COLUMNS = ["player_id", "outlook", "tip"]
TRAIT_COLUMNS = [
    "player_id", "trait", "panel", "grade", "grade_low", "grade_high", "percentile", "value", "in_grade", "kind", "sample",
    "inputs", "as_of", "model_version",
]
JSON_COLUMNS = {"outlook", "tip", "inputs"}
ARRAY_COLUMNS = {"secondary_positions"}
# Integer columns arrive as floats when a frame has gaps; Postgres rejects "108.0" for an integer.
INT_COLUMNS = {"player_id", "team_id", "age", "season", "fa_year", "control_years", "sample"}


# A publish must look like a population, not a filtered slice.
MIN_VALUE_ROWS = 500


def _records(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    now = datetime.now(UTC).isoformat()
    out = []
    for row in frame[columns].to_dict("records"):
        rec = {"updated_at": now}
        for key, val in row.items():
            if key in JSON_COLUMNS:
                rec[key] = val if val is not None else ([] if key != "tip" else {})
            elif key in ARRAY_COLUMNS:
                rec[key] = list(val) if isinstance(val, (list, tuple)) else []
            elif val is None or (isinstance(val, float) and val != val) or val is pd.NaT:
                rec[key] = None
            elif hasattr(val, "item"):
                rec[key] = val.item()
            else:
                rec[key] = val
            if key in INT_COLUMNS and isinstance(rec[key], float):
                rec[key] = int(rec[key])
        out.append(rec)
    return out


def _insert(table: str, order_by: str) -> str:
    """One INSERT per table: the rows travel as a single jsonb array and land in key order."""
    return (
        f"INSERT INTO {table} SELECT * FROM jsonb_populate_recordset(NULL::{table}, CAST(:rows AS jsonb)) "
        f"ORDER BY {order_by}"
    )


def publish(values: pd.DataFrame, traits: pd.DataFrame) -> None:
    if len(values) < MIN_VALUE_ROWS:
        raise ValueError(f"refusing to publish {len(values)} player_values rows (under {MIN_VALUE_ROWS})")
    value_rows = _records(values, VALUE_COLUMNS)
    detail_rows = _records(values, DETAIL_COLUMNS)
    trait_rows = _records(traits, TRAIT_COLUMNS)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE player_values, player_details, player_traits"))
        conn.execute(text(_insert("player_values", "player_id")), {"rows": json.dumps(value_rows, default=str)})
        conn.execute(text(_insert("player_details", "player_id")), {"rows": json.dumps(detail_rows, default=str)})
        conn.execute(text(_insert("player_traits", "player_id, trait")), {"rows": json.dumps(trait_rows, default=str)})
        # Short rows backfill free space on earlier pages, so an ordered insert alone leaves the
        # heap only mostly in key order; the rewrite puts the site's list read on consecutive pages.
        conn.execute(text("CLUSTER player_values USING player_values_pkey"))
    log.info(
        f"published {len(value_rows)} player_values, {len(detail_rows)} player_details "
        f"and {len(trait_rows)} player_traits rows"
    )
