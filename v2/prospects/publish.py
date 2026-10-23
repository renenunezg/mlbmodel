"""Publish the three prospect tables: one truncate and one ordered insert each, in one transaction.

Same shape as the player publish: rows travel as one jsonb array per table, land in key order, and
the site's revalidate trigger fires once per statement.
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
    "player_id", "player_name", "org", "org_id", "team", "team_id", "level", "source", "position_group", "secondary_positions",
    "age", "bats", "throws", "headshot_url", "as_of", "model_version", "season", "draft_year", "draft_round", "draft_pick",
    "signing_bonus_musd", "acquired", "peak_grade", "peak_grade_low", "peak_grade_high", "peak_grade_percentile", "grade_reason",
    "p_regular", "p_mlb", "eta_season", "eta_low", "eta_high", "trade_value_musd", "trade_value_low", "trade_value_high",
    "market_value_musd", "expected_control_war", "peak_war", "risk", "rookie_eligible", "on_40_man", "mlb_pa_or_ip",
    "service_days", "injured", "season_sample", "sample", "tracked_share",
]
DETAIL_COLUMNS = ["player_id", "outlook", "tip"]
TRAIT_COLUMNS = [
    "player_id", "trait", "panel", "grade", "grade_low", "grade_high", "percentile", "value", "in_grade", "kind", "sample",
    "inputs", "as_of", "model_version",
]
JSON_COLUMNS = {"outlook", "tip", "inputs"}
ARRAY_COLUMNS = {"secondary_positions"}
INT_COLUMNS = {"player_id", "org_id", "team_id", "season", "draft_year", "draft_pick", "eta_season", "eta_low", "eta_high",
               "service_days", "sample"}
MIN_VALUE_ROWS = 2000
BATCH_ROWS = 2000


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
    return (f"INSERT INTO {table} SELECT * FROM jsonb_populate_recordset(NULL::{table}, CAST(:rows AS jsonb)) "
            f"ORDER BY {order_by}")


def publish(values: pd.DataFrame, traits: pd.DataFrame) -> None:
    if len(values) < MIN_VALUE_ROWS:
        raise ValueError(f"refusing to publish {len(values)} prospect_values rows (under {MIN_VALUE_ROWS})")
    value_rows = _records(values, VALUE_COLUMNS)
    detail_rows = _records(values, DETAIL_COLUMNS)
    trait_rows = _records(traits, TRAIT_COLUMNS)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE prospect_values, prospect_details, prospect_traits"))
        for table, order_by, rows in (("prospect_values", "player_id", value_rows), ("prospect_details", "player_id", detail_rows),
                                      ("prospect_traits", "player_id, trait", trait_rows)):
            # The population is ten times the player table's, so each insert travels in slices the pooler accepts.
            for start in range(0, len(rows), BATCH_ROWS):
                conn.execute(text(_insert(table, order_by)), {"rows": json.dumps(rows[start:start + BATCH_ROWS], default=str)})
    log.info(f"published {len(value_rows)} prospect_values, {len(detail_rows)} prospect_details and {len(trait_rows)} prospect_traits rows")
