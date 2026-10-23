"""Publish the prospect table: one truncate and one ordered insert, in one transaction.

Rows travel to a temporary table in slices the pooler accepts (the population is ten times the
player table's), then one INSERT ... SELECT lands them in key order, so the site's revalidate
trigger fires twice per night: once for the truncate and once for the insert.
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
    "player_id", "player_name", "org", "org_id", "team", "team_id", "level", "position_group",
    "age", "bats", "throws", "headshot_url", "as_of", "model_version", "season", "draft_year", "draft_round", "draft_pick",
    "signing_bonus_musd", "acquired", "peak_grade", "peak_grade_low", "peak_grade_high", "peak_grade_percentile", "grade_reason",
    "p_regular", "p_mlb", "eta_season", "trade_value_musd", "trade_value_low", "trade_value_high",
    "market_value_musd", "expected_control_war", "peak_war", "risk", "on_40_man", "mlb_ab_or_ip",
    "service_days", "season_sample", "sample", "tracked_share", "trait_grades", "trait_pcts", "tip",
    "pitch_mix", "command_grade", "stuff_level", "stuff_pitches",
]
JSON_COLUMNS = {"tip"}
ARRAY_COLUMNS = {"trait_grades", "trait_pcts"}
NULLABLE_ARRAY_COLUMNS = {"pitch_mix"}
INT_COLUMNS = {"player_id", "org_id", "team_id", "season", "draft_year", "draft_pick", "eta_season", "service_days", "sample",
               "command_grade", "stuff_level", "stuff_pitches"}
MIN_VALUE_ROWS = 2000
BATCH_ROWS = 2000


def _records(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    now = datetime.now(UTC).isoformat()
    out = []
    for row in frame[columns].to_dict("records"):
        rec = {"updated_at": now}
        for key, val in row.items():
            if key in JSON_COLUMNS:
                rec[key] = val if val is not None else []
            elif key in ARRAY_COLUMNS:
                rec[key] = list(val) if isinstance(val, (list, tuple)) else []
            elif key in NULLABLE_ARRAY_COLUMNS:
                rec[key] = list(val) if isinstance(val, (list, tuple)) else None
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


def publish(values: pd.DataFrame) -> None:
    if len(values) < MIN_VALUE_ROWS:
        raise ValueError(f"refusing to publish {len(values)} prospect_values rows (under {MIN_VALUE_ROWS})")
    rows = _records(values, VALUE_COLUMNS)
    with engine.begin() as conn:
        conn.execute(text("CREATE TEMP TABLE prospect_values_in (LIKE prospect_values) ON COMMIT DROP"))
        for start in range(0, len(rows), BATCH_ROWS):
            conn.execute(text("INSERT INTO prospect_values_in SELECT * FROM jsonb_populate_recordset(NULL::prospect_values_in, CAST(:rows AS jsonb))"),
                         {"rows": json.dumps(rows[start:start + BATCH_ROWS], default=str)})
        conn.execute(text("TRUNCATE prospect_values"))
        conn.execute(text("INSERT INTO prospect_values SELECT * FROM prospect_values_in ORDER BY player_id"))
    log.info(f"published {len(rows)} prospect_values rows")
