"""Write the pitch run value tables the site reads: the study row, pitcher-seasons and pitch types.

Usage:
    python -m v2.pitch_value.run
    python -m v2.pitch_value.validate --confirm
    MLBMODEL_DB_WRITES=1 python -m v2.pitch_value.publish

One transaction: the tables are replaced whole, as the carry tables are. Only the latest season is written,
because that is all the site reads; the study row still describes every season the tables were fitted on.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import text

from backend.db import engine
from backend.log import setup_logging
from backend.team_mappings import TEAM_ID_BY_CODE, normalize_team
from v2.pitch_value.run import OUT_DIR
from v2.pitch_value.validate import BASELINES, HEADLINE

log = logging.getLogger(__name__)

MODEL_VERSION = "pitch-value-v1"
# The smallest samples the site lists. Display choices, not fitted: the site offers these and two larger cuts.
MIN_PITCHES = 500
MIN_TYPE_PITCHES = 200
# A completed season has about 470 pitchers at MIN_PITCHES; far fewer means a partial season or a broken input.
MIN_PITCHER_ROWS = 300
PITCHER_COLUMNS = ["player_id", "season", "player_name", "team", "pitches", "raw_runs", "context_runs", "runs"]
TYPE_COLUMNS = ["player_id", "season", "pitch_type", "pitches", "usage", "runs"]


def display_name(name: pd.Series) -> pd.Series:
    """Statcast writes 'Last, First'."""
    parts = name.str.split(", ", n=1, expand=True)
    return (parts[1] + " " + parts[0]).where(parts[1].notna(), name)


def pitcher_rows(pitchers: pd.DataFrame) -> pd.DataFrame:
    """The latest season's pitchers with enough pitches."""
    kept = pitchers[(pitchers["season"] == pitchers["season"].max()) & (pitchers["pitches"] >= MIN_PITCHES)]
    team = kept["team"].map(normalize_team)
    unknown = sorted(set(team) - set(TEAM_ID_BY_CODE))
    if unknown:
        raise ValueError(f"pitch value rows carry unknown team codes: {unknown}")
    return pd.DataFrame({
        "player_id": kept["pitcher"], "season": kept["season"], "player_name": display_name(kept["name"]), "team": team,
        "pitches": kept["pitches"], "raw_runs": kept["raw_runs"].round(2), "context_runs": kept["runs"].round(2),
        "runs": kept["neutral_runs"].round(2),
    })


def type_rows(types: pd.DataFrame, pitchers: pd.DataFrame) -> pd.DataFrame:
    """Pitch types with enough pitches, for the pitchers that are published."""
    published = pd.MultiIndex.from_frame(pitchers[["season", "player_id"]])
    kept = types[(types["pitches"] >= MIN_TYPE_PITCHES) & pd.MultiIndex.from_frame(types[["season", "pitcher"]]).isin(published)]
    return pd.DataFrame({
        "player_id": kept["pitcher"], "season": kept["season"], "pitch_type": kept["pitch_type"], "pitches": kept["pitches"],
        "usage": kept["usage"].round(4), "runs": kept["neutral_runs"].round(2),
    })


def _cuts(blocks: dict) -> list[dict]:
    return [{
        "innings": int(innings), "pairs": block["pairs"], "pitchers": block["pitchers"],
        "r": {name: round(block["r"][key], 3) for name, key in (("value", HEADLINE), *((b, b) for b in BASELINES))},
        "lead": {b: [round(v, 3) for v in (block["gain_over"][b]["r"], *block["gain_over"][b]["interval"])] for b in BASELINES},
    } for innings, block in blocks.items()]


def study_payload(pitchers: pd.DataFrame, validation: dict) -> dict:
    """What the page quotes: coverage, the next-season validation over every pair, and over the pairs the design never saw."""
    if "all" not in validation or "confirmation" not in validation:
        raise ValueError("validation.json lacks the confirmation pairs; run `python -m v2.pitch_value.validate --confirm`")
    return {"seasons": sorted(int(s) for s in pitchers["season"].unique()), "pitches": int(pitchers["pitches"].sum()),
            "min_pitches": MIN_PITCHES, "min_type_pitches": MIN_TYPE_PITCHES,
            "validation": _cuts(validation["all"]), "held_back": _cuts(validation["confirmation"])}


def _records(frame: pd.DataFrame, columns: list[str], now: str) -> str:
    # jsonb_populate_recordset does not apply column defaults, so every row carries its timestamp.
    return frame[columns].assign(updated_at=now).to_json(orient="records")


def publish(payload: dict, pitchers: pd.DataFrame, types: pd.DataFrame) -> None:
    if len(pitchers) < MIN_PITCHER_ROWS:
        raise ValueError(f"refusing to publish {len(pitchers)} pitch_values rows (under {MIN_PITCHER_ROWS})")
    now = datetime.now(UTC).isoformat()
    season = payload["seasons"][-1]
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM pitch_value_study"))
        conn.execute(text("INSERT INTO pitch_value_study (season, model_version, payload) VALUES (:s, :v, CAST(:p AS jsonb))"),
                     {"s": season, "v": MODEL_VERSION, "p": json.dumps(payload)})
        conn.execute(text("TRUNCATE pitch_type_values, pitch_values"))
        conn.execute(text("INSERT INTO pitch_values SELECT * FROM jsonb_populate_recordset(NULL::pitch_values, CAST(:rows AS jsonb)) "
                          "ORDER BY season, player_id"), {"rows": _records(pitchers, PITCHER_COLUMNS, now)})
        conn.execute(text("INSERT INTO pitch_type_values SELECT * FROM jsonb_populate_recordset(NULL::pitch_type_values, CAST(:rows AS jsonb)) "
                          "ORDER BY season, player_id, pitch_type"), {"rows": _records(types, TYPE_COLUMNS, now)})
    log.info(f"published pitch value {season} ({MODEL_VERSION}): {len(pitchers)} pitcher rows, {len(types)} pitch-type rows")


def main() -> None:
    setup_logging()
    seasons = pd.read_parquet(OUT_DIR / "pitcher_seasons.parquet")
    pitchers = pitcher_rows(seasons)
    types = type_rows(pd.read_parquet(OUT_DIR / "pitch_types.parquet"), pitchers)
    payload = study_payload(seasons, json.loads((OUT_DIR / "validation.json").read_text()))
    publish(payload, pitchers, types)


if __name__ == "__main__":
    main()
