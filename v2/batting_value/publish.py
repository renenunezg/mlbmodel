"""Write the batting run value tables the site reads: the study row, hitter-seasons and their splits.

Usage:
    python -m v2.batting_value.run
    python -m v2.batting_value.validate --confirm
    MLBMODEL_DB_WRITES=1 python -m v2.batting_value.publish

One transaction: the tables are replaced whole, as the pitch run value tables are. Only the latest season is
written, because that is all the site reads; the study row still describes every season the values were fitted on.
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
from v2.batting_value.aggregate import KEYS
from v2.batting_value.components import COMPONENTS
from v2.batting_value.contact import DECIDING_CUT, KEPT
from v2.batting_value.leaderboard import MIN_PA, MIN_TYPE_PITCHES, PITCH_TYPES, SITUATIONS
from v2.batting_value.run import OUT_DIR
from v2.batting_value.validate import BASELINES, SPLITS

log = logging.getLogger(__name__)

MODEL_VERSION = "batting-value-v1"
# A completed season has about 360 hitters at MIN_PA; far fewer means a partial season or a broken input.
MIN_HITTER_ROWS = 250
# The cut whose per-part table the page quotes.
PARTS_PA = "300"
# Rows per insert: the pooler stalls on very large statements.
INSERT_SLICE = 2000
HITTER_COLUMNS = ["player_id", "season", "player_name", "team", "pa", "pitches", "decision_runs", "contact_runs",
                  "quality_runs", "call_runs", "faced_runs", "raw_runs"]
SPLIT_COLUMNS = ["player_id", "season", "split", "pitches", "decision_runs", "contact_runs", "quality_runs"]
RUN_COLUMNS = [f"{name}_runs" for name in COMPONENTS]


def hitter_rows(hitters: pd.DataFrame) -> pd.DataFrame:
    """The latest season's hitters with enough plate appearances."""
    kept = hitters[(hitters["season"] == hitters["season"].max()) & (hitters["pa"] >= MIN_PA)]
    team = kept["team"].map(normalize_team)
    unknown = sorted(set(team) - set(TEAM_ID_BY_CODE))
    if unknown:
        raise ValueError(f"batting value rows carry unknown team codes: {unknown}")
    out = kept[["season", "pa", "pitches", *RUN_COLUMNS, "call_runs", "faced_runs", "raw_runs"]].round(2)
    return out.assign(player_id=kept["batter"], player_name=kept["name"], team=team)[HITTER_COLUMNS]


def split_rows(splits: dict[str, pd.DataFrame], hitters: pd.DataFrame) -> pd.DataFrame:
    """Count situations, pitch groups, pitch types and zones for the hitters that are published."""
    types = splits["pitch_types"]
    parts = [
        splits["situations"][splits["situations"]["situation"].isin(SITUATIONS)].rename(columns={"situation": "split"}),
        splits["pitch_groups"].rename(columns={"pitch_group": "split"}),
        types[types["pitch_type"].isin(PITCH_TYPES) & (types["pitches"] >= MIN_TYPE_PITCHES)].rename(columns={"pitch_type": "split"}),
        splits["zones"].rename(columns={"zone": "split"}),
    ]
    out = pd.concat([part[[*KEYS, "split", "pitches", *RUN_COLUMNS]] for part in parts], ignore_index=True)
    published = pd.MultiIndex.from_frame(hitters[["season", "player_id"]])
    out = out[pd.MultiIndex.from_frame(out[KEYS]).isin(published)].rename(columns={"batter": "player_id"})
    if out.duplicated(["player_id", "season", "split"]).any():
        raise ValueError("two split families share a code")
    return out.round(2)[SPLIT_COLUMNS]


def _triple(block: dict) -> list[float]:
    """[value, interval low, interval high]."""
    return [round(v, 3) for v in (block["r"], *block["interval"])]


def _cuts(blocks: dict) -> list[dict]:
    out = []
    for pa, block in blocks.items():
        r = block["r"]
        row = {"pa": int(pa), "pairs": block["pairs"], "hitters": block["hitters"],
               "r": {name: round(r[f"next_runs.{key}"]["r"], 3) for name, key in (("value", "total"), ("neutral", "neutral"), *((b, b) for b in BASELINES))},
               "lead": {b: _triple(r[f"gain.total_over_{b}"]) for b in BASELINES}}
        if "forward_grade" in block:
            grade = block["forward_grade"]
            row["grade"] = {"pairs": grade["pairs"], "r": round(grade["r"]["next_runs.grade"]["r"], 3),
                            "value": round(grade["r"]["next_runs.total"]["r"], 3),
                            "lead": _triple(grade["r"]["gain.total_over_grade"]),
                            "partial": _triple(grade["r"]["partial_given_grade.total"])}
        out.append(row)
    return out


def _parts(block: dict) -> dict:
    """Per measure: its match to the same season's wOBA, its repeat a season later, its correlation with next-season batting runs."""
    r = block["r"]
    return {"pa": int(PARTS_PA), "pairs": block["pairs"], "measures": {
        name: {"woba": round(r[f"same_season_woba.{name}"]["r"], 3), "repeat": _triple(r[f"repeat.{name}"]),
               "next_runs": _triple(r[f"next_runs.{name}"])}
        for name in ("total", *COMPONENTS, "call", "faced", *BASELINES)}}


def _split_repeats(validation: dict, split: str, listed: tuple[str, ...]) -> dict:
    blocks = validation[split]
    missing = [value for value in listed if value not in blocks["all"] or value not in blocks["confirmation"]]
    if missing:
        raise ValueError(f"validation.json has no repeat for the listed {split} {missing}")
    return {"pitches": SPLITS[split][1], "values": {
        value: {"pairs": blocks["all"][value]["pairs"], "repeat": _triple(blocks["all"][value]["r"]["repeat.total"]),
                "vs_rest": _triple(blocks["all"][value]["r"]["repeat.vs_rest"]),
                "held_back": _triple(blocks["confirmation"][value]["r"]["repeat.total"])}
        for value in listed}}


def study_payload(hitters: pd.DataFrame, validation: dict, contact_test: dict) -> dict:
    """What the page quotes: coverage, the next-season validation over every pair and over the pairs the design
    never saw, the parts, and how well the listed splits repeat."""
    if "all" not in validation or "confirmation" not in validation:
        raise ValueError("validation.json lacks the confirmation pairs; run `python -m v2.batting_value.validate --confirm`")
    test = contact_test[str(DECIDING_CUT)]
    return {
        "seasons": sorted(int(s) for s in hitters["season"].unique()), "pitches": int(hitters["pitches"].sum()),
        "min_pa": MIN_PA, "min_type_pitches": MIN_TYPE_PITCHES,
        "validation": _cuts(validation["all"]), "held_back": _cuts(validation["confirmation"]),
        "parts": _parts(validation["all"][PARTS_PA]),
        "counts": _split_repeats(validation, "situations", SITUATIONS),
        "pitch_types": _split_repeats(validation, "pitch_types", PITCH_TYPES),
        # The hitter contact model: what was added to exit velocity and launch angle, and what each candidate gained.
        "contact_model": {"kept": list(KEPT), "balls_in_play": DECIDING_CUT, "pairs": test["pairs"],
                          "gain": {name: [round(v, 3) for v in (gain["r"], *gain["interval"])] for name, gain in test["gain_over_plain"].items()}},
    }


def _insert(conn, table: str, frame: pd.DataFrame, now: str) -> None:
    # jsonb_populate_recordset does not apply column defaults, so every row carries its timestamp.
    for start in range(0, len(frame), INSERT_SLICE):
        rows = frame.iloc[start:start + INSERT_SLICE].assign(updated_at=now).to_json(orient="records")
        conn.execute(text(f"INSERT INTO {table} SELECT * FROM jsonb_populate_recordset(NULL::{table}, CAST(:rows AS jsonb))"), {"rows": rows})


def publish(payload: dict, hitters: pd.DataFrame, splits: pd.DataFrame) -> None:
    if len(hitters) < MIN_HITTER_ROWS:
        raise ValueError(f"refusing to publish {len(hitters)} batting_values rows (under {MIN_HITTER_ROWS})")
    now = datetime.now(UTC).isoformat()
    season = payload["seasons"][-1]
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM batting_value_study"))
        conn.execute(text("INSERT INTO batting_value_study (season, model_version, payload) VALUES (:s, :v, CAST(:p AS jsonb))"),
                     {"s": season, "v": MODEL_VERSION, "p": json.dumps(payload)})
        conn.execute(text("TRUNCATE batting_split_values, batting_values"))
        _insert(conn, "batting_values", hitters.sort_values(["season", "player_id"]), now)
        _insert(conn, "batting_split_values", splits.sort_values(["season", "player_id", "split"]), now)
    log.info(f"published batting value {season} ({MODEL_VERSION}): {len(hitters)} hitter rows, {len(splits)} split rows")


def build() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    seasons = pd.read_parquet(OUT_DIR / "hitter_seasons.parquet")
    hitters = hitter_rows(seasons)
    splits = split_rows({name: pd.read_parquet(OUT_DIR / f"{name}.parquet") for name in ("situations", "pitch_groups", "pitch_types", "zones")}, hitters)
    payload = study_payload(seasons, json.loads((OUT_DIR / "validation.json").read_text()), json.loads((OUT_DIR / "contact_test.json").read_text()))
    return payload, hitters, splits


def main() -> None:
    setup_logging()
    publish(*build())


if __name__ == "__main__":
    main()
