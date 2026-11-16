"""Write the xWAR tables the site reads: the study row and one row per player.

Usage:
    python -m v2.xwar.build
    python -m v2.xwar.validate --confirm
    MLBMODEL_DB_WRITES=1 python -m v2.xwar.publish

One transaction: the table is replaced whole, as the run value tables are. Only the latest season is written,
because that is all the site reads; the study row still describes every season behind the shrink.
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
from v2.pitch_value.publish import display_name
from v2.xwar import gap
from v2.xwar.build import FIP_HOME_RUN_RUNS, FIRST_SEASON, OUT_DIR

log = logging.getLogger(__name__)

MODEL_VERSION = "xwar-v1"
# The completed 2026 season has 1,470 players; under 70% of that is a partial season or a broken input.
MIN_ROWS = 1000
# Rows per insert: the pooler stalls on very large statements.
INSERT_SLICE = 2000
COUNTS = ["balls", "balls_allowed"]
MEASURES = ["war", "xwar", "realized_gap_runs", "earned_gap_runs", "home_runs_allowed", "expected_home_runs", "home_run_gap",
            "earned_home_runs"]
COLUMNS = ["player_id", "season", "player_name", "team", "war", "xwar", "balls", "realized_gap_runs", "earned_gap_runs",
           "balls_allowed", "home_runs_allowed", "expected_home_runs", "home_run_gap", "earned_home_runs"]


def rows(frame: pd.DataFrame) -> pd.DataFrame:
    """The latest season's players. A side a player did not play stays null."""
    kept = frame[frame["season"] == frame["season"].max()]
    team = kept["team"].map(normalize_team)
    unknown = sorted(set(team) - set(TEAM_ID_BY_CODE))
    if unknown:
        raise ValueError(f"xwar rows carry unknown team codes: {unknown}")
    # Whole numbers must reach Postgres as integers or null, never as 412.0.
    return kept.assign(player_name=display_name(kept["player_name"]), team=team,
                       **{c: kept[c].astype("Int64") for c in COUNTS}, **{c: kept[c].round(2) for c in MEASURES})[COLUMNS]


def coverage(frame: pd.DataFrame) -> dict:
    """The seasons behind the shrink and the players in the latest one."""
    seasons = sorted(int(s) for s in frame["season"].unique())
    return {"seasons": seasons, "players": int((frame["season"] == seasons[-1]).sum())}


def _triple(block: dict) -> list[float]:
    """[value, interval low, interval high]."""
    return [round(v, 3) for v in (block["r"], *block["interval"])]


def _side(shrink: gap.Shrink | None, development: dict, confirmation: dict) -> dict:
    """What a side keeps of its gap and the repeat that decided it. The pinned shrink must be the fitted one."""
    fitted = development["shrink"]
    if development["gate"]["passed"] != (shrink is not None):
        raise ValueError("gap.py does not match the gate in validation.json; run `python -m v2.xwar.validate` and pin its result")
    if shrink is not None and (abs(shrink.k - fitted["k"]) > 0.05 or abs(shrink.decay - fitted["decay"]) > 0.0005):
        raise ValueError(f"gap.py pins {shrink} but validation.json fitted k {fitted['k']:.1f}, decay {fitted['decay']:.3f}")
    return {"kept": shrink is not None, "k": None if shrink is None else shrink.k, "season_weight": None if shrink is None else shrink.decay,
            "min_balls": development["min_balls"], "pairs": development["gate"]["pairs"], "repeat": _triple(development["gate"]),
            "held_back": {"pairs": confirmation["gate"]["pairs"], "repeat": _triple(confirmation["gate"])}}


def study_payload(frame: pd.DataFrame, validation: dict) -> dict:
    """What the page quotes: coverage, what each side keeps of its gap and why, and the single read of the later pairs."""
    if "confirmation" not in validation:
        raise ValueError("validation.json lacks the confirmation pairs; run `python -m v2.xwar.validate --confirm`")
    development, confirmation = validation["development"], validation["confirmation"]
    rivals, later = development["hitter"]["rivals"], confirmation["next_season_fwar"]
    return {
        **coverage(frame), "first_season": FIRST_SEASON, "home_run_runs": round(FIP_HOME_RUN_RUNS, 3),
        "hitter": _side(gap.HITTER, development["hitter"], confirmation["hitter"]) | {
            # The fitted season weight against the two simpler readings of a hitter's record.
            "lead": {name: [round(v, 3) for v in (rivals[name]["fitted_lead"], *rivals[name]["interval"])] for name in rivals}},
        "pitcher": _side(gap.PITCHER, development["pitcher"], confirmation["pitcher"]),
        # A hitter's season per 600 PA against his next season's hitting fWAR per 600 PA, read three ways.
        "next_season_fwar": {"min_pa": later["min_pa"], "pairs": later["pairs"],
                             "r": {name: round(block["r"], 3) for name, block in later["r"].items()},
                             "lead": {name: _triple(block) for name, block in later["lead"].items()}},
    }


def _insert(conn, table: str, frame: pd.DataFrame, now: str) -> None:
    # jsonb_populate_recordset does not apply column defaults, so every row carries its timestamp.
    for start in range(0, len(frame), INSERT_SLICE):
        records = frame.iloc[start:start + INSERT_SLICE].assign(updated_at=now).to_json(orient="records")
        conn.execute(text(f"INSERT INTO {table} SELECT * FROM jsonb_populate_recordset(NULL::{table}, CAST(:rows AS jsonb))"), {"rows": records})


def publish(payload: dict, players: pd.DataFrame, keep_study: bool = False) -> None:
    """Replace the table. With ``keep_study`` the payload is only ``coverage``, merged into the study row so its
    validation (written by a full run with ``validate --confirm``) stays."""
    if len(players) < MIN_ROWS:
        raise ValueError(f"refusing to publish {len(players)} xwar rows (under {MIN_ROWS})")
    now = datetime.now(UTC).isoformat()
    season = payload["seasons"][-1]
    with engine.begin() as conn:
        if keep_study:
            updated = conn.execute(text("UPDATE xwar_study SET season = :s, payload = payload || CAST(:p AS jsonb) "
                                        "WHERE model_version = :v"), {"s": season, "v": MODEL_VERSION, "p": json.dumps(payload)})
            if updated.rowcount != 1:
                raise ValueError(f"expected one {MODEL_VERSION} study row to update, found {updated.rowcount}")
        else:
            conn.execute(text("DELETE FROM xwar_study"))
            conn.execute(text("INSERT INTO xwar_study (season, model_version, payload) VALUES (:s, :v, CAST(:p AS jsonb))"),
                         {"s": season, "v": MODEL_VERSION, "p": json.dumps(payload)})
        conn.execute(text("TRUNCATE xwar"))
        _insert(conn, "xwar", players.sort_values(["season", "player_id"]), now)
    log.info(f"published {len(players)} xwar rows for {season}")


def main() -> None:
    setup_logging()
    frame = pd.read_parquet(OUT_DIR / "player_seasons.parquet")
    publish(study_payload(frame, json.loads((OUT_DIR / "validation.json").read_text())), rows(frame))


if __name__ == "__main__":
    main()
