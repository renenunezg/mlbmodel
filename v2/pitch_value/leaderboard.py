"""The leaderboard the site shows: compact rows as JSON, and a standalone page to look at them.

Usage:
    python -m v2.pitch_value.leaderboard

Reads cache/pitch_value/pitcher_seasons.parquet and pitch_types.parquet (``python -m v2.pitch_value.run``)
and writes leaderboard.json and leaderboard.html beside them. The page is one file with the rows inlined.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from v2.pitch_value.run import OUT_DIR

TEMPLATE = Path(__file__).with_name("leaderboard.html")
# The smallest samples the page lets a reader choose; rows under them are not shipped.
MIN_PITCHES = 500
MIN_TYPE_PITCHES = 200


def display_name(name: pd.Series) -> pd.Series:
    """Statcast writes 'Last, First'."""
    parts = name.str.split(", ", n=1, expand=True)
    return (parts[1] + " " + parts[0]).where(parts[1].notna(), name)


def payload(pitchers: pd.DataFrame, types: pd.DataFrame) -> dict:
    pitchers = pitchers.assign(name=display_name(pitchers["name"]))
    who = pitchers.set_index(["season", "pitcher"])[["name", "team"]]
    kept = pitchers[pitchers["pitches"] >= MIN_PITCHES]
    types = types[types["pitches"] >= MIN_TYPE_PITCHES].join(who, on=["season", "pitcher"])
    return {
        "seasons": sorted(int(s) for s in pitchers["season"].unique()),
        # [season, pitcher id, name, team, pitches, runs as they happened, adjusted runs in context, context-neutral runs]
        "pitchers": [[int(r.season), int(r.pitcher), r.name, r.team, int(r.pitches), round(r.raw_runs, 1), round(r.runs, 1),
                      round(r.neutral_runs, 1)] for r in kept.itertuples()],
        # [season, pitcher id, name, team, pitch type, pitches, context-neutral runs, share of his pitches]
        "types": [[int(r.season), int(r.pitcher), r.name, r.team, r.pitch_type, int(r.pitches), round(r.neutral_runs, 1),
                   round(r.usage, 3)] for r in types.itertuples()],
    }


def main() -> None:
    data = payload(pd.read_parquet(OUT_DIR / "pitcher_seasons.parquet"), pd.read_parquet(OUT_DIR / "pitch_types.parquet"))
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    (OUT_DIR / "leaderboard.json").write_text(text)
    marker = "/*DATA*/null"
    template = TEMPLATE.read_text()
    if marker not in template:
        raise ValueError(f"{TEMPLATE.name} has no {marker} placeholder")
    (OUT_DIR / "leaderboard.html").write_text(template.replace(marker, text.replace("</", "<\\/")))
    print(f"{len(data['pitchers'])} pitcher rows, {len(data['types'])} pitch-type rows -> {OUT_DIR}")


if __name__ == "__main__":
    main()
