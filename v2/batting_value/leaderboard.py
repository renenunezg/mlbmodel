"""The leaderboard the site would show: compact rows as JSON, and a standalone page to look at them.

Usage:
    python -m v2.batting_value.leaderboard

Reads cache/batting_value/hitter_seasons.parquet, pitch_groups.parquet, pitch_types.parquet, zones.parquet and situations.parquet
(``python -m v2.batting_value.run``) and writes leaderboard.json and leaderboard.html beside them. The page is
one file with the rows inlined.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from v2.batting_value.aggregate import KEYS
from v2.batting_value.run import OUT_DIR

TEMPLATE = Path(__file__).with_name("leaderboard.html")
# The smallest sample the page lets a reader choose; hitter-seasons under it are not shipped.
MIN_PA = 200
PARTS = ("runs", "decision_runs", "contact_runs", "quality_runs")
# The count situations the page lists. Behind or even (0-1, 1-1) is left off: what a hitter does there beyond
# his other pitches does not repeat (+.05 [-.04, +.15] on the development pairs), so it is his overall level again.
SITUATIONS = ("two_strikes", "ahead", "first_pitch")
# The pitch types the page lists, in the order it lists them: the four with 100+ development pairs at 200+ pitches
# in both seasons, each with a year-to-year repeat clear of zero. Cutters, curveballs and sweepers repeat at .13 to
# .25 with intervals that reach zero at 150+ pitches; splitters and knuckle curves are too rare to test.
PITCH_TYPES = ("FF", "SI", "SL", "CH")
# A display floor, not a reliability claim: a hitter-season row for a pitch type he saw fewer times is not shipped.
MIN_TYPE_PITCHES = 100


def _split_rows(table: pd.DataFrame, by: str, kept: pd.MultiIndex) -> list[list]:
    table = table[pd.MultiIndex.from_frame(table[KEYS]).isin(kept)]
    return [[int(r.season), int(r.batter), getattr(r, by), int(r.pitches), *(round(getattr(r, part), 1) for part in PARTS)]
            for r in table.itertuples()]


def payload(hitters: pd.DataFrame, groups: pd.DataFrame, types: pd.DataFrame, zones: pd.DataFrame, situations: pd.DataFrame) -> dict:
    kept = hitters[hitters["pa"] >= MIN_PA]
    who = pd.MultiIndex.from_frame(kept[KEYS])
    return {
        "seasons": sorted(int(s) for s in hitters["season"].unique()),
        # [season, hitter id, name, team, PA, pitches, total, decision, contact, quality, called-strike luck, context-neutral runs]
        "hitters": [[int(r.season), int(r.batter), r.name, r.team, int(r.pa), int(r.pitches),
                     *(round(getattr(r, part), 1) for part in PARTS), round(r.call_runs, 1), round(r.neutral_runs, 1)]
                    for r in kept.itertuples()],
        # [season, hitter id, count situation / pitch group / zone, pitches, total, decision, contact, quality]; the
        # hitter's name, team, PA and season totals are on his row in "hitters"
        "counts": _split_rows(situations[situations["situation"].isin(SITUATIONS)], "situation", who),
        "groups": _split_rows(groups, "pitch_group", who),
        "pitch_types": list(PITCH_TYPES),
        "types": _split_rows(types[types["pitch_type"].isin(PITCH_TYPES) & (types["pitches"] >= MIN_TYPE_PITCHES)], "pitch_type", who),
        "zones": _split_rows(zones, "zone", who),
    }


def main() -> None:
    data = payload(*(pd.read_parquet(OUT_DIR / f"{name}.parquet") for name in ("hitter_seasons", "pitch_groups", "pitch_types", "zones", "situations")))
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    (OUT_DIR / "leaderboard.json").write_text(text)
    marker = "/*DATA*/null"
    template = TEMPLATE.read_text()
    if marker not in template:
        raise ValueError(f"{TEMPLATE.name} has no {marker} placeholder")
    (OUT_DIR / "leaderboard.html").write_text(template.replace(marker, text.replace("</", "<\\/")))
    print(f"{len(data['hitters'])} hitter rows, {len(data['counts'])} count rows, {len(data['groups'])} pitch-group rows, "
          f"{len(data['types'])} pitch-type rows, {len(data['zones'])} zone rows -> {OUT_DIR}")


if __name__ == "__main__":
    main()
