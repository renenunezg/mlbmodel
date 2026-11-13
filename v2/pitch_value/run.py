"""Fit the run expectancy table, value every pitch, and write the season aggregates.

Usage:
    python -m v2.pitch_value.run [--seasons 2021 2022 2023 2024 2025 2026]

Reads the local Statcast caches only. Writes under cache/pitch_value/: re288.csv, park_effects.csv,
pitcher_seasons.parquet and pitch_types.parquet.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from backend.log import setup_logging
from v2.pitch_value import aggregate, batted, context, expectancy, neutral, pitches, value

log = logging.getLogger(__name__)

OUT_DIR = pitches.STATCAST_DIR / "pitch_value"
DEFAULT_SEASONS = (2021, 2022, 2023, 2024, 2025, 2026)


def build(seasons: list[int]) -> dict[str, pd.DataFrame]:
    frame = pitches.load(seasons)
    table = expectancy.fit(frame)
    raw = value.pitch_values(frame, table)
    contact = batted.fit(frame, table)
    expected = batted.expected_values(frame, table, raw, contact)
    fitted = context.fit(frame, expected)
    neutral_value = neutral.neutral_values(frame, table, raw, contact)
    values = pd.DataFrame({
        "raw": raw, "expected": expected,
        "adjusted": context.adjust(frame, expected, fitted),
        "neutral": context.adjust(frame, neutral_value, context.fit(frame, neutral_value)),
    })
    log.info(f"{len(frame):,} pitches over {len(seasons)} seasons")
    return {
        "re288": table,
        "park_effects": fitted.parks,
        "pitcher_seasons": aggregate.pitcher_seasons(frame, values),
        "pitch_types": aggregate.pitch_types(frame, values),
    }


def write(outputs: dict[str, pd.DataFrame], out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs["re288"].to_csv(out_dir / "re288.csv", index=False)
    outputs["park_effects"].to_csv(out_dir / "park_effects.csv", index=False)
    outputs["pitcher_seasons"].to_parquet(out_dir / "pitcher_seasons.parquet", index=False)
    outputs["pitch_types"].to_parquet(out_dir / "pitch_types.parquet", index=False)
    log.info(f"wrote {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=list(DEFAULT_SEASONS))
    args = parser.parse_args()
    setup_logging()
    write(build(sorted(args.seasons)))


if __name__ == "__main__":
    main()
