"""Value every pitch from the hitter's side, split it into components, and write the season aggregates.

Usage:
    python -m v2.batting_value.run [--seasons 2021 2022 2023 2024 2025 2026]

Reads the local Statcast caches (and the cached Savant boards for names). Writes under cache/batting_value/:
hitter_seasons.parquet, pitch_groups.parquet, pitch_types.parquet, zones.parquet, situations.parquet,
calibration_{zone,count,pitch_group}.csv and checks.json. Stops if the components do not sum.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from backend.log import setup_logging
from v2.batting_value import aggregate, components, contact, difficulty, pitches, values
from v2.batting_value.components import COMPONENTS, SUM_TOLERANCE
from v2.pitch_value import batted

log = logging.getLogger(__name__)

OUT_DIR = contact.OUT_DIR
DEFAULT_SEASONS = contact.SEASONS
CALIBRATION_BY = ("zone", "count", "pitch_group")
# Plate appearances a hitter-season needs to enter the same-season wOBA check.
WOBA_CHECK_PA = 300
# The largest gap allowed between a predicted and an actual rate in any zone, count or pitch group.
CALIBRATION_TOLERANCE = 0.02


def build(seasons: list[int]) -> dict:
    frame = pitches.load(seasons)
    built = values.build(frame)
    neutral = built.frame["neutral"].rename("neutral")
    if contact.KEPT:
        hitter_model = contact.neutral_in_play(frame, built.table, built.contact, contact.KEPT)
        neutral[hitter_model.index] = hitter_model
    league = difficulty.fit(frame, neutral)
    parts = components.split(frame, neutral, league)

    seen = frame["seen"]
    swing = seen & frame["swing"]
    domain = {"decision": seen, "contact": swing, "quality": swing & frame["result"].eq("in_play"), "call": seen & ~swing}
    scale = values.environment(frame)
    scores = pd.DataFrame({
        "raw": built.frame["raw"], "expected": built.frame["expected"],
        "context": values.adjusted(frame, built.frame["expected"].rename("context"), seen, scale),
        **{name: values.adjusted(frame, parts[name], where, scale) for name, where in domain.items()},
        "faced": values.adjusted(frame, league["league"].rename("faced"), seen, scale),
    })
    # The park and the season come out of each part on its own pitches (a swing, a ball in play, a take), so
    # the adjusted neutral value is the sum of the adjusted parts. Adjusting the neutral value in one piece
    # shrinks one park effect instead of five and differs a little; the gap is reported, not used.
    scores["neutral"] = scores[[*domain, "faced"]].sum(axis=1)
    one_piece = values.adjusted(frame, neutral, seen, scale) - scores["neutral"]
    one_piece_gap = one_piece[seen].groupby([frame.loc[seen, "season"], frame.loc[seen, "batter"]]).sum()
    calibration = {by: difficulty.calibration(frame, league, neutral, by) for by in CALIBRATION_BY}
    worst = max(difficulty.worst_rate_gap(table) for table in calibration.values())
    if worst > CALIBRATION_TOLERANCE:
        raise ValueError(f"a league rate misses its zone, count or pitch-group average by {worst:.3f}")
    hitters = aggregate.hitter_seasons(frame, scores)
    log.info(f"{int(seen.sum()):,} pitches seen over {len(seasons)} seasons, {len(hitters):,} hitter-seasons")
    return {
        "hitter_seasons": hitters,
        "pitch_groups": aggregate.splits(frame, scores, "pitch_group"),
        "pitch_types": aggregate.splits(frame, scores, "pitch_type"),
        "zones": aggregate.splits(frame, scores, "zone"),
        "situations": aggregate.splits(frame, scores, "situation"),
        "calibration": calibration,
        "checks": checks(frame, neutral, league, parts, hitters) | {
            "calibration_worst_rate_gap": worst,
            "one_piece_adjustment_gap_runs": {"max": float(one_piece_gap.abs().max()), "sd": float(one_piece_gap.std())},
        },
    }


def checks(frame: pd.DataFrame, neutral: pd.Series, league: pd.DataFrame, parts: pd.DataFrame, hitters: pd.DataFrame) -> dict:
    seen = frame["seen"]
    pitch_gap = float(np.abs(parts[seen].sum(axis=1) - (neutral[seen] - league.loc[seen, "league"])).max())
    season_gap = hitters["neutral_runs"] - hitters[["runs", "call_runs", "faced_runs"]].sum(axis=1)
    if season_gap.abs().max() > SUM_TOLERANCE * len(frame):
        raise ValueError(f"a hitter-season's parts miss its neutral total by {season_gap.abs().max():.3g} runs")
    checked = hitters[hitters["pa"] >= WOBA_CHECK_PA]
    woba = checked["woba_value"] / checked["woba_denom"]
    r = lambda column: float(np.corrcoef(checked[column] / checked["pa"], woba)[0, 1])
    out = {
        "pitches_seen": int(seen.sum()),
        "balls_in_play_untracked_share": float(1.0 - batted.tracked(frame)[seen & frame["result"].eq("in_play")].mean()),
        "pitch_sum_gap_max_runs": pitch_gap,
        "hitter_season_sum_gap_max_runs": float(season_gap.abs().max()),
        # Before the park and season are removed. Zero means the league models are unbiased over all pitches.
        "league_mean_per_pitch": {name: float(parts.loc[seen, name].mean()) for name in (*COMPONENTS, "call")},
        "sd_per_pitch": {name: float(parts.loc[seen, name].std()) for name in (*COMPONENTS, "call")},
        "same_season_woba_r": {"pa": WOBA_CHECK_PA, "hitter_seasons": len(checked),
                               **{name: r(f"{name}_runs") for name in ("raw", "expected", "neutral")}, "total": r("runs")},
    }
    log.info(f"components sum to neutral minus league within {pitch_gap:.2g} runs a pitch; hitter-season totals within "
             f"{out['hitter_season_sum_gap_max_runs']:.2g} runs after the park and season adjustments")
    log.info(f"same-season wOBA, {WOBA_CHECK_PA}+ PA: " + ", ".join(f"{k} {v:+.3f}" for k, v in out["same_season_woba_r"].items() if isinstance(v, float)))
    return out


def write(outputs: dict, out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("hitter_seasons", "pitch_groups", "pitch_types", "zones", "situations"):
        outputs[name].to_parquet(out_dir / f"{name}.parquet", index=False)
    for by, table in outputs["calibration"].items():
        table.to_csv(out_dir / f"calibration_{by}.csv", index=False)
    (out_dir / "checks.json").write_text(json.dumps(outputs["checks"], indent=1))
    log.info(f"wrote {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=list(DEFAULT_SEASONS))
    args = parser.parse_args()
    setup_logging()
    write(build(sorted(args.seasons)))


if __name__ == "__main__":
    main()
