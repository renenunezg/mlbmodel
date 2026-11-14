"""What does a season's batting run value say about the hitter's next season, against wOBA, xwOBA and the grade?

Each pair is a hitter's season and his next one. The target is batting runs per plate appearance in the second
season (linear weights, ``v2.players.runs``). The measures are the first season's wOBA, Savant xwOBA, and the
batting values per plate appearance: raw, context-neutral, the total (decision + contact + quality) and each
part. For every measure: its match to the same season's wOBA, how well it repeats a season later, and its
correlation with the target. Intervals resample hitters, so a hitter's several pairs move together.

Two versions of the batting-only hitter grade stand in for "what we already knew". The forward grade is the
preseason grade of the second season (2024 to 2026), from a model fitted on nothing after the first season.
The leave-one-pair-out grade covers all five pairs: the live grade's weights refitted without the pair being
graded. The partial correlations ask what a measure adds once the grade is known.

The count situations (first pitch, hitter ahead, two strikes, behind or even), the pitch types and the directions
(inside, away, up, down) are checked for one thing: does a hitter's value per 100 pitches in a situation, against
a pitch type or on one side of the zone, and its gap to his value on all his other pitches, repeat the next season?

The pairs are split by first season: design choices are made on the development pairs and the confirmation
pairs are read once the design is fixed.

Usage:
    python -m v2.batting_value.validate [--confirm]

Reads cache/batting_value/hitter_seasons.parquet, situations.parquet, pitch_types.parquet, sides.parquet and heights.parquet (run
``python -m v2.batting_value.run`` first) and writes validation.json beside them.
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from backend.log import setup_logging
from v2.batting_value.aggregate import KEYS
from v2.batting_value.run import OUT_DIR
from v2.data.pa_dataset import OUTCOMES
from v2.players import runs, savant

log = logging.getLogger(__name__)

# Plate appearances required in both seasons of a pair. The middle one is the headline.
PA_CUTS = (150, 300, 450)
MEASURES = ("woba", "xwoba", "raw", "neutral", "total", "decision", "contact", "quality", "call", "faced")
BASELINES = ("woba", "xwoba")
COMPARED = ("total", "neutral")
# First seasons of the pairs a design may be chosen on, and of the pairs kept back to confirm it.
DEVELOPMENT = (2021, 2022, 2023)
CONFIRMATION = (2024, 2025)
# Seasons with a forward grade model fitted on nothing later than two seasons before.
GRADED_SEASONS = (2024, 2025, 2026)
GRADES = ("forward_grade", "loo_grade")
# The live grade model, whose development pairs the leave-one-pair-out grade refits.
LIVE_MODEL = ("relief_history_release", "validated.json")
BOOTSTRAP_DRAWS = 2000
# Splits checked for repeating: file -> (column, pitches a hitter needs in one value of it in both seasons of a pair).
# A pitch type is a smaller share of a hitter's pitches than a count situation, so it gets the lower bar.
# A direction is a third or more of his pitches, like a count situation, and gets the same bar.
SPLITS = {"situations": ("situation", 300), "pitch_types": ("pitch_type", 200), "sides": ("side", 300), "heights": ("height", 300)}
# Pairs a value of a split needs before its repeat is reported.
SPLIT_MIN_PAIRS = 100
SPLIT_MEASURES = {"total": "runs", "decision": "decision_runs", "contact": "contact_runs", "quality": "quality_runs"}


def forward_grade(season: int) -> pd.Series:
    """Each hitter's batting-only preseason grade for ``season``, from the cached forward grade model."""
    path = OUT_DIR / f"forward_grade_{season}.parquet"
    if path.exists():
        return pd.read_parquet(path)["grade"]
    from v2.players import bio, features, forward

    cached = forward.CACHE_DIR / f"forward_grade_model_{season - 2}_relief_batting.json"
    if not cached.exists():
        raise FileNotFoundError(f"{cached.name} is missing; run the forward evaluation in v2/players first")
    material = forward.SeasonMaterial(season, *forward.grade_models(season - 2, reliever_inputs=True))
    graded = forward.grade_as_of(material, bio.season_dates(season)[0])
    grade = graded.loc[graded["fit_pool"].eq(features.HITTER_POOL), "grade_batting"].rename("grade").rename_axis("batter")
    grade.to_frame().to_parquet(path)
    return grade


def loo_grades() -> pd.DataFrame:
    """season (the pair's first), batter and the batting-only grade from weights fitted without that pair."""
    path = OUT_DIR / "loo_grades.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from v2.players import features, fit, forward
    from v2.players.model import ValueModel

    live = ValueModel.load(forward.CACHE_DIR.joinpath(*LIVE_MODEL))
    data = fit.SeasonData(fit.required_seasons(live.dev_seasons[-1]))
    development = forward.batting_only([fit.build_pair(data, t, live) for t in live.dev_seasons[:-1]])
    in_season = forward.batting_only(fit.cut_pairs(data, live.dev_seasons, live))
    rows = []
    for held in development:
        others = [p for p in development if p is not held]
        model = ValueModel.from_payload(live.to_payload())
        model.trait_weights = fit.fit_trait_weights(others)
        # The in-season pairs of the held pair's target season leave with it, as in the grade's own evaluation.
        model.grade_weights = fit.fit_grade_weights(others, model, [p for p in in_season if p["t"] != held["t"] + 1])
        x, _, _ = fit.grade_design(held, model)
        hitter = ~held["group"].isin(features.PITCHER_GROUPS)
        grade = features.project_runs(x, held["group"], model)[hitter]
        rows.append(pd.DataFrame({"season": held["t"], "batter": grade.index, "loo_grade": grade.to_numpy()}))
    out = pd.concat(rows, ignore_index=True).dropna()
    out.to_parquet(path, index=False)
    return out


def hitter_seasons(table: pd.DataFrame) -> pd.DataFrame:
    """Per hitter-season: plate appearances, every measure per plate appearance, and batting runs per plate appearance."""
    seasons = sorted(int(s) for s in table["season"].unique())
    out = table[[*KEYS, "pa"]].copy()
    out["woba"] = table["woba_value"] / table["woba_denom"]
    boards = pd.concat([savant.load_board("expected_statistics_batter", s) for s in seasons])
    out["xwoba"] = boards.set_index(["season", "player_id"])["est_woba"].reindex(pd.MultiIndex.from_frame(out[KEYS])).to_numpy()
    for measure, column in (("raw", "raw_runs"), ("neutral", "neutral_runs"), ("total", "runs"), ("decision", "decision_runs"),
                            ("contact", "contact_runs"), ("quality", "quality_runs"), ("call", "call_runs"), ("faced", "faced_runs")):
        out[measure] = table[column] / table["pa"]
    weights = {s: np.array([runs.linear_weights(s)[k] for k in OUTCOMES]) for s in seasons}
    counts = table[list(OUTCOMES)].to_numpy(float)
    out["batting_runs"] = [row @ weights[s] / row.sum() if row.sum() else np.nan for row, s in zip(counts, table["season"])]
    grades = pd.concat([forward_grade(s).to_frame().assign(season=s - 1) for s in GRADED_SEASONS if s - 1 in seasons])
    out = out.join(grades.set_index("season", append=True).swaplevel()["grade"].rename("forward_grade"), on=KEYS)
    return out.join(loo_grades().set_index(KEYS)["loo_grade"], on=KEYS)


def pairs(seasons: pd.DataFrame) -> pd.DataFrame:
    nxt = seasons.drop(columns=list(GRADES)).assign(season=seasons["season"] - 1)
    return seasons.merge(nxt, on=KEYS, suffixes=("", "_next"))


def split_pairs(splits: pd.DataFrame, hitters: pd.DataFrame, by: str) -> pd.DataFrame:
    """Per hitter-season and value of ``by``: runs per 100 pitches, each part, and ``vs_rest`` (the total per 100
    minus his total per 100 on all his other pitches), beside the same numbers a season later."""
    out = splits[[*KEYS, by, "pitches"]].copy()
    for measure, column in SPLIT_MEASURES.items():
        out[measure] = 100.0 * splits[column] / splits["pitches"]
    whole = hitters.set_index(KEYS)[["pitches", "runs"]].reindex(pd.MultiIndex.from_frame(splits[KEYS]))
    other = whole["pitches"].to_numpy() - splits["pitches"].to_numpy()
    out["vs_rest"] = out["total"] - 100.0 * (whole["runs"].to_numpy() - splits["runs"].to_numpy()) / np.where(other > 0, other, np.nan)
    return out.merge(out.assign(season=out["season"] - 1), on=[*KEYS, by], suffixes=("", "_next"))


def _split_statistics(frame: pd.DataFrame) -> dict[str, float]:
    return {f"repeat.{m}": _r(frame[m].to_numpy(), frame[f"{m}_next"].to_numpy()) for m in (*SPLIT_MEASURES, "vs_rest")}


def _r(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def _partial(x: np.ndarray, y: np.ndarray, given: np.ndarray) -> float:
    """Correlation of x and y once ``given`` is known."""
    xy, xg, yg = _r(x, y), _r(x, given), _r(y, given)
    return (xy - xg * yg) / np.sqrt((1 - xg ** 2) * (1 - yg ** 2))


def _statistics(frame: pd.DataFrame) -> dict[str, float]:
    column = {name: frame[name].to_numpy() for name in frame.columns}
    target = column["batting_runs_next"]
    out = {}
    for m in MEASURES:
        out[f"same_season_woba.{m}"] = _r(column[m], column["woba"])
        out[f"repeat.{m}"] = _r(column[m], column[f"{m}_next"])
        out[f"next_runs.{m}"] = _r(column[m], target)
    for m in COMPARED:
        for base in BASELINES:
            out[f"gain.{m}_over_{base}"] = out[f"next_runs.{m}"] - out[f"next_runs.{base}"]
    return out


def _grade_statistics(frame: pd.DataFrame, grade: str) -> dict[str, float]:
    column = {name: frame[name].to_numpy() for name in frame.columns}
    target, grade = column["batting_runs_next"], column[grade]
    out = {"next_runs.grade": _r(grade, target)}
    for m in MEASURES:
        out[f"next_runs.{m}"] = _r(column[m], target)
        out[f"partial_given_grade.{m}"] = _partial(column[m], target, grade)
    for m in COMPARED:
        out[f"gain.{m}_over_grade"] = out[f"next_runs.{m}"] - out["next_runs.grade"]
    return out


def bootstrap(frame: pd.DataFrame, statistics, rng: np.random.Generator) -> dict:
    """Point values and 95% intervals of ``statistics`` over hitters drawn with replacement."""
    hitters = frame["batter"].unique()
    rows = {b: np.flatnonzero(frame["batter"].to_numpy() == b) for b in hitters}
    point = statistics(frame)
    draws = {name: [] for name in point}
    for _ in range(BOOTSTRAP_DRAWS):
        sample = statistics(frame.iloc[np.concatenate([rows[b] for b in rng.choice(hitters, len(hitters))])])
        for name, v in sample.items():
            draws[name].append(v)
    return {"pairs": len(frame), "hitters": len(hitters),
            "r": {name: {"r": v, "interval": [float(q) for q in np.percentile(draws[name], [2.5, 97.5])]} for name, v in point.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true", help="also read the confirmation pairs")
    args = parser.parse_args()
    setup_logging()
    table = pd.read_parquet(OUT_DIR / "hitter_seasons.parquet")
    frame = pairs(hitter_seasons(table))
    rng = np.random.default_rng(0)
    sets = {"development": DEVELOPMENT} | ({"confirmation": CONFIRMATION, "all": DEVELOPMENT + CONFIRMATION} if args.confirm else {})
    report = {}
    for name, first_seasons in sets.items():
        for cut in PA_CUTS:
            kept = frame[frame["season"].isin(first_seasons) & (frame["pa"] >= cut) & (frame["pa_next"] >= cut)]
            block = report.setdefault(name, {})[str(cut)] = bootstrap(kept.dropna(subset=["xwoba", "xwoba_next"]), _statistics, rng)
            for grade in GRADES:
                graded = kept.dropna(subset=[grade])
                if len(graded):
                    block[grade] = bootstrap(graded, lambda f: _grade_statistics(f, grade), rng)
            log.info(f"{name}, {cut}+ PA both seasons: {block['pairs']} pairs, {block['hitters']} hitters")
            for title, part in (("", block), *((g, block[g]) for g in GRADES if g in block)):
                if title:
                    log.info(f" given the {title.replace('_', ' ')}: {part['pairs']} pairs")
                for key, v in part["r"].items():
                    log.info(f"  {key:>32}  {v['r']:+.3f}  [{v['interval'][0]:+.3f}, {v['interval'][1]:+.3f}]")
    for split, (by, pitches) in SPLITS.items():
        paired = split_pairs(pd.read_parquet(OUT_DIR / f"{split}.parquet"), table, by)
        paired = paired[(paired["pitches"] >= pitches) & (paired["pitches_next"] >= pitches)].dropna(subset=["vs_rest", "vs_rest_next"])
        for name, first_seasons in sets.items():
            for value, group in paired[paired["season"].isin(first_seasons)].groupby(by):
                if len(group) < SPLIT_MIN_PAIRS:
                    continue
                block = report.setdefault(split, {}).setdefault(name, {})[value] = bootstrap(group, _split_statistics, rng)
                log.info(f"{name}, {value}, {pitches}+ pitches both seasons: {block['pairs']} pairs  " + "  ".join(
                    f"{key.split('.')[1]} {v['r']:+.2f} [{v['interval'][0]:+.2f}, {v['interval'][1]:+.2f}]" for key, v in block["r"].items()))
    (OUT_DIR / "validation.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
