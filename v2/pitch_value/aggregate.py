"""Season totals per pitcher and per pitch type, in runs and in runs per 100 pitches.

``values`` holds one column per valuation of the same pitches: ``raw`` (what happened), ``expected`` (expected
contact), ``adjusted`` (expected contact with park, hitter and season removed) and ``neutral`` (the same
adjustments on the context-neutral value).
"""
from __future__ import annotations

import pandas as pd

from v2.pitch_value.batted import tracked
from v2.pitch_value.value import thrown


def _totals(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    out = frame.groupby(by).agg(
        pitches=("adjusted", "size"), raw_runs=("raw", "sum"), expected_runs=("expected", "sum"), runs=("adjusted", "sum"),
        neutral_runs=("neutral", "sum"), in_play=("in_play", "sum"), neutral_contact_runs=("neutral_contact", "sum"),
    ).reset_index()
    out["per_100"] = 100.0 * out["runs"] / out["pitches"]
    out["neutral_per_100"] = 100.0 * out["neutral_runs"] / out["pitches"]
    return out


def _frame(pitches: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    in_play = tracked(pitches)
    frame = pitches[["season", "pitcher", "pitcher_name", "team", "pitch_type"]].join(values).assign(
        in_play=in_play, neutral_contact=values["neutral"].where(in_play, 0.0))
    return frame[thrown(pitches)]


def pitcher_seasons(pitches: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    """One row per pitcher and season, best context-neutral total first."""
    frame = _frame(pitches, values)
    out = _totals(frame, ["season", "pitcher"])
    who = frame.groupby(["season", "pitcher"]).agg(
        name=("pitcher_name", "first"), team=("team", lambda t: t.value_counts().index[0])).reset_index()
    return who.merge(out, on=["season", "pitcher"]).sort_values(["season", "neutral_runs"]).reset_index(drop=True)


def pitch_types(pitches: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    """One row per pitcher, season and pitch type, with the share of his pitches it made up."""
    frame = _frame(pitches, values).dropna(subset=["pitch_type"])
    out = _totals(frame, ["season", "pitcher", "pitch_type"])
    out["usage"] = out["pitches"] / out.groupby(["season", "pitcher"])["pitches"].transform("sum")
    return out.sort_values(["season", "neutral_runs"]).reset_index(drop=True)
