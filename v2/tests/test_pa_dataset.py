"""Unit tests for v2/data/pa_dataset.py - synthetic data, no disk I/O."""
from __future__ import annotations

import numpy as np
import pandas as pd

from v2.data.pa_dataset import (
    EVENT_TO_OUTCOME,
    EVENT_TO_OUT_SUBTYPE,
    NON_PA_EVENTS,
    OUTCOMES,
    transform_pitch_frame,
)


def _synthetic_pitches(events: list[str | None]) -> pd.DataFrame:
    """Build a minimal pitch-level frame with one row per supplied event."""
    n = len(events)
    return pd.DataFrame({
        "game_pk":        np.arange(n, dtype="int64"),
        "game_date":      pd.to_datetime("2026-04-01"),
        "batter":         np.arange(100, 100 + n, dtype="int64"),
        "pitcher":        np.arange(200, 200 + n, dtype="int64"),
        "stand":          ["R"] * n,
        "p_throws":       ["R"] * n,
        "home_team":      ["LAD"] * n,
        "away_team":      ["NYY"] * n,
        "balls":          [1] * n,
        "strikes":        [2] * n,
        "events":         events,
        "launch_speed":   [90.0] * n,
        "launch_angle":   [15.0] * n,
        "inning":         [1] * n,
        "inning_topbot":  ["Top"] * n,
    })


def test_every_canonical_bucket_reachable():
    """Every canonical outcome must be reachable from at least one statcast event."""
    reached = set(EVENT_TO_OUTCOME.values())
    assert reached == set(OUTCOMES), f"missing buckets: {set(OUTCOMES) - reached}"


def test_event_mapping_round_trip():
    """Every event in EVENT_TO_OUTCOME produces exactly one row with a valid outcome."""
    events = list(EVENT_TO_OUTCOME)
    df = transform_pitch_frame(_synthetic_pitches(events))
    assert len(df) == len(events)
    assert set(df["outcome"]) <= set(OUTCOMES)
    assert df["outcome"].isna().sum() == 0


def test_non_pa_events_dropped():
    """Caught stealing, pickoffs, balks etc. must not appear in the output."""
    events = ["single", "caught_stealing_2b", "pickoff_1b", "balk", "strikeout"]
    df = transform_pitch_frame(_synthetic_pitches(events))
    assert len(df) == 2
    assert set(df["outcome"]) == {"1B", "K"}
    assert not (df["events"].isin(NON_PA_EVENTS)).any()


def test_unmapped_event_dropped_with_warning(capsys):
    """Unknown event types are dropped (not coerced) and surfaced in stdout."""
    events = ["single", "made_up_event_xyz"]
    df = transform_pitch_frame(_synthetic_pitches(events))
    assert len(df) == 1
    assert df.iloc[0]["outcome"] == "1B"
    out = capsys.readouterr().out
    assert "made_up_event_xyz" in out


def test_null_events_dropped():
    """Pitch rows that don't terminate a PA (events is null) are excluded."""
    events = ["single", None, None, "home_run"]
    df = transform_pitch_frame(_synthetic_pitches(events))
    assert len(df) == 2
    assert list(df["outcome"]) == ["1B", "HR"]


def test_out_subtype_only_for_outs():
    """out_subtype is set for outs (and K_DP) and null for everything else."""
    events = ["single", "field_out", "grounded_into_double_play", "home_run", "walk"]
    df = transform_pitch_frame(_synthetic_pitches(events)).set_index("events")
    assert pd.isna(df.loc["single", "out_subtype"])
    assert pd.isna(df.loc["home_run", "out_subtype"])
    assert pd.isna(df.loc["walk", "out_subtype"])
    assert df.loc["field_out", "out_subtype"] == "field_out"
    assert df.loc["grounded_into_double_play", "out_subtype"] == "gidp"


def test_strikeouts_have_correct_subtype():
    """K_DP retains the K outcome but its subtype distinguishes it for the simulator."""
    events = ["strikeout", "strikeout_double_play"]
    df = transform_pitch_frame(_synthetic_pitches(events))
    assert (df["outcome"] == "K").all()
    sub = df.set_index("events")["out_subtype"]
    assert pd.isna(sub["strikeout"])
    assert sub["strikeout_double_play"] == "k_dp"


def test_schema_types():
    """Output schema has expected columns and types."""
    events = ["single", "home_run", "walk"]
    df = transform_pitch_frame(_synthetic_pitches(events))
    expected = {
        "game_pk", "game_date", "batter", "pitcher", "stand", "p_throws",
        "home_team", "away_team", "balls", "strikes", "events", "outcome",
        "out_subtype", "launch_speed", "launch_angle", "inning", "inning_topbot",
    }
    assert expected.issubset(df.columns)
    assert df["batter"].dtype == np.int64
    assert df["pitcher"].dtype == np.int64


def test_event_subtype_keys_are_subset_of_outcome_keys():
    """Every key in EVENT_TO_OUT_SUBTYPE must also be in EVENT_TO_OUTCOME (and map to OUT or K)."""
    extra = set(EVENT_TO_OUT_SUBTYPE) - set(EVENT_TO_OUTCOME)
    assert not extra, f"out_subtype maps unknown events: {extra}"
    for evt in EVENT_TO_OUT_SUBTYPE:
        assert EVENT_TO_OUTCOME[evt] in {"OUT", "K"}, (
            f"{evt} has subtype but outcome={EVENT_TO_OUTCOME[evt]}"
        )
