"""Season totals per hitter, and the same totals split by pitch group, pitch type, attack zone, direction and count situation.

``scores`` holds one column per valuation of the same pitches: ``raw`` (what happened), ``expected`` (expected
contact), ``context`` (expected contact with park and season removed), ``neutral`` (context-neutral, park and
season removed), the components ``decision``, ``contact`` and ``quality``, ``call`` (called-strike luck on
takes, not credited) and ``faced`` (the league-average value of the pitches he saw).
"""
from __future__ import annotations

import pandas as pd

from v2.batting_value.components import COMPONENTS
from v2.data.pa_dataset import OUTCOMES
from v2.players import savant
from v2.players.runs import FULL_SEASON_PA

KEYS = ["season", "batter"]


def _totals(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    out = frame.groupby(by).agg(
        pitches=("neutral", "size"), swings=("swing", "sum"), balls_in_play=("in_play", "sum"),
        **{f"{name}_runs": (name, "sum") for name in ("raw", "expected", "context", "neutral", *COMPONENTS, "call", "faced")},
    ).reset_index()
    out["runs"] = out[[f"{name}_runs" for name in COMPONENTS]].sum(axis=1)
    out["per_100"] = 100.0 * out["runs"] / out["pitches"]
    return out


def _frame(pitches: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    frame = pitches[[*KEYS, "bat_team", "swing", "pitch_group", "pitch_type", "zone", "side", "height", "situation", "outcome",
                     "woba_value", "woba_denom"]].join(scores)
    return frame.assign(in_play=pitches["result"].eq("in_play"))[pitches["seen"]]


def names(seasons: list[int]) -> pd.Series:
    """(season, batter) -> 'First Last', from Savant's expected-statistics board (every hitter with a plate appearance)."""
    boards = pd.concat([savant.load_board("expected_statistics_batter", int(s)) for s in seasons])
    parts = boards["last_name, first_name"].str.split(", ", n=1, expand=True)
    name = (parts[1] + " " + parts[0]).where(parts[1].notna(), boards["last_name, first_name"])
    return pd.Series(name.to_numpy(), index=pd.MultiIndex.from_frame(boards[["season", "player_id"]]))


def hitter_seasons(pitches: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """One row per hitter and season, best total first, with the plate-appearance outcomes the validation needs."""
    frame = _frame(pitches, scores)
    out = _totals(frame, KEYS)
    who = frame.groupby(KEYS).agg(team=("bat_team", lambda t: t.value_counts().index[0]), pa=("outcome", "count"),
                                  woba_value=("woba_value", "sum"), woba_denom=("woba_denom", "sum")).reset_index()
    outcomes = pd.crosstab([frame["season"], frame["batter"]], frame["outcome"]).reindex(columns=list(OUTCOMES), fill_value=0)
    out = who.merge(out, on=KEYS).join(outcomes, on=KEYS)
    out[list(OUTCOMES)] = out[list(OUTCOMES)].fillna(0).astype(int)
    name = names(sorted(out["season"].unique())).reindex(pd.MultiIndex.from_frame(out[KEYS])).to_numpy()
    out.insert(2, "name", pd.Series(name, index=out.index).fillna(out["batter"].astype(str)))
    out["per_600"] = FULL_SEASON_PA * out["runs"] / out["pa"].where(out["pa"] > 0)
    return out.sort_values(["season", "runs"], ascending=[True, False]).reset_index(drop=True)


def splits(pitches: pd.DataFrame, scores: pd.DataFrame, by: str) -> pd.DataFrame:
    """One row per hitter, season and value of ``by`` (pitch_group, pitch_type, zone, side, height or situation); pitches
    with no value of ``by`` (untracked, a pitch type outside the groups, or the middle third of a direction) are left out."""
    out = _totals(_frame(pitches, scores).dropna(subset=[by]), [*KEYS, by])
    return out.sort_values(["season", "runs"], ascending=[True, False]).reset_index(drop=True)
