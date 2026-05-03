"""Per-PA dataset loader for v2 Bayesian fits.

Reads pitch-level Statcast parquet (shared with v1), filters to PA-terminating
events, and emits one row per plate appearance with two outcome columns:
  - `outcome`     - 8-way bucket (K/BB/HBP/1B/2B/3B/HR/OUT) for skill fits.
  - `out_subtype` - productive vs destructive sub-bucket when OUT, for the
                    simulator's baserunner advancement logic.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"

OUTCOMES = ("K", "BB", "HBP", "1B", "2B", "3B", "HR", "OUT")

# All outs collapse to OUT here; productive/destructive distinction lives in
# out_subtype below so the simulator can advance baserunners correctly.
EVENT_TO_OUTCOME = {
    "single": "1B",
    "double": "2B",
    "triple": "3B",
    "home_run": "HR",
    "strikeout": "K",
    "strikeout_double_play": "K",
    "walk": "BB",
    "intent_walk": "BB",
    "hit_by_pitch": "HBP",
    "field_out": "OUT",
    "force_out": "OUT",
    "grounded_into_double_play": "OUT",
    "double_play": "OUT",
    "triple_play": "OUT",
    "fielders_choice": "OUT",
    "fielders_choice_out": "OUT",
    "sac_fly": "OUT",
    "sac_fly_double_play": "OUT",
    "sac_bunt": "OUT",
    "sac_bunt_double_play": "OUT",
    "field_error": "OUT",
    "catcher_interf": "OUT",
}

EVENT_TO_OUT_SUBTYPE = {
    "field_out": "field_out",
    "force_out": "force_out",
    "grounded_into_double_play": "gidp",
    "double_play": "dp",
    "triple_play": "tp",
    "fielders_choice": "fc",
    "fielders_choice_out": "fc",
    "sac_fly": "sac_fly",
    "sac_fly_double_play": "sac_fly_dp",
    "sac_bunt": "sac_bunt",
    "sac_bunt_double_play": "sac_bunt_dp",
    "field_error": "roe",
    "catcher_interf": "ci",
    "strikeout_double_play": "k_dp",
}

# PA-terminating-looking but not real plate appearances; dropped.
NON_PA_EVENTS = {
    "caught_stealing_2b", "caught_stealing_3b", "caught_stealing_home",
    "pickoff_1b", "pickoff_2b", "pickoff_3b",
    "pickoff_caught_stealing_2b", "pickoff_caught_stealing_3b",
    "pickoff_caught_stealing_home",
    "stolen_base_2b", "stolen_base_3b", "stolen_base_home",
    "wild_pitch", "passed_ball", "balk",
    "other_advance", "runner_double_play", "batter_interference",
    "game_advisory",
    "truncated_pa",  # PA cut short by inning rollover; not a real outcome
}

PA_FRAME_COLS = [
    "game_pk", "game_date", "batter", "pitcher",
    "stand", "p_throws", "home_team", "away_team",
    "balls", "strikes", "events", "outcome", "out_subtype",
    "launch_speed", "launch_angle",
    "inning", "inning_topbot",
]


def _load_year(year: int) -> pd.DataFrame:
    path = CACHE_DIR / f"statcast_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Populate via the v1 pipeline first or run "
            f"`python -m v2.data.build_cache --years {year}`."
        )
    return pd.read_parquet(path)


def transform_pitch_frame(pitch_df: pd.DataFrame) -> pd.DataFrame:
    """Pure transform - kept separate from disk I/O for synthetic-data tests."""
    pa = pitch_df[pitch_df["events"].notna()].copy()
    pa = pa[~pa["events"].isin(NON_PA_EVENTS)]

    pa["outcome"] = pa["events"].map(EVENT_TO_OUTCOME)
    pa["out_subtype"] = pa["events"].map(EVENT_TO_OUT_SUBTYPE)

    unmapped = pa[pa["outcome"].isna()]["events"].value_counts()
    if not unmapped.empty:
        print(f"  pa_dataset: dropping {int(unmapped.sum())} rows with unmapped events:")
        for evt, n in unmapped.items():
            print(f"    {evt}: {n}")
        pa = pa[pa["outcome"].notna()]

    pa = pa[PA_FRAME_COLS].reset_index(drop=True)
    pa["game_date"] = pd.to_datetime(pa["game_date"]).dt.date
    pa["batter"] = pa["batter"].astype("int64")
    pa["pitcher"] = pa["pitcher"].astype("int64")
    return pa


def load_pa_dataset(start_year: int, end_year: int) -> pd.DataFrame:
    """Load PA-level frame for [start_year, end_year] inclusive.

    Returns one row per plate appearance with canonical outcome + out_subtype.
    """
    if end_year < start_year:
        raise ValueError(f"end_year ({end_year}) < start_year ({start_year})")

    frames = [_load_year(y) for y in range(start_year, end_year + 1)]
    pitch_df = pd.concat(frames, ignore_index=True)
    return transform_pitch_frame(pitch_df)
