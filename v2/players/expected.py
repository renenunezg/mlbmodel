"""Statcast expected outcomes for hitters, as pseudo-counts behind the shrink's prior mean.

A hitter's plate appearances are re-expressed as the outcome counts his
underlying contact and plate-skill measurements imply: strikeouts and walks from
a same-season regression of K and BB rates on whiff, zone-contact, chase and
zone-swing rates; the in-play split (single, double, triple, home run, out)
from a smoothed table of outcome shares by exit velocity and launch angle;
hit-by-pitch and untracked batted balls at their actual outcome. The expected
counts are PA-sized, so one season of them weighs like one season of outcomes.

Both tables are fitted on the model's development seasons only. The counts
enter ``inputs.prior_rates`` exactly as the Triple-A pseudo-counts do: scaled by
``ValueModel.expected_scale`` they move the prior mean with weight
A / (strength + A) and never add to the player's sample. Scale zero switches
the path off, which is the served model's behaviour.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from v2.data.pa_dataset import EVENT_TO_OUTCOME, NON_PA_EVENTS, OUTCOMES
from v2.players import inputs

log = logging.getLogger(__name__)

PA_COLUMNS = ["player_id", "game_date", "outcome", "bip", "launch_speed", "launch_angle"]
IN_PLAY = ("1B", "2B", "3B", "HR", "OUT")
PLATE_INPUTS = ("contact.whiff_rate", "contact.zone_contact_rate", "discipline.chase_rate", "discipline.zone_swing_rate")
PLATE_RATES = {"K": "contact.k_rate", "BB": "discipline.bb_rate"}
MIN_PLATE_SAMPLE = 50.0  # plate appearances for a hitter-season to inform the K and BB regressions
# Exit velocity and launch angle cells of the in-play table, and the Gaussian
# kernel (in cells) that smooths neighbouring cells into each other.
EV_EDGES = np.arange(40.0, 122.0, 2.0)
LA_EDGES = np.arange(-60.0, 93.0, 3.0)
KERNEL_SD_CELLS = 1.0
TABLE_PRIOR = 2.0  # pseudo-observations of the league in-play split in every cell
SCALE_CANDIDATES = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)


def pa_events(season: int) -> pd.DataFrame:
    """One row per regular-season plate appearance of a hitter, cached at cache/players/pa_events_{season}.parquet."""
    path = inputs.CACHE_DIR / f"pa_events_{season}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frame = pa_events_from_pitches(inputs.load_pitches(season))
    inputs.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def pa_events_from_pitches(pitches: pd.DataFrame) -> pd.DataFrame:
    p = pitches[pitches["events"].notna() & ~pitches["events"].isin(NON_PA_EVENTS)]
    out = pd.DataFrame({
        "player_id": p["batter"].astype("int64"), "game_date": pd.to_datetime(p["game_date"]).dt.date,
        "outcome": p["events"].map(EVENT_TO_OUTCOME), "bip": p["description"].eq("hit_into_play"),
        "launch_speed": p["launch_speed"], "launch_angle": p["launch_angle"],
    })
    return out.dropna(subset=["outcome"]).reset_index(drop=True)[PA_COLUMNS]


# Fitting.


def _cells(ev: pd.Series, la: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(row, column, tracked) cell indices of batted balls; untracked balls get no cell."""
    tracked = ev.notna() & la.notna()
    r = np.clip(np.searchsorted(EV_EDGES, ev.fillna(EV_EDGES[0]).to_numpy(float), side="right") - 1, 0, len(EV_EDGES) - 2)
    c = np.clip(np.searchsorted(LA_EDGES, la.fillna(LA_EDGES[0]).to_numpy(float), side="right") - 1, 0, len(LA_EDGES) - 2)
    return r, c, tracked.to_numpy()


def _smooth(counts: np.ndarray) -> np.ndarray:
    """Gaussian smoothing over the two table axes, separably, per outcome layer."""
    radius = int(np.ceil(3 * KERNEL_SD_CELLS))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / KERNEL_SD_CELLS) ** 2)
    kernel /= kernel.sum()

    def along(a: np.ndarray, axis: int) -> np.ndarray:
        out = np.zeros_like(a)
        for k, o in zip(kernel, offsets, strict=True):
            out += k * np.roll(a, o, axis=axis) * _valid_shift(a.shape[axis], o, axis, a.ndim)
        return out

    return along(along(counts, 0), 1)


def _valid_shift(n: int, offset: int, axis: int, ndim: int) -> np.ndarray:
    """Mask that drops the wrap-around of np.roll at the table's edges."""
    mask = np.ones(n)
    if offset > 0:
        mask[:offset] = 0.0
    elif offset < 0:
        mask[offset:] = 0.0
    shape = [1] * ndim
    shape[axis] = n
    return mask.reshape(shape)


def fit_in_play_table(events_by_season: dict[int, pd.DataFrame]) -> dict:
    """Smoothed in-play outcome shares by exit velocity and launch angle cell, from the given seasons."""
    frame = pd.concat(events_by_season.values(), ignore_index=True)
    bip = frame[frame["bip"] & frame["outcome"].isin(IN_PLAY)]
    r, c, tracked = _cells(bip["launch_speed"], bip["launch_angle"])
    counts = np.zeros((len(EV_EDGES) - 1, len(LA_EDGES) - 1, len(IN_PLAY)))
    for k, outcome in enumerate(IN_PLAY):
        hit = (bip["outcome"].to_numpy() == outcome) & tracked
        np.add.at(counts[:, :, k], (r[hit], c[hit]), 1.0)
    league = counts.sum(axis=(0, 1))
    league /= league.sum()
    smoothed = _smooth(counts) + TABLE_PRIOR * league
    shares = smoothed / smoothed.sum(axis=2, keepdims=True)
    return {"ev_edges": EV_EDGES.tolist(), "la_edges": LA_EDGES.tolist(), "outcomes": list(IN_PLAY),
            "shares": np.round(shares, 5).tolist(), "batted_balls": int(tracked.sum()), "seasons": sorted(events_by_season)}


def fit_plate_rates(inputs_by_season: dict[int, pd.DataFrame], counts_by_season: dict[int, pd.DataFrame]) -> dict:
    """PA-weighted same-season regressions of K and BB rates on the plate-skill inputs, hitters with 50 PA."""
    rows = []
    for season, frame in inputs_by_season.items():
        bat = frame[frame["role"].eq("bat") & frame["input"].isin(PLATE_INPUTS)]
        wide = bat.pivot_table(index="player_id", columns="input", values="value", aggfunc="first")
        counts = counts_by_season[season]
        counts = counts[counts["role"].eq("bat")].drop_duplicates("player_id").set_index("player_id")
        n = counts[list(OUTCOMES)].sum(axis=1)
        wide = wide.reindex(n.index[n >= MIN_PLATE_SAMPLE]).dropna()
        wide["n"] = n.reindex(wide.index)
        for outcome in PLATE_RATES:
            wide[outcome] = counts.loc[wide.index, outcome] / wide["n"]
        rows.append(wide)
    data = pd.concat(rows)
    x = np.column_stack([np.ones(len(data)), data[list(PLATE_INPUTS)].to_numpy(float)])
    w = np.sqrt(data["n"].to_numpy(float))
    out: dict = {"inputs": list(PLATE_INPUTS), "hitter_seasons": int(len(data))}
    for outcome in PLATE_RATES:
        beta, *_ = np.linalg.lstsq(x * w[:, None], data[outcome].to_numpy(float) * w, rcond=None)
        out[outcome] = beta.tolist()
    return out


def fit(events_by_season: dict[int, pd.DataFrame], inputs_by_season: dict[int, pd.DataFrame], counts_by_season: dict[int, pd.DataFrame]) -> dict:
    return {"in_play": fit_in_play_table(events_by_season), "plate": fit_plate_rates(inputs_by_season, counts_by_season)}


# Expected counts.


def expected_counts(events: pd.DataFrame, season_inputs: pd.DataFrame, model: dict, season: int) -> pd.DataFrame:
    """Expected outcome counts per hitter in the ``outcome_counts`` layout (batting rows only).

    ``events`` are the plate appearances to re-express (a season, or a season
    cut by date) and ``season_inputs`` the long input frame built from the same
    pitches, which carries the plate-skill rates.
    """
    cols = list(OUTCOMES)
    actual = pd.crosstab(events["player_id"], events["outcome"]).reindex(columns=cols, fill_value=0).astype(float)
    pa = actual.sum(axis=1)
    # Strikeouts and walks from the plate-skill regression, clipped to the rates seen in the fit.
    plate = model["plate"]
    bat = season_inputs[season_inputs["role"].eq("bat") & season_inputs["input"].isin(plate["inputs"])]
    wide = bat.pivot_table(index="player_id", columns="input", values="value", aggfunc="first").reindex(actual.index)
    have = wide[plate["inputs"]].notna().all(axis=1)
    x = np.column_stack([np.ones(len(wide)), wide[plate["inputs"]].fillna(0.0).to_numpy(float)])
    expected = pd.DataFrame(0.0, index=actual.index, columns=cols)
    for outcome in ("K", "BB"):
        rate = pd.Series(np.clip(x @ np.array(plate[outcome]), 0.0, 1.0), index=actual.index)
        expected[outcome] = (rate * pa).where(have, actual[outcome])
    expected["HBP"] = actual["HBP"]
    # In-play split from the exit velocity and launch angle table; untracked balls keep their outcome.
    table = model["in_play"]
    shares = np.array(table["shares"])
    bip = events[events["bip"] & events["outcome"].isin(IN_PLAY)]
    r, c, tracked = _cells(bip["launch_speed"], bip["launch_angle"])
    per_ball = np.where(tracked[:, None], shares[r, c], 0.0)
    for k, outcome in enumerate(IN_PLAY):
        own = (bip["outcome"].to_numpy() == outcome) & ~tracked
        per_ball[own, k] = 1.0
    in_play = pd.DataFrame(per_ball, columns=list(IN_PLAY), index=bip["player_id"].to_numpy()).groupby(level=0).sum().reindex(actual.index).fillna(0.0)
    # Balls in play that were strikeouts or walks in expectation, or the reverse, rescale the in-play block
    # so the expected counts total the actual plate appearances.
    remainder = (pa - expected[["K", "BB", "HBP"]].sum(axis=1)).clip(lower=0.0)
    total_in_play = in_play.sum(axis=1)
    factor = (remainder / total_in_play.where(total_in_play > 0)).fillna(0.0)
    for outcome in IN_PLAY:
        expected[outcome] = in_play[outcome] * factor
    out = expected.reset_index().rename(columns={"index": "player_id"})
    out["player_id"] = out["player_id"].astype("int64")
    out.insert(1, "role", "bat")
    out["season"] = season
    return out[["player_id", "role", *cols, "season"]]
