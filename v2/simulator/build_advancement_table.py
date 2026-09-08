"""Build empirical advancement + out-subtype tables from the 2024+25 statcast cache.

Output:
    v2/simulator/tables/advancement.parquet  (state, outs, outcome, out_subtype, new_state, runs, prob)
    v2/simulator/tables/out_subtype.parquet  (state, outs, out_subtype, prob)

Run:
    env/bin/python -m v2.simulator.build_advancement_table --years 2024 2025

Known biases (documented for the gate-failure debug order):
- pre-state read from terminating pitch, so mid-AB stolen bases shift it (~0.5% of PAs)
- outs_added = next_outs - outs, so a CS between PAs inflates the prior PA's outs (~0.3%)
- wild pitches / passed balls / steals between PAs advance runners; not modeled
- HR is hardcoded to runs = 1 + popcount(state), ignoring inside-the-park edge cases
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from backend.log import setup_logging
from v2.data.pa_dataset import (
    EVENT_TO_OUT_SUBTYPE,
    EVENT_TO_OUTCOME,
    NON_PA_EVENTS,
    OUTCOMES,
)
from v2.simulator.baserunner import SUBTYPE_ORDER, TABLE_VERSION, legal_transitions
from v2.simulator.gb_quartiles import MEDIAN_Q, build_gb_quartiles

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"
TABLES_DIR = Path(__file__).resolve().parent / "tables"

ADVANCEMENT_MIN_OBS = 100
SUBTYPE_MIN_OBS = 50

OUTCOME_TO_IDX = {o: i for i, o in enumerate(OUTCOMES)}
OUT_IDX = OUTCOME_TO_IDX["OUT"]
HR_IDX = OUTCOME_TO_IDX["HR"]
BB_IDX = OUTCOME_TO_IDX["BB"]
HBP_IDX = OUTCOME_TO_IDX["HBP"]


_WALK_LOOKUP = {
    0: (1, 0),   # empty → 1B
    1: (3, 0),   # 1B → 1B+2B
    2: (3, 0),   # 2B → 1B+2B (runner stays at 2B)
    3: (7, 0),   # 1B+2B → loaded
    4: (5, 0),   # 3B → 1B+3B
    5: (7, 0),   # 1B+3B → loaded
    6: (7, 0),   # 2B+3B → loaded (no force on 2B/3B since 1B was empty)
    7: (7, 1),   # loaded → loaded, forces 1 run
}


def _walk_advance(state: int) -> tuple[int, int]:
    """Forced advance for BB / HBP. Returns (new_state, runs_scored)."""
    return _WALK_LOOKUP[state]


def _state_from_runners(on_1b: pd.Series, on_2b: pd.Series, on_3b: pd.Series) -> np.ndarray:
    b1 = on_1b.notna().astype(np.int64).to_numpy()
    b2 = on_2b.notna().astype(np.int64).to_numpy()
    b3 = on_3b.notna().astype(np.int64).to_numpy()
    return b1 | (b2 << 1) | (b3 << 2)


def _load_pa_rows(years: list[int], before: str | None = None) -> pd.DataFrame:
    """Load PA-terminating rows with state/runs/outs deltas resolved."""
    frames = []
    for y in years:
        path = CACHE_DIR / f"statcast_{y}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        cols = [
            "game_pk", "game_date", "at_bat_number", "pitch_number",
            "inning", "inning_topbot",
            "events", "outs_when_up",
            "on_1b", "on_2b", "on_3b",
            "bat_score", "post_bat_score",
            "batter", "pitcher",
        ]
        frames.append(pd.read_parquet(path, columns=cols))
    df = pd.concat(frames, ignore_index=True)

    if before is not None:
        df = df[pd.to_datetime(df["game_date"]) < pd.Timestamp(before)]

    # one row per PA: keep only the terminating pitch (events not null and not non-PA).
    df = df[df["events"].notna()]
    df = df[~df["events"].isin(NON_PA_EVENTS)]
    df["outcome"] = df["events"].map(EVENT_TO_OUTCOME)
    df = df[df["outcome"].notna()].copy()
    df["out_subtype"] = df["events"].map(EVENT_TO_OUT_SUBTYPE)

    # sort by half-inning + at_bat_number to derive next-PA state
    df = df.sort_values(["game_pk", "inning", "inning_topbot", "at_bat_number"]).reset_index(drop=True)

    df["state"] = _state_from_runners(df["on_1b"], df["on_2b"], df["on_3b"])
    df["outs"] = df["outs_when_up"].fillna(0).astype(np.int64)
    raw_runs = (df["post_bat_score"].fillna(0) - df["bat_score"].fillna(0))
    df["runs"] = raw_runs.clip(lower=0, upper=4).astype(np.int64)

    grp = df.groupby(["game_pk", "inning", "inning_topbot"], sort=False)
    df["next_state"] = grp["state"].shift(-1)
    df["next_outs"] = grp["outs"].shift(-1)

    # if next-PA exists in same half-inning, use its pre-state; else inning ended.
    inning_ends = df["next_state"].isna()
    df["new_state"] = np.where(inning_ends, 0, df["next_state"]).astype(np.int64)
    df["outs_added"] = np.where(inning_ends, 3 - df["outs"], df["next_outs"] - df["outs"]).astype(np.int64)

    # HR override (deterministic; ignores empirical noise on rare bases-loaded states).
    is_hr = df["outcome"] == "HR"
    df.loc[is_hr, "new_state"] = 0
    df.loc[is_hr, "runs"] = 1 + df.loc[is_hr, "state"].apply(lambda s: bin(int(s)).count("1")).astype(np.int64)
    df.loc[is_hr, "outs_added"] = 0

    df["outcome_idx"] = df["outcome"].map(OUTCOME_TO_IDX).astype(np.int64)
    # subtype only meaningful when outcome=OUT; for non-OUT we use a single sentinel "_NA_"
    df["subtype_key"] = np.where(df["outcome_idx"] == OUT_IDX, df["out_subtype"].fillna("field_out"), "_NA_")

    df["batter"] = df["batter"].astype(np.int64)
    df["pitcher"] = df["pitcher"].astype(np.int64)

    # filter degenerate rows (negative outs or outs_added > 3, sometimes from data quirks)
    df = df[(df["outs_added"].between(0, 3)) & (df["runs"].between(0, 4))].reset_index(drop=True)
    return df[["state", "outs", "outcome_idx", "subtype_key",
               "new_state", "runs", "outs_added", "batter", "pitcher", "game_date"]]


def _neutral_transition(state: int, outs: int, outcome: int) -> tuple[int, int, int]:
    if outcome == HR_IDX:
        return 0, 1 + state.bit_count(), 0
    if outcome in (BB_IDX, HBP_IDX):
        ns, runs = _walk_advance(state)
        return ns, runs, 0
    if OUTCOMES[outcome] in ("1B", "2B", "3B"):
        bases = {"1B": 1, "2B": 2, "3B": 3}[OUTCOMES[outcome]]
        shifted = state << bases
        return (shifted & 7) | (1 << (bases - 1)), (shifted >> 3).bit_count(), 0
    return (0 if outs == 2 else state), 0, 1


def build_advancement(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink within the same base/out state, with physically legal support only."""
    df = df.loc[legal_transitions(df)].copy()
    group_cols = ["state", "outs", "outcome_idx"]
    pooled = {key: g for key, g in df.groupby(group_cols)}
    cells = {key: g for key, g in df.groupby([*group_cols, "subtype_key"])}
    result = []
    for state in range(8):
        for outs in range(3):
            for outcome in range(len(OUTCOMES)):
                subtypes = SUBTYPE_ORDER[1:] if outcome == OUT_IDX else ("_NA_",)
                for subtype in subtypes:
                    deterministic = outcome in (HR_IDX, BB_IDX, HBP_IDX) or subtype == "ci"
                    if deterministic:
                        key = _neutral_transition(state, outs, BB_IDX if subtype == "ci" else outcome)
                        distribution = {key: 1.0}
                    else:
                        cell = cells.get((state, outs, outcome, subtype), df.iloc[:0])
                        marginal = pooled.get((state, outs, outcome), df.iloc[:0])
                        weight = min(len(cell) / ADVANCEMENT_MIN_OBS, 1.0)
                        distribution = {}
                        for rows, fraction in ((cell, weight), (marginal, 1.0 - weight)):
                            if len(rows) and fraction:
                                counts = rows.groupby(["new_state", "runs", "outs_added"]).size()
                                for key, count in counts.items():
                                    distribution[key] = distribution.get(key, 0.0) + fraction * count / len(rows)
                        if not distribution:
                            distribution = {_neutral_transition(state, outs, outcome): 1.0}
                    total = sum(distribution.values())
                    for (ns, runs, added), probability in distribution.items():
                        result.append((state, outs, outcome, subtype, ns, runs, added, probability / total))
    out = pd.DataFrame(result, columns=[
        "state", "outs", "outcome_idx", "subtype_key", "new_state", "runs", "outs_added", "prob",
    ])
    if not legal_transitions(out).all():
        raise ValueError("Impossible advancement transition after smoothing")
    out["table_version"] = TABLE_VERSION
    if "game_date" in df:
        out["training_max_date"] = str(pd.to_datetime(df["game_date"]).max().date())
    return out


def _dist(g: pd.DataFrame) -> dict[str, float]:
    n = g["n"].sum()
    return {r.subtype_key: r.n / n for r in g.itertuples()}


def build_out_subtype(df: pd.DataFrame) -> pd.DataFrame:
    """P(out_subtype | state, outs, batter_gb_q, pitcher_gb_q) for outcome=OUT.

    Two-level shrinkage on thin cells:
      L0 cell    (state, outs, b_q, p_q)  - the stratified target
      L1 marginal (state, outs)           - drops quartiles, keeps base/out context
      L2 marginal (outs)                  - coarse backstop when L1 itself is thin

    Thin extreme-quartile cells shrink toward the neutral (state, outs) league
    rate, so a small high-GB matchup cell can't manufacture an extreme GIDP rate.
    """
    o = df[df["outcome_idx"] == OUT_IDX].copy()

    cell = o.groupby(["state", "outs", "b_q", "p_q", "subtype_key"]).size().rename("n").reset_index()
    cell_n = cell.groupby(["state", "outs", "b_q", "p_q"])["n"].transform("sum")
    cell["prob"] = cell["n"] / cell_n
    cell["cell_n"] = cell_n

    l1 = o.groupby(["state", "outs", "subtype_key"]).size().rename("n").reset_index()
    l1_dist = {(s, ou): _dist(g) for (s, ou), g in l1.groupby(["state", "outs"])}


    keys = pd.DataFrame(
        [(s, ou, b, p) for s in range(8) for ou in range(3) for b in range(4) for p in range(4)],
        columns=["state", "outs", "b_q", "p_q"],
    )
    rows = []
    for _, k in keys.iterrows():
        c = cell[
            (cell["state"] == k.state) & (cell["outs"] == k.outs)
            & (cell["b_q"] == k.b_q) & (cell["p_q"] == k.p_q)
        ]
        n_cell = int(c["cell_n"].iloc[0]) if len(c) else 0
        if n_cell >= SUBTYPE_MIN_OBS:
            for _, r in c.iterrows():
                rows.append((k.state, k.outs, k.b_q, k.p_q, r.subtype_key, float(r.prob)))
            continue
        # pick the fallback: L1 if it has enough obs, else the coarse L2.
        fallback = l1_dist.get((k.state, k.outs), {"field_out": 1.0})
        w = n_cell / SUBTYPE_MIN_OBS
        mix: dict[str, float] = {}
        for _, r in c.iterrows():
            mix[r.subtype_key] = w * float(r.prob)
        for st, p in fallback.items():
            mix[st] = mix.get(st, 0.0) + (1 - w) * p
        tot = sum(mix.values())
        for st, p in mix.items():
            rows.append((k.state, k.outs, k.b_q, k.p_q, st, p / tot))
    result = pd.DataFrame(rows, columns=["state", "outs", "b_q", "p_q", "subtype_key", "prob"])
    result["table_version"] = TABLE_VERSION
    if "game_date" in df:
        result["training_max_date"] = str(pd.to_datetime(df["game_date"]).max().date())
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2024, 2025])
    args = ap.parse_args()

    log.info(f"Loading PA rows from years {args.years} ...")
    df = _load_pa_rows(args.years)
    log.info(f"{len(df):,} PAs after filtering")

    runs_per_pa = df["runs"].mean()
    log.info(f"sanity: runs per PA = {runs_per_pa:.4f}  (MLB norm ~0.12)")

    log.info("Building GB quartiles ...")
    gbq = build_gb_quartiles(args.years)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    gbq.to_parquet(TABLES_DIR / "gb_quartiles.parquet", index=False)
    bat_map = dict(zip(gbq.loc[gbq.role == "B", "player_id"], gbq.loc[gbq.role == "B", "gb_q"]))
    pit_map = dict(zip(gbq.loc[gbq.role == "P", "player_id"], gbq.loc[gbq.role == "P", "gb_q"]))
    df["b_q"] = df["batter"].map(bat_map).fillna(MEDIAN_Q).astype(np.int64)
    df["p_q"] = df["pitcher"].map(pit_map).fillna(MEDIAN_Q).astype(np.int64)
    # mean-conservation sanity: runs/PA by pitcher GB quartile (should rise as
    # GB% drops; the stratification must not crush runs in the high-GB bin).
    rp = df.groupby("p_q")["runs"].mean().round(4).to_dict()
    log.info(f"runs/PA by pitcher GB quartile (0=low GB .. 3=high GB): {rp}")

    log.info("Building advancement table ...")
    adv = build_advancement(df)
    log.info(f"{len(adv):,} rows")

    log.info("Building out-subtype table ...")
    subt = build_out_subtype(df)
    log.info(f"{len(subt):,} rows")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    adv.to_parquet(TABLES_DIR / "advancement.parquet", index=False)
    subt.to_parquet(TABLES_DIR / "out_subtype.parquet", index=False)
    log.info(f"Wrote {TABLES_DIR}/advancement.parquet and out_subtype.parquet")


if __name__ == "__main__":
    setup_logging()
    main()
