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
from pathlib import Path

import numpy as np
import pandas as pd

from v2.data.pa_dataset import (
    EVENT_TO_OUTCOME,
    EVENT_TO_OUT_SUBTYPE,
    NON_PA_EVENTS,
    OUTCOMES,
)
from v2.simulator.gb_quartiles import MEDIAN_Q, build_gb_quartiles

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


def _load_pa_rows(years: list[int]) -> pd.DataFrame:
    """Load PA-terminating rows with state/runs/outs deltas resolved."""
    frames = []
    for y in years:
        path = CACHE_DIR / f"statcast_{y}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        cols = [
            "game_pk", "at_bat_number", "pitch_number",
            "inning", "inning_topbot",
            "events", "outs_when_up",
            "on_1b", "on_2b", "on_3b",
            "bat_score", "post_bat_score",
            "batter", "pitcher",
        ]
        frames.append(pd.read_parquet(path, columns=cols))
    df = pd.concat(frames, ignore_index=True)

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
               "new_state", "runs", "outs_added", "batter", "pitcher"]]


def build_advancement(df: pd.DataFrame) -> pd.DataFrame:
    """P(new_state, runs, outs_added | state, outs, outcome, subtype).

    HR rows are deterministic (override in _load_pa_rows) and bypass smoothing.
    """
    df = df.copy()
    deterministic_mask = df["outcome_idx"].isin([HR_IDX, BB_IDX, HBP_IDX])
    df_smooth = df[~deterministic_mask]

    cells = (
        df_smooth.groupby(["state", "outs", "outcome_idx", "subtype_key", "new_state", "runs", "outs_added"])
        .size().rename("n").reset_index()
    )
    totals = cells.groupby(["state", "outs", "outcome_idx", "subtype_key"])["n"].transform("sum")
    cells["prob"] = cells["n"] / totals
    cells["cell_n"] = totals

    # Smoothing: marginal P(new_state, runs, outs_added | outcome, subtype) when cell_n < ADVANCEMENT_MIN_OBS.
    marg = (
        df_smooth.groupby(["outcome_idx", "subtype_key", "new_state", "runs", "outs_added"])
        .size().rename("m_n").reset_index()
    )
    marg_total = marg.groupby(["outcome_idx", "subtype_key"])["m_n"].transform("sum")
    marg["m_prob"] = marg["m_n"] / marg_total

    # cells with low n → mix toward marginal. Build full grid keyed by (state, outs, outcome, subtype).
    keys = df_smooth[["state", "outs", "outcome_idx", "subtype_key"]].drop_duplicates().reset_index(drop=True)
    rows = []
    for _, k in keys.iterrows():
        cell = cells[
            (cells["state"] == k.state) & (cells["outs"] == k.outs)
            & (cells["outcome_idx"] == k.outcome_idx) & (cells["subtype_key"] == k.subtype_key)
        ]
        n_cell = int(cell["cell_n"].iloc[0]) if len(cell) else 0
        m = marg[(marg["outcome_idx"] == k.outcome_idx) & (marg["subtype_key"] == k.subtype_key)]
        if n_cell >= ADVANCEMENT_MIN_OBS:
            for _, r in cell.iterrows():
                rows.append((k.state, k.outs, k.outcome_idx, k.subtype_key,
                             int(r.new_state), int(r.runs), int(r.outs_added), float(r.prob)))
        else:
            # blend: w = n_cell / ADVANCEMENT_MIN_OBS toward cell, rest toward marginal
            w = n_cell / ADVANCEMENT_MIN_OBS
            mix = {}
            for _, r in cell.iterrows():
                mix[(int(r.new_state), int(r.runs), int(r.outs_added))] = w * float(r.prob)
            for _, r in m.iterrows():
                key = (int(r.new_state), int(r.runs), int(r.outs_added))
                mix[key] = mix.get(key, 0.0) + (1 - w) * float(r.m_prob)
            tot = sum(mix.values())
            for (ns, ru, oa), p in mix.items():
                rows.append((k.state, k.outs, k.outcome_idx, k.subtype_key, ns, ru, oa, p / tot))
    out = pd.DataFrame(rows, columns=[
        "state", "outs", "outcome_idx", "subtype_key", "new_state", "runs", "outs_added", "prob",
    ])

    # Append deterministic rows for HR / BB / HBP. These are structurally constrained
    # and don't deserve smoothing.
    det_rows = []
    for st in range(8):
        for ou in range(3):
            # HR: clears bases, runs = 1 + popcount(state)
            det_rows.append((st, ou, HR_IDX, "_NA_", 0, 1 + bin(st).count("1"), 0, 1.0))
            # BB / HBP: forced advance, no outs
            new_st, r = _walk_advance(st)
            det_rows.append((st, ou, BB_IDX, "_NA_", new_st, r, 0, 1.0))
            det_rows.append((st, ou, HBP_IDX, "_NA_", new_st, r, 0, 1.0))
    out = pd.concat([out, pd.DataFrame(det_rows, columns=out.columns)], ignore_index=True)
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
    l1_n = {(s, ou): int(g["n"].sum()) for (s, ou), g in l1.groupby(["state", "outs"])}
    l1_dist = {(s, ou): _dist(g) for (s, ou), g in l1.groupby(["state", "outs"])}

    l2 = o.groupby(["outs", "subtype_key"]).size().rename("n").reset_index()
    l2_dist = {ou: _dist(g) for ou, g in l2.groupby("outs")}

    keys = o[["state", "outs", "b_q", "p_q"]].drop_duplicates().reset_index(drop=True)
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
        if l1_n.get((k.state, k.outs), 0) >= SUBTYPE_MIN_OBS:
            fallback = l1_dist[(k.state, k.outs)]
        else:
            fallback = l2_dist[k.outs]
        w = n_cell / SUBTYPE_MIN_OBS
        mix: dict[str, float] = {}
        for _, r in c.iterrows():
            mix[r.subtype_key] = w * float(r.prob)
        for st, p in fallback.items():
            mix[st] = mix.get(st, 0.0) + (1 - w) * p
        tot = sum(mix.values())
        for st, p in mix.items():
            rows.append((k.state, k.outs, k.b_q, k.p_q, st, p / tot))
    return pd.DataFrame(rows, columns=["state", "outs", "b_q", "p_q", "subtype_key", "prob"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2024, 2025])
    args = ap.parse_args()

    print(f"Loading PA rows from years {args.years} ...")
    df = _load_pa_rows(args.years)
    print(f"  {len(df):,} PAs after filtering")

    runs_per_pa = df["runs"].mean()
    print(f"  sanity: runs per PA = {runs_per_pa:.4f}  (MLB norm ~0.12)")

    print("Building GB quartiles ...")
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
    print(f"  runs/PA by pitcher GB quartile (0=low GB .. 3=high GB): {rp}")

    print("Building advancement table ...")
    adv = build_advancement(df)
    print(f"  {len(adv):,} rows")

    print("Building out-subtype table ...")
    subt = build_out_subtype(df)
    print(f"  {len(subt):,} rows")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    adv.to_parquet(TABLES_DIR / "advancement.parquet", index=False)
    subt.to_parquet(TABLES_DIR / "out_subtype.parquet", index=False)
    print(f"Wrote {TABLES_DIR}/advancement.parquet and out_subtype.parquet")


if __name__ == "__main__":
    main()
