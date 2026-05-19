"""Per-batter and per-pitcher groundball-rate quartiles.

GB% = ground_ball / balls-in-play, computed on PA-terminating statcast rows
(bb_type non-null). Players below MIN_BIP get the median bin (q=2) so a thin
sample can't land in an extreme quartile and skew the out-subtype mix.

Quartile ids 0..3 (0 = lowest GB%, 3 = highest). Unknown players resolve to 2.

Output: v2/simulator/tables/gb_quartiles.parquet (player_id, role, gb_q)
Run as part of: build_advancement_table.py (called there), or standalone:
    env/bin/python -m v2.simulator.gb_quartiles --years 2024 2025
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from v2.data.pa_dataset import NON_PA_EVENTS

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"
TABLES_DIR = Path(__file__).resolve().parent / "tables"

BIP_TYPES = ("ground_ball", "fly_ball", "line_drive", "popup")
MIN_BIP = 100          # below this, assign the median quartile
MEDIAN_Q = 2           # fallback bin for thin / unknown players
N_QUARTILES = 4


def _gb_rate_by(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Return per-id (gb, bip, gb_pct) for balls in play."""
    bip = df[df["bb_type"].isin(BIP_TYPES)]
    g = bip.groupby(id_col)
    out = pd.DataFrame({
        "bip": g.size(),
        "gb": g["bb_type"].apply(lambda s: (s == "ground_ball").sum()),
    })
    out["gb_pct"] = out["gb"] / out["bip"]
    return out.reset_index().rename(columns={id_col: "player_id"})


def _assign_quartiles(rates: pd.DataFrame, role: str) -> pd.DataFrame:
    qualified = rates[rates["bip"] >= MIN_BIP].copy()
    # qcut on qualified players only; everyone else (and ties beyond bins) → median
    qualified["gb_q"] = pd.qcut(
        qualified["gb_pct"], N_QUARTILES, labels=False, duplicates="drop"
    ).astype(np.int64)
    rates = rates.merge(qualified[["player_id", "gb_q"]], on="player_id", how="left")
    rates["gb_q"] = rates["gb_q"].fillna(MEDIAN_Q).astype(np.int64)
    rates["role"] = role
    return rates[["player_id", "role", "gb_q"]]


def build_gb_quartiles(years: list[int]) -> pd.DataFrame:
    frames = []
    for y in years:
        path = CACHE_DIR / f"statcast_{y}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_parquet(path, columns=["events", "bb_type", "batter", "pitcher"]))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["events"].notna() & ~df["events"].isin(NON_PA_EVENTS)]

    bat = _assign_quartiles(_gb_rate_by(df, "batter"), "B")
    pit = _assign_quartiles(_gb_rate_by(df, "pitcher"), "P")
    return pd.concat([bat, pit], ignore_index=True)


@dataclass
class GBQuartiles:
    batter: dict[int, int]
    pitcher: dict[int, int]

    def batter_q(self, ids: np.ndarray) -> np.ndarray:
        return np.array([self.batter.get(int(i), MEDIAN_Q) for i in ids], dtype=np.int64)

    def pitcher_q(self, ids: np.ndarray) -> np.ndarray:
        return np.array([self.pitcher.get(int(i), MEDIAN_Q) for i in ids], dtype=np.int64)


def load_gb_quartiles() -> GBQuartiles:
    df = pd.read_parquet(TABLES_DIR / "gb_quartiles.parquet")
    bat = dict(zip(df.loc[df.role == "B", "player_id"].astype(int),
                   df.loc[df.role == "B", "gb_q"].astype(int)))
    pit = dict(zip(df.loc[df.role == "P", "player_id"].astype(int),
                   df.loc[df.role == "P", "gb_q"].astype(int)))
    return GBQuartiles(batter=bat, pitcher=pit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2024, 2025])
    args = ap.parse_args()
    df = build_gb_quartiles(args.years)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TABLES_DIR / "gb_quartiles.parquet", index=False)
    for role, name in (("B", "batters"), ("P", "pitchers")):
        sub = df[df.role == role]
        dist = sub["gb_q"].value_counts().sort_index().to_dict()
        print(f"  {name}: {len(sub):,}  quartile counts {dist}")
    print(f"Wrote {TABLES_DIR}/gb_quartiles.parquet")


if __name__ == "__main__":
    main()
