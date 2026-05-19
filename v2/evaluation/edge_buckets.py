"""Is the totals bleed an EV-threshold/selection problem or a modeling problem?

Grades EVERY candidate bet (no edge floor) for a window, buckets by model
edge, and reports flat-stake ROI per (market, edge bucket). If ROI rises
monotonically with edge and is positive at high edge, the fix is the threshold
(raise it). If every bucket is negative regardless of edge, it's the model.

    env/bin/python -m v2.evaluation.edge_buckets --start 2026-03-26 --end 2026-05-09
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sqlalchemy import text

from backend.db import engine
from v2.evaluation.backtester import _PRED_COLS, _load_games
from v2.evaluation.metrics import _american_to_decimal, _american_to_implied, attach_actuals

BUCKETS = [-1, 0.0, 0.02, 0.045, 0.065, 0.10, 0.15, 1.0]


def _candidates(e: pd.DataFrame) -> pd.DataFrame:
    """All gradeable bets with edge + flat-stake P/L, no edge floor applied."""
    e = e.copy()
    for c in ("win_prob", "p_cover", "p_over", "p_under", "moneyline", "spread",
              "spread_odds", "total", "total_over_odds", "total_under_odds"):
        e[c] = pd.to_numeric(e[c], errors="coerce")
    won_ml = (e["actual_win"].astype(int) == 1)
    mg = e["actual_margin"].astype(float)
    rows = []

    m = e[(e["ev_flag"] == e["team"]) & e["moneyline"].notna() & e["win_prob"].notna()].copy()
    m["edge"] = m["win_prob"] - _american_to_implied(m["moneyline"].to_numpy())
    m["dec"] = _american_to_decimal(m["moneyline"].to_numpy())
    m["won"] = won_ml.loc[m.index]
    m["bet_type"] = "ml"
    rows.append(m[["bet_type", "edge", "dec", "won"]])

    r = e[(e["run_line_ev_flag"] == e["team"]) & e["spread"].notna()
          & e["spread_odds"].notna() & e["p_cover"].notna()].copy()
    sp = r["spread"].astype(float); rm = mg.loc[r.index]
    r["won"] = np.where(sp < 0, rm >= sp.abs(), np.where(sp > 0, (rm > 0) | (rm >= -sp), False))
    r["edge"] = r["p_cover"] - _american_to_implied(r["spread_odds"].to_numpy())
    r["dec"] = _american_to_decimal(r["spread_odds"].to_numpy())
    r["bet_type"] = "rl"
    rows.append(r[["bet_type", "edge", "dec", "won"]])

    t = e[e["total_play"].isin(["Over", "Under"]) & e["total"].notna()
          & e["game_total"].notna()].copy()
    over = t["total_play"] == "Over"
    t["p_sel"] = np.where(over, t["p_over"], t["p_under"])
    t["odds_sel"] = np.where(over, t["total_over_odds"], t["total_under_odds"])
    t = t[t["p_sel"].notna() & t["odds_sel"].notna()].copy()
    gt = t["game_total"].astype(float); tl = t["total"].astype(float)
    t["won"] = np.where(over.loc[t.index], gt > tl, gt < tl)
    t["edge"] = t["p_sel"] - _american_to_implied(t["odds_sel"].to_numpy())
    t["dec"] = _american_to_decimal(t["odds_sel"].to_numpy())
    t = t.sort_values(["game_pk", "team"]).drop_duplicates("game_pk", keep="first")
    t["bet_type"] = "total"
    rows.append(t[["bet_type", "edge", "dec", "won"]])

    led = pd.concat(rows, ignore_index=True)
    led["won"] = led["won"].astype(bool)
    led["pnl"] = np.where(led["won"], led["dec"] - 1.0, -1.0)  # flat 1u stake
    return led


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--filter-lineup-source", default="queue_live")
    a = p.parse_args()
    s = pd.Timestamp(a.start).date(); en = pd.Timestamp(a.end).date()

    cols = _PRED_COLS + ", lineup_source"
    with engine.begin() as conn:
        v2 = pd.read_sql(text(
            f"SELECT {cols} FROM model_outputs_season WHERE date::date BETWEEN :s AND :e"
        ), conn, params={"s": s, "e": en})
    if a.filter_lineup_source:
        v2 = v2[v2["lineup_source"].str.contains(a.filter_lineup_source, na=False)]
    ev = attach_actuals(v2, _load_games(s, en))
    led = _candidates(ev)

    print(f"\nedge-bucket ROI (flat 1u, no edge floor)  {a.start}..{a.end}  rows={len(ev)}")
    for mkt in ("ml", "rl", "total"):
        sub = led[led["bet_type"] == mkt]
        print(f"\n{mkt.upper()}  (n={len(sub)})")
        print(f"  {'edge bucket':>14} | {'n':>5} | {'ROI':>8} | {'win%':>6}")
        sub = sub.assign(b=pd.cut(sub["edge"], BUCKETS))
        for bk, g in sub.groupby("b", observed=True):
            if len(g) == 0:
                continue
            roi = g["pnl"].sum() / len(g)
            print(f"  {str(bk):>14} | {len(g):>5} | {roi:>+7.1%} | {g['won'].mean():>5.0%}")


if __name__ == "__main__":
    main()
