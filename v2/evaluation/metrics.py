"""Head-to-head metric library for the Phase 6 v1-vs-v2 backtest.

Thin layer over backend.metrics + backend.evaluate_model. The v1 and v2
prediction tables share a schema, so the same ledger reconstruction works for
both. Anything not already in backend/ goes here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.metrics import brier_score, calibration_curve, log_loss

# Mirrors bet_ledger_v's hardcoded edge gates. Kept literal (not imported from
# EV_THRESHOLDS) so this stays pinned to the view's grading even if thresholds
# move elsewhere - the head-to-head must grade v1 and v2 identically.
_ML_EDGE = 0.045
_RL_EDGE = 0.045
_TOTAL_EDGE = 0.065


def _american_to_implied(o: np.ndarray) -> np.ndarray:
    o = o.astype(float)
    return np.where(o > 0, 100.0 / (o + 100.0), (-o) / ((-o) + 100.0))


def _american_to_decimal(o: np.ndarray) -> np.ndarray:
    o = o.astype(float)
    return np.where(o > 0, 1.0 + o / 100.0, 1.0 + 100.0 / np.abs(o))


def _build_ledger_from_eval(e: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the bet ledger from a single model's predictions+actuals.

    Mirrors bet_ledger_v exactly, but grades the passed-in eval_df instead of
    re-joining the date-partitioned live view (which would return v1's rows for
    every pre-cutover date regardless of which model is being summarized).

    eval_df must carry attach_actuals output: actual_win, actual_margin,
    game_total, plus the _PRED_COLS prediction fields.
    """
    if e.empty:
        return pd.DataFrame(columns=["bet_type", "stake", "payout"])
    e = e.copy()
    for c in ("win_prob", "p_cover", "p_over", "p_under", "moneyline", "spread",
              "spread_odds", "total", "total_over_odds", "total_under_odds",
              "kelly_quarter_ml", "kelly_quarter_rl", "kelly_quarter_total"):
        e[c] = pd.to_numeric(e[c], errors="coerce")
    won_ml = e["actual_win"].astype(int) == 1
    margin = e["actual_margin"].astype(float)

    frames = []

    # --- moneyline ---
    m = e[(e["ev_flag"] == e["team"]) & e["moneyline"].notna()
          & e["kelly_quarter_ml"].notna() & (e["kelly_quarter_ml"] > 0)
          & e["win_prob"].notna()].copy()
    if not m.empty:
        m["edge"] = m["win_prob"] - _american_to_implied(m["moneyline"].to_numpy())
        m = m[m["edge"] >= _ML_EDGE]
        m["bet_type"] = "ml"
        m["stake"] = m["kelly_quarter_ml"].astype(float)
        m["dec"] = _american_to_decimal(m["moneyline"].to_numpy())
        m["won"] = won_ml.loc[m.index]
        frames.append(m[["bet_type", "stake", "dec", "won"]])

    # --- run line ---
    r = e[(e["run_line_ev_flag"] == e["team"]) & e["spread"].notna()
          & e["spread_odds"].notna() & e["kelly_quarter_rl"].notna()
          & (e["kelly_quarter_rl"] > 0) & e["p_cover"].notna()].copy()
    if not r.empty:
        sp = r["spread"].astype(float)
        mg = margin.loc[r.index]
        r["won"] = np.where(
            sp < 0, mg >= sp.abs(),
            np.where(sp > 0, (mg > 0) | (mg >= -sp), False),
        )
        r["edge"] = r["p_cover"] - _american_to_implied(r["spread_odds"].to_numpy())
        r = r[r["edge"] >= _RL_EDGE]
        r["bet_type"] = "rl"
        r["stake"] = r["kelly_quarter_rl"].astype(float)
        r["dec"] = _american_to_decimal(r["spread_odds"].to_numpy())
        frames.append(r[["bet_type", "stake", "dec", "won"]])

    # --- totals (one bet per game: first team alphabetically, matching the
    #     view's row_number() OVER (PARTITION BY game_pk ORDER BY team)) ---
    t = e[e["total_play"].isin(["Over", "Under"]) & e["total"].notna()
          & e["game_total"].notna() & e["kelly_quarter_total"].notna()
          & (e["kelly_quarter_total"] > 0)].copy()
    if not t.empty:
        is_over = t["total_play"] == "Over"
        t["p_sel"] = np.where(is_over, t["p_over"], t["p_under"])
        t["odds_sel"] = np.where(is_over, t["total_over_odds"], t["total_under_odds"])
        t = t[t["p_sel"].notna() & t["odds_sel"].notna()].copy()
        gt = t["game_total"].astype(float)
        tl = t["total"].astype(float)
        t["won"] = np.where(is_over.loc[t.index], gt > tl, gt < tl)
        t["edge"] = t["p_sel"] - _american_to_implied(t["odds_sel"].to_numpy())
        t = t[t["edge"] >= _TOTAL_EDGE]
        t = t.sort_values(["game_pk", "team"]).drop_duplicates("game_pk", keep="first")
        t["bet_type"] = "total"
        t["stake"] = t["kelly_quarter_total"].astype(float)
        t["dec"] = _american_to_decimal(t["odds_sel"].to_numpy())
        frames.append(t[["bet_type", "stake", "dec", "won"]])

    if not frames:
        return pd.DataFrame(columns=["bet_type", "stake", "payout"])
    led = pd.concat(frames, ignore_index=True)
    led["won"] = led["won"].astype(bool)
    led["payout"] = led["stake"] * np.where(led["won"], led["dec"], 0.0)
    return led[["bet_type", "stake", "payout"]]


def max_calibration_gap(bins: list[dict]) -> float:
    """Max |predicted_mean - observed_rate| across non-empty decile bins.

    Returns NaN when bins is empty. Counts-weighted is intentionally NOT used
    here: the gate is about worst-bin drift, not average drift.
    """
    if not bins:
        return float("nan")
    return max(abs(b["predicted_mean"] - b["observed_rate"]) for b in bins)


def flagged_roi(ledger: pd.DataFrame, market: str) -> dict:
    """ROI + bet count for a given market slice of a bet ledger."""
    sub = ledger[ledger["bet_type"] == market] if not ledger.empty else ledger
    if sub.empty:
        return {"n_bets": 0, "stake": 0.0, "pnl": 0.0, "roi": float("nan")}
    stake = float(sub["stake"].sum())
    pnl = float(sub["payout"].sum() - stake)
    return {
        "n_bets": int(len(sub)),
        "stake": round(stake, 4),
        "pnl": round(pnl, 4),
        "roi": round(pnl / stake, 4) if stake > 0 else float("nan"),
    }


def attach_actuals(pred_df: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Merge actual outcomes onto a predictions frame keyed (game_pk, team).

    games: SELECT game_pk, game_date, home_team, away_team, home_score, away_score FROM games WHERE status='Final'.
    Returns rows where the merge succeeded; adds actual_runs, actual_margin,
    actual_win, game_total, winning_team.
    """
    g = games.dropna(subset=["home_score", "away_score"]).copy()
    g["winning_team"] = np.where(g["home_score"] > g["away_score"], g["home_team"], g["away_team"])
    g["game_total"] = g["home_score"] + g["away_score"]

    home = g.rename(columns={"home_team": "team", "home_score": "actual_runs"})[
        ["game_date", "game_pk", "team", "actual_runs", "winning_team", "game_total"]
    ].copy()
    home["actual_margin"] = (g["home_score"] - g["away_score"]).values

    away = g.rename(columns={"away_team": "team", "away_score": "actual_runs"})[
        ["game_date", "game_pk", "team", "actual_runs", "winning_team", "game_total"]
    ].copy()
    away["actual_margin"] = (g["away_score"] - g["home_score"]).values

    actuals = pd.concat([home, away], ignore_index=True)
    merged = pred_df.merge(
        actuals[["game_pk", "game_date", "team", "actual_runs",
                 "winning_team", "actual_margin", "game_total"]],
        on=["game_pk", "team"],
        how="inner",
        suffixes=("", "_actual"),
    ).dropna(subset=["actual_runs"])
    merged["actual_win"] = (merged["team"] == merged["winning_team"]).astype(int)
    return merged


def model_summary(eval_df: pd.DataFrame) -> dict:
    """All Phase 6 metrics for one model. Input: predictions + actuals merged."""
    if eval_df.empty:
        return {"n_games": 0, "n_team_rows": 0}

    probs = eval_df["win_prob"].astype(float).values
    outcomes = eval_df["actual_win"].astype(float).values

    bins = calibration_curve(probs, outcomes, n_bins=10)
    ledger = _build_ledger_from_eval(eval_df)

    return {
        "n_team_rows": int(len(eval_df)),
        "n_games": int(eval_df["game_pk"].nunique()),
        "brier": round(brier_score(probs, outcomes), 6),
        "log_loss": round(log_loss(probs, outcomes), 6),
        "calibration_bins": bins,
        "max_calibration_gap": round(max_calibration_gap(bins), 4),
        "roi_ml": flagged_roi(ledger, "ml"),
        "roi_rl": flagged_roi(ledger, "rl"),
        "roi_total": flagged_roi(ledger, "total"),
    }


def evaluate_gates(v1: dict, v2: dict) -> dict:
    """Apply the three Phase 6 acceptance gates. Returns per-gate pass/fail + verdict.

    Gates (all three must pass for green-light):
      1. Brier and log-loss: v2 not worse than v1 by more than 1% relative.
      2. Flagged-bet ROI (each market): v2 within 2pp absolute of v1 (or better).
      3. Calibration: v2 max decile-bin gap <= 5pp.
    """
    def rel_worse(v2v: float, v1v: float, tol: float) -> bool:
        if not (np.isfinite(v2v) and np.isfinite(v1v) and v1v > 0):
            return True
        return (v2v - v1v) / v1v > tol

    brier_gate = not rel_worse(v2["brier"], v1["brier"], 0.01)
    logloss_gate = not rel_worse(v2["log_loss"], v1["log_loss"], 0.01)

    def roi_ok(market_key: str) -> bool:
        r1 = v1[market_key]["roi"]
        r2 = v2[market_key]["roi"]
        if not np.isfinite(r1) or not np.isfinite(r2):
            return False
        return (r2 - r1) >= -0.02

    roi_gate = roi_ok("roi_ml") and roi_ok("roi_rl") and roi_ok("roi_total")

    calib_gate = np.isfinite(v2["max_calibration_gap"]) and v2["max_calibration_gap"] <= 0.05

    gates = {
        "brier_within_1pct": brier_gate,
        "logloss_within_1pct": logloss_gate,
        "roi_within_2pp": roi_gate,
        "calibration_max_5pp": calib_gate,
    }
    gates["all_pass"] = all([brier_gate, logloss_gate, roi_gate, calib_gate])
    return gates
