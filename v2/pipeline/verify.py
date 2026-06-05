"""v2-specific post-score sanity checks.

Exits 0 on pass, 1 on any hard failure. Called by daily_run.py after scoring.

Usage:
    python -m v2.pipeline.verify --date 2026-05-11
"""
from __future__ import annotations

import argparse
import sys
from datetime import date

import pandas as pd
from sqlalchemy import text

from backend.db import engine

EXPECTED_RUNS_MIN = 0.5
EXPECTED_RUNS_MAX = 15.0
WIN_PROB_MIN = 0.02
WIN_PROB_MAX = 0.98
POSTERIOR_AGE_MAX_DAYS = 2
WIN_PROB_SUM_TOL = 0.02
ANTI_CORR_TOL = 0.01
MIN_LIVE_LINEUP_PCT = 0.80


def run_checks(date_str: str) -> bool:
    """Run all checks. Returns True on pass, False on any failure."""
    q = text("""
        SELECT o.team, g.home_team, g.away_team, o.expected_runs, o.win_prob,
               o.win_prob_p10, o.win_prob_p90, o.posterior_age_days,
               o.lineup_source, o.game_pk, o.starter,
               o.ev_flag, o.run_line_ev_flag, o.total_play
        FROM model_outputs o
        JOIN games g USING (game_pk)
        WHERE o.date::date = :d
    """)
    with engine.begin() as conn:
        df = pd.read_sql(q, conn, params={"d": date_str})

    if df.empty:
        print(f"[verify] FAIL: no rows in model_outputs for {date_str}")
        return False

    failures = []

    # At least 2 rows per game (one per team)
    rows_per_game = df.groupby("game_pk").size()
    bad_games = rows_per_game[rows_per_game != 2]
    if not bad_games.empty:
        failures.append(f"games with != 2 rows: {bad_games.index.tolist()}")

    # Nulls in critical columns
    for col in ("expected_runs", "win_prob", "team", "home_team", "away_team"):
        n_null = df[col].isna().sum()
        if n_null:
            failures.append(f"{col} has {n_null} nulls")

    # Range checks
    xr = df["expected_runs"].dropna()
    out_of_range = xr[(xr < EXPECTED_RUNS_MIN) | (xr > EXPECTED_RUNS_MAX)]
    if not out_of_range.empty:
        failures.append(f"expected_runs out of [{EXPECTED_RUNS_MIN}, {EXPECTED_RUNS_MAX}]: {out_of_range.tolist()}")

    wp = df["win_prob"].dropna()
    out_of_range = wp[(wp < WIN_PROB_MIN) | (wp > WIN_PROB_MAX)]
    if not out_of_range.empty:
        failures.append(f"win_prob out of [{WIN_PROB_MIN}, {WIN_PROB_MAX}]: {out_of_range.tolist()}")

    # Paired win probs sum to ~1
    for gp, grp in df.groupby("game_pk"):
        if len(grp) == 2:
            total = grp["win_prob"].sum()
            if abs(total - 1.0) > WIN_PROB_SUM_TOL:
                failures.append(f"game {gp} win_prob sum = {total:.4f}")

    # Anti-correlation: away.p10 ≈ 1 - home.p90 per game
    for gp, grp in df.groupby("game_pk"):
        if len(grp) == 2:
            home = grp[grp["team"] == grp["home_team"].iloc[0]]
            away = grp[grp["team"] == grp["away_team"].iloc[0]]
            if len(home) == 1 and len(away) == 1:
                diff = abs(float(away["win_prob_p10"].iloc[0]) - (1.0 - float(home["win_prob_p90"].iloc[0])))
                if diff > ANTI_CORR_TOL:
                    failures.append(f"game {gp} anti-correlation violated (diff={diff:.4f})")

    # A missing starter means the sim used a league-mean arm, so any flag on the
    # game is untrustworthy; build_game_rows suppresses it. A flag here means that
    # gate regressed.
    flag_cols = ["ev_flag", "run_line_ev_flag", "total_play"]
    for gp, grp in df.groupby("game_pk"):
        if grp["starter"].isna().any():
            flagged = grp[flag_cols].apply(lambda s: s.ne("No Play").any(), axis=0)
            if flagged.any():
                failures.append(f"game {gp} has a missing starter but is flagged on {flagged[flagged].index.tolist()}")

    # Posterior age
    max_age = df["posterior_age_days"].max()
    if max_age is not None and max_age > POSTERIOR_AGE_MAX_DAYS:
        failures.append(f"posterior_age_days = {max_age} > {POSTERIOR_AGE_MAX_DAYS}")

    # Lineup source distribution (only warn, not fail, pre-lineup-posting)
    if "lineup_source" in df.columns:
        live_pct = df["lineup_source"].str.contains("lineup_live").mean()
        if live_pct < MIN_LIVE_LINEUP_PCT:
            print(f"[verify] WARN: only {live_pct:.0%} of rows have live lineups (expected ≥{MIN_LIVE_LINEUP_PCT:.0%} post-posting)")

    if failures:
        for f in failures:
            print(f"[verify] FAIL: {f}")
        return False

    print(f"[verify] all checks passed for {date_str} ({len(df)} rows, {len(df) // 2} games)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=str(date.today()))
    args = ap.parse_args()
    ok = run_checks(args.date)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
