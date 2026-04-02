"""
Pipeline verification script.

Runs sanity checks against the production DB after pipeline execution.
Exits 0 on all checks passing, 1 on any failure.

Usage: python verify_pipeline.py
"""

import sys
from datetime import date, timedelta
import pandas as pd
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()
from backend.db import engine

CHECKS_PASSED = 0
CHECKS_FAILED = 0
WARNINGS = 0


def check(name, condition, warning_only=False):
    global CHECKS_PASSED, CHECKS_FAILED, WARNINGS
    if condition:
        print(f"  PASS  {name}")
        CHECKS_PASSED += 1
    elif warning_only:
        print(f"  WARN  {name}")
        WARNINGS += 1
    else:
        print(f"  FAIL  {name}")
        CHECKS_FAILED += 1


def main():
    today = date.today()
    yesterday = today - timedelta(days=1)
    week_ago = today - timedelta(days=7)

    print(f"Verification — {today}")
    print("=" * 50)

    with engine.connect() as conn:
        # 1. model_outputs has rows for today (or yesterday)
        mo_count = conn.execute(
            text("SELECT COUNT(*) FROM model_outputs WHERE date::date IN (:d1, :d2)"),
            {"d1": str(today), "d2": str(yesterday)},
        ).scalar()
        check("model_outputs has recent rows", mo_count > 0)

        # 2. Expected runs in reasonable range
        xr_range = conn.execute(
            text("SELECT MIN(expected_runs), MAX(expected_runs) FROM model_outputs")
        ).fetchone()
        if xr_range and xr_range[0] is not None:
            check(f"expected_runs range [{xr_range[0]:.1f}, {xr_range[1]:.1f}]",
                  xr_range[0] >= 0.5 and xr_range[1] <= 15)
        else:
            check("expected_runs exist", False)

        # 3. Win probabilities in reasonable range
        wp_range = conn.execute(
            text("SELECT MIN(win_prob), MAX(win_prob) FROM model_outputs")
        ).fetchone()
        if wp_range and wp_range[0] is not None:
            check(f"win_prob range [{wp_range[0]:.3f}, {wp_range[1]:.3f}]",
                  wp_range[0] >= 0.01 and wp_range[1] <= 0.99)
        else:
            check("win_prob exist", False)

        # 4. Paired teams per game_pk have win_probs summing to ~1.0
        pair_check = conn.execute(text("""
            SELECT game_pk, SUM(win_prob) as total_wp, COUNT(*) as n
            FROM model_outputs
            GROUP BY game_pk
            HAVING COUNT(*) = 2
        """)).fetchall()
        if pair_check:
            bad_pairs = [r for r in pair_check if abs(r[1] - 1.0) > 0.05]
            check(f"win_prob pairs sum to ~1.0 ({len(pair_check)} games, {len(bad_pairs)} off)",
                  len(bad_pairs) == 0)
        else:
            check("win_prob pairs exist", False)

        # 5. No nulls in critical columns
        null_check = conn.execute(text("""
            SELECT COUNT(*) FROM model_outputs
            WHERE expected_runs IS NULL OR win_prob IS NULL OR team IS NULL
        """)).scalar()
        check("no nulls in critical columns", null_check == 0)

        # 6. Games table has today's games
        games_today = conn.execute(
            text("SELECT COUNT(*) FROM games WHERE game_date = :d"),
            {"d": str(today)},
        ).scalar()
        check(f"games table has today's games ({games_today})", games_today > 0)

        # 7. pitcher_stats non-empty (warning if within first week of season)
        ps_count = conn.execute(text("SELECT COUNT(*) FROM pitcher_stats")).scalar()
        is_early = today.month <= 4 and today.day <= 7
        check(f"pitcher_stats non-empty ({ps_count} rows)",
              ps_count > 0, warning_only=is_early)

        # 8. bullpen_stats non-empty
        bs_count = conn.execute(text("SELECT COUNT(*) FROM bullpen_stats")).scalar()
        check(f"bullpen_stats non-empty ({bs_count} rows)",
              bs_count > 0, warning_only=is_early)

        # 9. odds table has recent data
        odds_count = conn.execute(
            text("SELECT COUNT(*) FROM odds WHERE game_pk IN (SELECT game_pk FROM games WHERE game_date IN (:d1, :d2))"),
            {"d1": str(today), "d2": str(today + timedelta(days=1))},
        ).scalar()
        check(f"odds has data for today/tomorrow ({odds_count} rows)", odds_count > 0)

        # 10. Feature fallback rate (xfip at league avg 4.20)
        if mo_count > 0:
            fallback_count = conn.execute(
                text("SELECT COUNT(*) FROM model_outputs WHERE expected_runs = 4.5")
            ).scalar()
            fallback_pct = fallback_count / mo_count * 100 if mo_count > 0 else 0
            check(f"league-avg fallback rate ({fallback_pct:.0f}%)",
                  fallback_pct < 50, warning_only=is_early)

    print(f"\n{'=' * 50}")
    print(f"Results: {CHECKS_PASSED} passed, {CHECKS_FAILED} failed, {WARNINGS} warnings")

    if CHECKS_FAILED > 0:
        print("VERIFICATION FAILED")
        return 1
    else:
        print("VERIFICATION PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
