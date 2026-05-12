"""Phase 6 head-to-head: join v1 + v2 predictions with actuals, score, report.

Usage:
    python -m v2.evaluation.backtester --start 2025-03-27 --end 2025-09-28
    python -m v2.evaluation.backtester --start 2025-07-15 --end 2025-07-15 --out smoke.json

Reads from Supabase:
    - model_outputs_season_v1_archive  (v1, pre-computed all season)
    - model_outputs_season             (v2, populated by replay.py)
    - games                            (Final outcomes)

Filters v2 to lineup_source LIKE '%queue_cache%' so stub-queue rows don't
confound the comparison. Inner-joins v1 and v2 on (game_pk, team) so both
models score the same set of games.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from backend.db import engine
from v2.evaluation.metrics import attach_actuals, evaluate_gates, model_summary
from v2.evaluation.report import format_head_to_head


REPORTS_DIR = Path(__file__).resolve().parent / "reports"


# Columns shared by both tables and needed for ledger reconstruction.
_PRED_COLS = (
    "game_pk, team, date, expected_runs, win_prob, our_odds, "
    "p_cover, p_over, p_under, total_diff, "
    "moneyline, spread, spread_odds, total, total_over_odds, total_under_odds, "
    "ev_flag, run_line_ev_flag, total_play, "
    "kelly_quarter_ml, kelly_quarter_rl, kelly_quarter_total"
)


def _load_predictions(table: str, start, end, extra_cols: str = "") -> pd.DataFrame:
    cols = _PRED_COLS + (", " + extra_cols if extra_cols else "")
    q = text(
        f"SELECT {cols} FROM {table} "
        "WHERE date::date BETWEEN :s AND :e"
    )
    with engine.begin() as conn:
        return pd.read_sql(q, conn, params={"s": start, "e": end})


def _load_games(start, end) -> pd.DataFrame:
    q = text(
        "SELECT game_pk, game_date, home_team, away_team, home_score, away_score, status "
        "FROM games WHERE game_date BETWEEN :s AND :e AND status = 'Final'"
    )
    with engine.begin() as conn:
        return pd.read_sql(q, conn, params={"s": start, "e": end})


def run(start, end, out_path: Path | None, lineup_filter: str) -> dict:
    print(f"[backtester] loading predictions {start} -> {end}")
    v1 = _load_predictions("model_outputs_season_v1_archive", start, end)
    v2 = _load_predictions("model_outputs_season", start, end, extra_cols="lineup_source")
    games = _load_games(start, end)
    print(f"[backtester] v1 rows={len(v1)}, v2 rows={len(v2)}, final games={len(games)}")

    if v2.empty:
        print("[backtester] v2 table empty for this window. Run replay.py first.")
        sys.exit(2)

    if lineup_filter:
        before = len(v2)
        v2 = v2[v2["lineup_source"].str.contains(lineup_filter, regex=True, na=False)].copy()
        print(f"[backtester] lineup filter '{lineup_filter}': kept {len(v2)}/{before} v2 rows")

    # Inner join on (game_pk, team) so both models score the same rows.
    common = v1.merge(v2[["game_pk", "team"]], on=["game_pk", "team"], how="inner")
    keys = common[["game_pk", "team"]].drop_duplicates()
    v1 = v1.merge(keys, on=["game_pk", "team"], how="inner")
    v2 = v2.merge(keys, on=["game_pk", "team"], how="inner")
    print(f"[backtester] common rows after intersect: {len(keys)}")

    v1_eval = attach_actuals(v1, games)
    v2_eval = attach_actuals(v2, games)
    print(f"[backtester] after actuals merge: v1={len(v1_eval)}, v2={len(v2_eval)}")

    v1_summary = model_summary(v1_eval)
    v2_summary = model_summary(v2_eval)
    gates = evaluate_gates(v1_summary, v2_summary)

    window = f"{start} to {end}"
    report = format_head_to_head(v1_summary, v2_summary, gates, window=window)
    print()
    print(report)

    blob = {
        "window": {"start": str(start), "end": str(end)},
        "lineup_filter": lineup_filter,
        "v1": v1_summary,
        "v2": v2_summary,
        "gates": gates,
        "report_text": report,
    }
    if out_path is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = REPORTS_DIR / f"{start}_to_{end}.json"
    out_path.write_text(json.dumps(blob, indent=2, default=str))
    print(f"\n[backtester] wrote {out_path}")
    return blob


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--out", default=None, help="JSON output path")
    p.add_argument(
        "--filter-lineup-source",
        default="queue_cache",
        help="substring match on v2 lineup_source (empty string = no filter)",
    )
    args = p.parse_args()
    start = pd.Timestamp(args.start).date()
    end = pd.Timestamp(args.end).date()
    out = Path(args.out) if args.out else None
    blob = run(start, end, out, args.filter_lineup_source)
    sys.exit(0 if blob["gates"]["all_pass"] else 1)


if __name__ == "__main__":
    main()
