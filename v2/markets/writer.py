"""Build per-team rows from sim arrays + odds, write to model_outputs / model_outputs_season.

Row schema mirrors the live Supabase columns. Keyed by (game_pk, team). The
daily table is rebuilt per-date (delete + insert); the season table is upserted
keyed on (game_pk, team).

`win_prob_p10` / `win_prob_p90` are passed in by the caller (computed from
per-posterior-draw win-prob samples). The writer doesn't bootstrap them, since
that would only capture MC noise and mislead consumers.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import JSONB

from backend.db import engine
from v2.bayesian._common import POSTERIORS_DIR
from v2.markets.ev import (
    flag_ml,
    flag_runline,
    flag_total_play,
    high_variance_flag,
    kelly_pair,
    kelly_total,
    ml_confidence,
    our_odds_from_prob,
    rl_confidence,
)
from v2.markets.probs import market_probs, runs_percentiles




def build_game_rows(
    *,
    game_pk: int,
    game_date: pd.Timestamp,
    start_time: pd.Timestamp | None,
    home_team: str,
    away_team: str,
    home_starter: str | None,
    away_starter: str | None,
    home_runs: np.ndarray,
    away_runs: np.ndarray,
    home_odds: dict | None,
    away_odds: dict | None,
    lineup_source: str,
    lineups_locked: bool,
    posterior_age_days: int,
    home_wp_p10: float | None = None,
    home_wp_p90: float | None = None,
    lineup_hash: str | None = None,
) -> list[dict]:
    """Return two dict rows (home + away) ready to write to model_outputs.

    `home_odds` / `away_odds` come from the `odds` table; either may be None when
    no odds row matched. Per-row keys we read: moneyline, spread, spread_odds,
    total, total_over_odds, total_under_odds.
    """
    h = np.asarray(home_runs)
    a = np.asarray(away_runs)
    n = len(h)

    home_total_line = _get(home_odds, "total")
    home_spread = _get(home_odds, "spread")  # signed from home perspective per odds_api.py
    home_spread_odds = _get(home_odds, "spread_odds")
    home_total_over = _get(home_odds, "total_over_odds")
    home_total_under = _get(home_odds, "total_under_odds")
    home_ml = _get(home_odds, "moneyline")

    away_total_line = _get(away_odds, "total")
    away_spread = _get(away_odds, "spread")  # signed from away perspective
    away_spread_odds = _get(away_odds, "spread_odds")
    away_total_over = _get(away_odds, "total_over_odds")
    away_total_under = _get(away_odds, "total_under_odds")
    away_ml = _get(away_odds, "moneyline")

    # Same total line for both sides (they always match in MLB books). Prefer
    # whichever side actually populated it.
    total_line = home_total_line if pd.notna(home_total_line) else away_total_line
    spread_home = home_spread if pd.notna(home_spread) else (
        -float(away_spread) if pd.notna(away_spread) else None
    )

    probs = market_probs(h, a, total_line, spread_home)
    p_home_win = probs["p_home_win"]
    p_away_win = probs["p_away_win"]
    p_home_cover = probs["p_home_cover"]
    p_away_cover = probs["p_away_cover"]
    p_over = probs["p_over"]
    p_under = probs["p_under"]

    h_p10, h_p50, h_p90 = runs_percentiles(h)
    a_p10, a_p50, a_p90 = runs_percentiles(a)
    total_arr = h + a
    t_p10, t_p50, t_p90 = runs_percentiles(total_arr)

    # Empirical run-distribution histograms (21 bins, 0..20 runs) for the frontend.
    # The capped-bin trick: runs > 20 land in bin 20. Tail mass past 20 is tiny.
    h_clipped = np.minimum(h, 20)
    a_clipped = np.minimum(a, 20)
    h_hist = (np.bincount(h_clipped, minlength=21)[:21] / n).round(5).tolist()
    a_hist = (np.bincount(a_clipped, minlength=21)[:21] / n).round(5).tolist()

    # away band is the complement of the home band (perfectly anti-correlated
    # by construction: every per-draw home_wp_k pairs with away_wp_k = 1 - home_wp_k).
    if home_wp_p10 is not None and home_wp_p90 is not None:
        away_wp_p10 = round(1.0 - home_wp_p90, 4)
        away_wp_p90 = round(1.0 - home_wp_p10, 4)
    else:
        away_wp_p10 = away_wp_p90 = None

    home_xR = float(h.mean())
    away_xR = float(a.mean())
    our_total = round(home_xR + away_xR, 4)
    home_total_diff = round(our_total - float(total_line), 4) if pd.notna(total_line) else None
    away_total_diff = home_total_diff

    now = datetime.now(timezone.utc)
    base = {
        "game_pk": int(game_pk),
        "date": pd.Timestamp(game_date).to_pydatetime().replace(tzinfo=None),
        "start_time": pd.Timestamp(start_time).to_pydatetime() if start_time is not None else None,
        "our_total": our_total,
        "lineups_locked": lineups_locked,
        "lineup_source": lineup_source,
        "lineup_hash": lineup_hash,
        "prediction_updated_at": now,
        "posterior_age_days": int(posterior_age_days),
    }

    home_row = {
        **base,
        "team": home_team,
        "starter": home_starter,
        "expected_runs": round(home_xR, 4),
        "expected_runs_p10": round(h_p10, 4),
        "expected_runs_p50": round(h_p50, 4),
        "expected_runs_p90": round(h_p90, 4),
        "total_p10": round(t_p10, 4),
        "total_p50": round(t_p50, 4),
        "total_p90": round(t_p90, 4),
        "win_prob": p_home_win,
        "win_prob_p10": home_wp_p10,
        "win_prob_p90": home_wp_p90,
        "our_odds": our_odds_from_prob(p_home_win),
        "moneyline": _to_float(home_ml),
        "spread": _to_float(home_spread),
        "spread_odds": _to_float(home_spread_odds),
        "total": _to_float(total_line),
        "total_over_odds": _to_float(home_total_over),
        "total_under_odds": _to_float(home_total_under),
        "p_cover": p_home_cover,
        "p_over": p_over,
        "p_under": p_under,
        "total_diff": home_total_diff,
        "ev_flag": flag_ml(home_team, p_home_win, home_ml),
        "run_line_ev_flag": flag_runline(home_team, p_home_cover, home_spread_odds),
        "total_play": flag_total_play(p_over, p_under, home_total_over, home_total_under, home_total_diff),
        "ml_confidence": ml_confidence(p_home_win, home_ml),
        "run_line_confidence": rl_confidence(p_home_cover, home_spread_odds),
        "high_variance_flag": high_variance_flag(h),
        "runs_hist": h_hist,
    }
    home_row.update(_kelly_block(home_row, p_home_win, p_home_cover, p_over, p_under,
                                 home_ml, home_spread_odds, home_total_over, home_total_under))

    away_row = {
        **base,
        "team": away_team,
        "starter": away_starter,
        "expected_runs": round(away_xR, 4),
        "expected_runs_p10": round(a_p10, 4),
        "expected_runs_p50": round(a_p50, 4),
        "expected_runs_p90": round(a_p90, 4),
        "total_p10": round(t_p10, 4),
        "total_p50": round(t_p50, 4),
        "total_p90": round(t_p90, 4),
        "win_prob": p_away_win,
        "win_prob_p10": away_wp_p10,
        "win_prob_p90": away_wp_p90,
        "our_odds": our_odds_from_prob(p_away_win),
        "moneyline": _to_float(away_ml),
        "spread": _to_float(away_spread),
        "spread_odds": _to_float(away_spread_odds),
        "total": _to_float(total_line),
        "total_over_odds": _to_float(away_total_over),
        "total_under_odds": _to_float(away_total_under),
        "p_cover": p_away_cover,
        "p_over": p_over,
        "p_under": p_under,
        "total_diff": away_total_diff,
        "ev_flag": flag_ml(away_team, p_away_win, away_ml),
        "run_line_ev_flag": flag_runline(away_team, p_away_cover, away_spread_odds),
        "total_play": flag_total_play(p_over, p_under, away_total_over, away_total_under, away_total_diff),
        "ml_confidence": ml_confidence(p_away_win, away_ml),
        "run_line_confidence": rl_confidence(p_away_cover, away_spread_odds),
        "high_variance_flag": high_variance_flag(a),
        "runs_hist": a_hist,
    }
    away_row.update(_kelly_block(away_row, p_away_win, p_away_cover, p_over, p_under,
                                 away_ml, away_spread_odds, away_total_over, away_total_under))

    return [home_row, away_row]


def _kelly_block(row, win_prob, p_cover, p_over, p_under,
                 moneyline, spread_odds, total_over_odds, total_under_odds):
    full_ml, q_ml = kelly_pair(win_prob, moneyline)
    full_rl, q_rl = kelly_pair(p_cover, spread_odds) if p_cover is not None else (float("nan"), float("nan"))
    full_t, q_t = kelly_total(row["total_play"], p_over, p_under, total_over_odds, total_under_odds)
    return {
        "kelly_full_ml": full_ml,
        "kelly_quarter_ml": q_ml,
        "kelly_full_rl": full_rl,
        "kelly_quarter_rl": q_rl,
        "kelly_full_total": full_t,
        "kelly_quarter_total": q_t,
    }


def _get(d: dict | None, key: str):
    if d is None:
        return float("nan")
    v = d.get(key)
    return v if v is not None else float("nan")


def _to_float(v):
    if v is None:
        return None
    if pd.isna(v):
        return None
    return float(v)


def write_daily(date: pd.Timestamp, rows: list[dict]) -> None:
    """Upsert per (game_pk, team) into model_outputs.

    Per-game upsert (not date-wide DELETE) so a partial scoring run, e.g. the
    hourly lineup refresh that only touches changed games, can't blank out the
    rest of the day's slate between DELETE and INSERT.
    """
    if not rows:
        return
    df = pd.DataFrame(rows)
    pairs = list(df[["game_pk", "team"]].itertuples(index=False, name=None))
    if not pairs:
        return
    with engine.begin() as conn:
        for game_pk, team in pairs:
            conn.execute(
                text("DELETE FROM model_outputs WHERE game_pk = :g AND team = :t"),
                {"g": int(game_pk), "t": team},
            )
        df.to_sql("model_outputs", con=conn, if_exists="append", index=False, dtype={"runs_hist": JSONB})


def append_season(rows: list[dict]) -> None:
    """Upsert into model_outputs_season keyed on (game_pk, team). Idempotent."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    pairs = list(df[["game_pk", "team"]].itertuples(index=False, name=None))
    if not pairs:
        return
    with engine.begin() as conn:
        for game_pk, team in pairs:
            conn.execute(
                text("DELETE FROM model_outputs_season WHERE game_pk = :g AND team = :t"),
                {"g": int(game_pk), "t": team},
            )
        df.to_sql("model_outputs_season", con=conn, if_exists="append", index=False, dtype={"runs_hist": JSONB})


def posterior_age_days(now: datetime | None = None, posteriors_dir: Path = POSTERIORS_DIR) -> int:
    """Days since the most recent NetCDF trace mtime."""
    now = now or datetime.now(timezone.utc)
    files = list(posteriors_dir.glob("*.nc"))
    if not files:
        return -1
    newest = max(f.stat().st_mtime for f in files)
    return int((now.timestamp() - newest) // 86400)
