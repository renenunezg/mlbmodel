"""Sample-based market probabilities and percentile bands.

Phase 5 derives ML / RL / totals probabilities directly from the simulator's
(home_runs, away_runs) sample arrays rather than refitting an analytic NB. The
empirical approach uses the simulator's actual variance structure (which Phase 4
already calibrated to within 5% of MLB norms) and naturally produces the
percentile columns p10/p50/p90 in the v2 schema.

Pushes (home_runs - away_runs == -spread, or total == line) only happen at
integer spreads/lines, which are rare in MLB. When they occur the push mass is
split 50/50 to mirror v1 settlement convention.
"""
from __future__ import annotations

import numpy as np


def market_probs(
    home_runs: np.ndarray,
    away_runs: np.ndarray,
    total_line: float | None,
    spread_home: float | None,
) -> dict:
    """Return ML / RL / totals probabilities from sim arrays.

    Args:
        home_runs, away_runs: (n_sims,) int arrays from simulate_game.
        total_line: book total runs (e.g. 8.5). None to skip totals.
        spread_home: signed run-line from home perspective (e.g. -1.5 = home favored).
            None to skip RL.

    Returns dict with: p_home_win, p_away_win, p_home_cover, p_away_cover,
    p_over, p_under. Keys whose input is missing are None.
    """
    h = np.asarray(home_runs)
    a = np.asarray(away_runs)
    n = len(h)
    if n == 0 or len(a) != n:
        raise ValueError("home_runs and away_runs must be same non-empty length")

    margin = h - a
    p_home_win_strict = float((margin > 0).mean())
    p_away_win_strict = float((margin < 0).mean())
    p_tie = float((margin == 0).mean())
    # Ties shouldn't happen (game_sim resolves via extras), but split 50/50 if so.
    p_home_win = p_home_win_strict + 0.5 * p_tie
    p_away_win = p_away_win_strict + 0.5 * p_tie

    out = {
        "p_home_win": round(p_home_win, 4),
        "p_away_win": round(p_away_win, 4),
        "p_home_cover": None,
        "p_away_cover": None,
        "p_over": None,
        "p_under": None,
    }

    if spread_home is not None and not _isnan(spread_home):
        # Home covers when (h - a) > -spread_home. Push at equality.
        threshold = -float(spread_home)
        p_home_strict = float((margin > threshold).mean())
        p_push_rl = float((margin == threshold).mean())
        p_away_strict = float((margin < threshold).mean())
        out["p_home_cover"] = round(p_home_strict + 0.5 * p_push_rl, 4)
        out["p_away_cover"] = round(p_away_strict + 0.5 * p_push_rl, 4)

    if total_line is not None and not _isnan(total_line):
        totals = h + a
        line = float(total_line)
        p_over_strict = float((totals > line).mean())
        p_under_strict = float((totals < line).mean())
        p_push_t = float((totals == line).mean())
        out["p_over"] = round(p_over_strict + 0.5 * p_push_t, 4)
        out["p_under"] = round(p_under_strict + 0.5 * p_push_t, 4)

    return out


def runs_percentiles(arr: np.ndarray) -> tuple[float, float, float]:
    """Return (p10, p50, p90) of runs."""
    a = np.asarray(arr)
    p10, p50, p90 = np.quantile(a, [0.10, 0.50, 0.90])
    return float(p10), float(p50), float(p90)


def _isnan(x) -> bool:
    try:
        return bool(np.isnan(x))
    except (TypeError, ValueError):
        return False
