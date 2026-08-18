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

from backend.simulation import american_to_prob
from backend.strategy import HOME_FIELD_LOGIT, MARKET_ANCHOR_W_MODEL


def market_probs(
    home_runs: np.ndarray,
    away_runs: np.ndarray,
    total_line: float | None,
    spread_home: float | None,
) -> dict:
    """ML / RL / totals empirical probabilities from sim arrays. Missing inputs → None keys."""
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


def consensus_home_prob(home_odds: dict | None, away_odds: dict | None) -> float | None:
    """De-vigged market home win prob, averaged over books quoting both sides.

    Offers are paired by book so one book's vig cancels within its own pair.
    Books quoting only one side are skipped. None when no complete pair exists.
    """
    home_by_book = _ml_by_book(home_odds)
    away_by_book = _ml_by_book(away_odds)
    probs = []
    for book, home_ml in home_by_book.items():
        away_ml = away_by_book.get(book)
        if away_ml is None:
            continue
        imp_home = american_to_prob(home_ml)
        imp_away = american_to_prob(away_ml)
        overround = imp_home + imp_away
        if overround <= 0:
            continue
        probs.append(imp_home / overround)
    return float(np.mean(probs)) if probs else None


def anchor_home_prob(p_home_sim: float, p_market_home: float | None) -> float:
    """Published home win prob: HFA-shifted sim logit blended toward the market.

    The sim carries no home-field advantage, so HOME_FIELD_LOGIT is added to its
    logit first; the market prob already prices HFA, so the blend double-counts
    nothing. With no market pair the shifted sim prob passes through unblended.
    Monotonic in p_home_sim, so it can also transform the win-prob band
    endpoints without breaking their ordering or pairwise anti-correlation.
    """
    logit_sim = _logit(p_home_sim) + HOME_FIELD_LOGIT
    if p_market_home is None:
        return round(_sigmoid(logit_sim), 4)
    blended = MARKET_ANCHOR_W_MODEL * logit_sim + (1.0 - MARKET_ANCHOR_W_MODEL) * _logit(p_market_home)
    return round(_sigmoid(blended), 4)


def _ml_by_book(odds: dict | None) -> dict:
    if odds is None:
        return {}
    offers = odds.get("offers") or [odds]
    out = {}
    for offer in offers:
        ml = offer.get("moneyline")
        if ml is None or _isnan(ml):
            continue
        out[offer.get("book")] = float(ml)
    return out


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-4), 1.0 - 1e-4)
    return float(np.log(p / (1.0 - p)))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


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
