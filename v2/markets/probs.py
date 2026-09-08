"""Sample-based market probabilities and percentile bands.

ML / RL / totals probabilities come straight from the simulator's
(home_runs, away_runs) sample arrays rather than an analytic fit, so they carry
the simulator's own variance structure and yield the p10/p50/p90 columns directly.

Pushes (home_runs - away_runs == -spread, or total == line) only happen at
integer spreads/lines, which are rare in MLB. When they occur the push mass is
split 50/50 to mirror v1 settlement convention.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

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


def paired_market_quotes(home_odds: dict | None, away_odds: dict | None) -> list[dict]:
    """Pair prices and retain the exact observation timestamps used in scoring."""
    home_prices, away_prices = _ml_by_book(home_odds), _ml_by_book(away_odds)
    home_offers = {o.get("book"): o for o in (home_odds or {}).get("offers", [home_odds] if home_odds else [])}
    away_offers = {o.get("book"): o for o in (away_odds or {}).get("offers", [away_odds] if away_odds else [])}
    result = []
    now = pd.Timestamp.now(tz="UTC")
    for book in sorted(home_prices.keys() & away_prices.keys()):
        h, a = home_offers[book], away_offers[book]
        ht, at = h.get("scraped_at"), a.get("scraped_at")
        pair = {"book": book, "home_moneyline": home_prices[book], "away_moneyline": away_prices[book]}
        # Legacy callers can calculate a probability without timestamps, but
        # those quotes cannot pass the frozen-forecast research gate.
        if pd.notna(ht) or pd.notna(at):
            if pd.isna(ht) or pd.isna(at):
                continue
            ht, at = pd.to_datetime(ht, utc=True), pd.to_datetime(at, utc=True)
            if abs((ht - at).total_seconds()) > 5 or max(ht, at) > now:
                continue
            pair.update(home_quoted_at=ht.isoformat(), away_quoted_at=at.isoformat())
        for side, offer in (("home", h), ("away", a)):
            for key in ("spread", "spread_odds"):
                value = offer.get(key)
                pair[f"{side}_{key}"] = float(value) if pd.notna(value) else None
        result.append(pair)
    return result


def consensus_home_prob(home_odds: dict | None, away_odds: dict | None) -> float | None:
    probs = []
    for pair in paired_market_quotes(home_odds, away_odds):
        h = american_to_prob(pair["home_moneyline"])
        a = american_to_prob(pair["away_moneyline"])
        probs.append(h / (h + a))
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
        book = offer.get("book")
        ml = offer.get("moneyline")
        if not book or ml is None or not np.isfinite(float(ml)) or abs(float(ml)) < 100:
            continue
        out[book] = float(ml)
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
