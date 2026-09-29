"""Minimum-relative-entropy adjustment of actual simulated score pairs.

Moneyline keeps the existing logit blend, run lines keep market consensus, and
totals blend conditional (non-push) probabilities. No synthetic scores are added.
The empirical counts are retained so future research can replay the actual joint
forecast rather than reconstructing it from marginal histograms.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp

from backend.data.odds_api import MAX_QUOTE_AGE
from backend.simulation import american_to_prob
from backend.strategy import MARKET_ANCHOR_W_MODEL
from v2.markets.probs import anchor_home_prob, consensus_cover_prob, consensus_home_prob

VERSION = "joint-entropy-v1"
TOTALS_MODEL_WEIGHT = 0.5
# Operational guards, not accuracy claims. A missed refresh must not turn an
# old quote into a new model opinion or concentrate forecasts on rare samples.
MIN_EFFECTIVE_SAMPLE_FRACTION = 0.5
MAX_SAMPLE_WEIGHT_RATIO = 5.0
TARGET_TOLERANCE = 1e-6


def fresh_odds(odds: dict | None, *, as_of: datetime, start_time) -> dict | None:
    """Keep executable, timestamped pregame offers; invalid markets stay absent."""
    now = pd.Timestamp(as_of)
    start = pd.to_datetime(start_time, utc=True)
    offers = []
    seen = set()
    duplicates = set()
    for source in (odds or {}).get("offers") or ([odds] if odds else []):
        book = source.get("book")
        try:
            quoted = pd.to_datetime(source.get("scraped_at"), utc=True)
        except (TypeError, ValueError):
            continue
        if (not book or pd.isna(quoted) or not pd.Timedelta(0) <= now - quoted <= MAX_QUOTE_AGE
                or (pd.notna(start) and quoted >= start)):
            continue
        if book in seen:
            duplicates.add(book)
        seen.add(book)
        offer = {**source, "scraped_at": quoted.isoformat()}
        for field in ("moneyline", "spread_odds", "total_over_odds", "total_under_odds"):
            value = _number(offer.get(field))
            offer[field] = value if value is not None and abs(value) >= 100 else None
        spread, total = _number(offer.get("spread")), _number(offer.get("total"))
        offer["spread"] = spread if spread in (-1.5, 1.5) else None
        offer["total"] = total if total is not None and total > 0 and (2 * total).is_integer() else None
        offers.append(offer)
    offers = [o for o in offers if o["book"] not in duplicates]
    return {**offers[0], "offers": offers} if offers else None


def market_snapshot(home_odds: dict | None, away_odds: dict | None) -> dict:
    """Stable record of the fresh offers used by a forecast, including totals."""
    fields = ("book", "scraped_at", "moneyline", "spread", "spread_odds",
              "total", "total_over_odds", "total_under_odds")
    return {
        side: [{key: offer.get(key) for key in fields}
               for offer in sorted((package or {}).get("offers", []), key=lambda o: o["book"])]
        for side, package in (("home", home_odds), ("away", away_odds))
    }


def _number(value) -> float | None:
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def total_consensus(home_odds: dict | None, away_odds: dict | None) -> dict[float, float]:
    """Over/under are a pair within one offer, even without a moneyline pair."""
    by_book: dict[str, list[dict]] = {}
    for package in (home_odds, away_odds):
        for offer in (package or {}).get("offers", []):
            by_book.setdefault(offer["book"], []).append(offer)
    by_line: dict[float, list[float]] = {}
    for offers in by_book.values():
        fields = ("total", "total_over_odds", "total_under_odds")
        offers = [o for o in offers if all(o.get(k) is not None for k in fields)]
        if not offers:
            continue
        first = offers[0]
        if any(any(o.get(k) != first[k] for k in fields)
               or abs((pd.Timestamp(o["scraped_at"]) - pd.Timestamp(first["scraped_at"])).total_seconds()) > 5
               for o in offers[1:]):
            continue
        over, under = (american_to_prob(first[k]) for k in fields[1:])
        by_line.setdefault(first["total"], []).append(over / (over + under))
    return {line: float(np.mean(values)) for line, values in by_line.items()}


def adjust_joint(
    home_runs: np.ndarray, away_runs: np.ndarray,
    home_odds: dict | None, away_odds: dict | None,
) -> tuple[np.ndarray | None, dict]:
    """Return per-simulation weights, or an explicit guarded legacy fallback.

    Callers supply fresh_odds outputs. Constraints use one frozen consensus per
    event, before any executable-offer shopping. Failure never partially applies
    a new forecast or quietly publishes an unmatched market target.
    """
    h, a = np.asarray(home_runs), np.asarray(away_runs)
    if (h.ndim != 1 or a.shape != h.shape or not len(h)
            or not np.isfinite(h).all() or not np.isfinite(a).all()
            or (h < 0).any() or (a < 0).any() or (h != np.floor(h)).any() or (a != np.floor(a)).any()):
        raise ValueError("Joint adjustment requires matching nonempty integer score arrays")
    scores, inverse, counts = np.unique(np.column_stack((h, a)), axis=0, return_inverse=True, return_counts=True)
    base = counts / len(h)
    margin = scores[:, 0] - scores[:, 1]
    total = scores.sum(axis=1)
    win = (margin > 0).astype(float) + 0.5 * (margin == 0)
    raw_win = float(base @ win)
    market_win = consensus_home_prob(home_odds, away_odds)
    target_win = anchor_home_prob(round(raw_win, 4), market_win)
    features = [win]
    targets = [{"market": "moneyline", "point": None, "raw_probability": raw_win,
                "market_probability": market_win, "target_probability": target_win,
                "model_weight": MARKET_ANCHOR_W_MODEL if market_win is not None else 1.0}]
    for line in (-1.5, 1.5):
        market = consensus_cover_prob(home_odds, away_odds, line)
        if market is None:
            continue
        cover = (margin > -line).astype(float)
        features.append(cover)
        targets.append({"market": "runline", "point": line, "raw_probability": float(base @ cover),
                        "market_probability": market, "target_probability": market, "model_weight": 0.0})
    for line, market in sorted(total_consensus(home_odds, away_odds).items()):
        over, push = total > line, total == line
        settled = float(base @ (~push))
        if settled <= 0:
            continue
        raw = float(base @ over) / settled
        target = float(expit(TOTALS_MODEL_WEIGHT * logit(np.clip(raw, 1e-7, 1 - 1e-7))
                             + (1 - TOTALS_MODEL_WEIGHT) * logit(market)))
        # E[over + target * push] = target preserves the quoted conditional
        # probability without treating an integer-line push as a loss or win.
        features.append(over.astype(float) + target * push)
        targets.append({"market": "total", "point": line, "raw_probability": raw,
                        "market_probability": market, "target_probability": target,
                        "model_weight": TOTALS_MODEL_WEIGHT})

    matrix = np.array(features).T
    target = np.array([t["target_probability"] for t in targets])
    context = {"version": VERSION, "status": "fallback", "sample_count": len(h), "targets": targets,
               "quotes": {"home": (home_odds or {}).get("offers", []),
                          "away": (away_odds or {}).get("offers", [])},
               "raw_joint_counts": [[int(s[0]), int(s[1]), int(n)] for s, n in zip(scores, counts)]}
    log_base = np.log(base)

    def objective(coefficient):
        logits = log_base + matrix @ coefficient
        normalizer = logsumexp(logits)
        probability = np.exp(logits - normalizer)
        return normalizer - coefficient @ target, matrix.T @ probability - target

    result = minimize(objective, np.zeros(len(target)), jac=True, method="L-BFGS-B",
                      bounds=[(-30, 30)] * len(target),
                      options={"gtol": 1e-9, "ftol": 1e-13, "maxiter": 200})
    logits = log_base + matrix @ result.x
    probability = np.exp(logits - logsumexp(logits))
    residual = float(np.max(np.abs(matrix.T @ probability - target)))
    ratio = probability / base
    ess_fraction = float(1 / np.sum(probability * ratio))
    max_ratio = float(ratio.max())
    context.update(target_error=residual, effective_sample_fraction=ess_fraction, max_weight_ratio=max_ratio)
    if not np.isfinite(probability).all() or residual > TARGET_TOLERANCE:
        context["reason"] = "infeasible_or_unconverged_targets"
        return None, context
    if ess_fraction < MIN_EFFECTIVE_SAMPLE_FRACTION or max_ratio > MAX_SAMPLE_WEIGHT_RATIO:
        context["reason"] = "excessive_reweighting"
        return None, context
    context.update(status="adjusted", reason=None,
                   joint_distribution=[[int(s[0]), int(s[1]), float(p)] for s, p in zip(scores, probability)])
    return ratio[inverse] / len(h), context
