"""+EV flagging, Kelly sizing, and confidence for v2.

Reuses backend.kelly + backend.simulation.american_to_prob + backend.strategy's pick policy.
Behavior intentionally mirrors v1's text-flag conventions ("No Play", "Over",
"Under", or team code) so the site renders v2 rows without changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.kelly import american_to_decimal, compute_kelly_row, kelly_fraction
from backend.simulation import american_to_prob, convert_to_odds
from backend.strategy import (
    EV_THRESHOLDS,
    MONEYLINE_ENABLED,
    PICK_MIN_EV,
    PICK_POLICY,
    RUNLINE_ENABLED,
    TOTALS_ENABLED,
)

HIGH_VARIANCE_RUNS_STDEV = 4.0


def our_odds_from_prob(p: float) -> int:
    """American odds implied by our model's win probability."""
    return int(convert_to_odds(p))


def ml_confidence(win_prob: float, moneyline) -> float:
    if not MONEYLINE_ENABLED:
        return float("nan")
    if pd.isna(moneyline):
        return float("nan")
    return float(win_prob - american_to_prob(moneyline))


def rl_confidence(p_cover: float, spread_odds) -> float:
    if not RUNLINE_ENABLED:
        return float("nan")
    if pd.isna(spread_odds) or pd.isna(p_cover):
        return float("nan")
    return float(p_cover - american_to_prob(spread_odds))


def choose_pick(candidates: list[dict], daily_minimum: bool = False) -> dict | None:
    """The game's one pick: the candidate with the highest expected value per unit staked.

    Each candidate is {"market": "ml" | "rl", "team", "probability", "price", "book"}
    with the probability already priced against the paired market. Returns the
    candidate with expected_value, edge (probability minus the vig-inclusive
    implied probability), tier and policy added, or None below PICK_MIN_EV
    unless daily_minimum, which publishes it as tier "minimum".
    """
    enabled = {"ml": MONEYLINE_ENABLED, "rl": RUNLINE_ENABLED}
    scored = []
    for candidate in candidates:
        probability, price = candidate["probability"], candidate["price"]
        if not enabled[candidate["market"]] or probability is None or pd.isna(probability) or pd.isna(price):
            continue
        scored.append({**candidate,
                       "expected_value": round(probability * american_to_decimal(price) - 1, 4),
                       "edge": round(probability - american_to_prob(price), 4)})
    if not scored:
        return None
    pick = max(scored, key=lambda candidate: candidate["expected_value"])
    if pick["expected_value"] >= 0:
        tier = "value"
    elif pick["expected_value"] >= PICK_MIN_EV:
        tier = "lean"
    elif daily_minimum:
        tier = "minimum"
    else:
        return None
    return {**pick, "policy": PICK_POLICY, "tier": tier}


def flag_total_play(
    p_over: float,
    p_under: float,
    total_over_odds,
    total_under_odds,
    total_diff: float | None = None,
    threshold: float = EV_THRESHOLDS["totals"],
) -> str:
    """Return 'Over' / 'Under' / 'No Play'."""
    if not TOTALS_ENABLED:
        return "No Play"
    over_book = american_to_prob(total_over_odds) if not pd.isna(total_over_odds) else float("nan")
    under_book = american_to_prob(total_under_odds) if not pd.isna(total_under_odds) else float("nan")

    if pd.notna(p_over) and pd.notna(over_book) and (p_over - over_book) >= threshold:
        return "Over"
    if pd.notna(p_under) and pd.notna(under_book) and (p_under - under_book) >= threshold:
        return "Under"
    # Fallback (mirrors v1 strategy.py): no book odds → use total_diff direction.
    if pd.isna(over_book) and pd.isna(under_book) and total_diff is not None and pd.notna(total_diff):
        if total_diff >= 1:
            return "Over"
        if total_diff <= -1:
            return "Under"
    return "No Play"


def kelly_pair(p_model: float, american_odds) -> tuple[float, float]:
    """(full, quarter) Kelly. Reuses backend.kelly.compute_kelly_row."""
    if pd.isna(american_odds) or pd.isna(p_model):
        return float("nan"), float("nan")
    return compute_kelly_row(p_model, american_odds)


def kelly_total(
    total_play: str,
    p_over: float,
    p_under: float,
    total_over_odds,
    total_under_odds,
) -> tuple[float, float]:
    """Kelly for the chosen side of the total. (0, 0) when total_play == 'No Play'."""
    if total_play == "Over":
        full = kelly_fraction(p_over, american_to_decimal(total_over_odds))
    elif total_play == "Under":
        full = kelly_fraction(p_under, american_to_decimal(total_under_odds))
    else:
        return 0.0, 0.0
    if pd.isna(full):
        return float("nan"), float("nan")
    return float(full), round(full * 0.25, 6)


def high_variance_flag(
    samples: np.ndarray, threshold: float = HIGH_VARIANCE_RUNS_STDEV, weights: np.ndarray | None = None,
) -> str:
    """'Yes' / 'No' based on stdev of per-team runs across sims."""
    mean = np.average(samples, weights=weights)
    s = float(np.sqrt(np.average((samples - mean) ** 2, weights=weights)))
    return "Yes" if s > threshold else "No"
