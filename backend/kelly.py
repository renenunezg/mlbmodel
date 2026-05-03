"""Kelly criterion utilities for bet sizing.

Quarter-Kelly is the standard for sports betting due to parameter uncertainty
(model probabilities are estimates, not ground truth). Full Kelly is optimal
under perfect information but leads to ruin-level volatility in practice.
"""
import numpy as np
import pandas as pd


def american_to_decimal(odds):
    """Convert American odds to decimal odds (includes stake).

    Examples: +100 → 2.0, -150 → 1.667, +200 → 3.0
    """
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 1 + odds / 100
    else:
        return 1 + 100 / abs(odds)


def kelly_fraction(p_model, decimal_odds):
    """Full Kelly fraction: optimal bet size as fraction of bankroll.

    f* = (p * b - q) / b  where b = decimal_odds - 1, q = 1 - p.
    Clipped to [0, 1] - never recommend shorting or betting > bankroll.
    """
    if pd.isna(p_model) or pd.isna(decimal_odds):
        return np.nan
    b = decimal_odds - 1
    if b <= 0:
        return 0.0
    f = (p_model * (b + 1) - 1) / b
    return float(np.clip(f, 0.0, 1.0))


def quarter_kelly(p_model, decimal_odds):
    """Quarter-Kelly: conservative bet size = 25% of full Kelly."""
    f = kelly_fraction(p_model, decimal_odds)
    if pd.isna(f):
        return np.nan
    return round(f * 0.25, 6)


def compute_kelly_row(p_model, american_odds):
    """Convenience: American odds in, (full_kelly, quarter_kelly) out."""
    dec = american_to_decimal(american_odds)
    fk = kelly_fraction(p_model, dec)
    qk = quarter_kelly(p_model, dec)
    return fk, qk
