"""Odds conversions retained for ingestion and research consumers."""
import numpy as np
import pandas as pd


def convert_to_odds(p):
    """Win probability to American odds."""
    if p < 0.5:
        return round(((1 - p) / p) * 100)
    elif p > 0.5:
        return round(-(p / (1 - p)) * 100) if p < 1 else -1000
    else:
        return 100


def american_to_prob(odds):
    """American odds to implied probability."""
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)
