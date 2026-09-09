"""+EV flagging and Kelly sizing."""
import logging
import os
from datetime import date

import numpy as np
import pandas as pd

from backend.kelly import american_to_decimal, compute_kelly_row, kelly_fraction
from backend.simulation import american_to_prob

log = logging.getLogger(__name__)


# Totals bar is higher; total-runs markets are noisier than sides.
EV_THRESHOLDS = {
    "ml": 0.045,
    "rl": 0.045,
    "totals": 0.065,
}

# Moneyline recommendation switch.
MONEYLINE_ENABLED = True

# Run-line recommendation switch.
RUNLINE_ENABLED = True

# Totals are off: every totals edge bucket lost money on the 2026 backtest and
# model edge had no relationship to outcome.
TOTALS_ENABLED = False

# Set to 0 for weather-off control runs.
WEATHER_ENABLED = os.getenv("MLBMODEL_WEATHER_ENABLED", "1") == "1"

# Weight of the sim in the logit-scale blend with the de-vigged market
# consensus. The raw sim prob flags big underdogs +EV systematically. 0.2 was
# the log-loss optimum on 2026 games (a 2026-09-09 sweep over 1272 v2 games
# put it at 0 and worsening monotonically); at 0.2 the 4.5% ML threshold
# needs a 20+ point sim/market disagreement and no ML pick surfaced for three
# weeks. Raised to 0.5 on 2026-09-09 by decision, accepting the calibration
# cost, so the moneyline market is not silently off. 0.5 is the highest
# weight at which the measured big-underdog failure (sim 40% vs market 28%)
# still stays No Play; see test_market_anchor_stops_flagging_big_dogs.
MARKET_ANCHOR_W_MODEL = 0.5

# Home-field shift on the sim's home logit, applied before market anchoring.
# The sim itself has none: batting last and walkoff logic net to ~zero.
HOME_FIELD_LOGIT = 0.09

# v1 → v2 model cutover. Eval / calibration / feature-importance writes are
# blocked before this date so the v1-archive history stays frozen.
V1_CUTOVER_DATE = date(2026, 5, 12)


# Warn once per distinct error rather than once per row.
_flag_warnings_seen: set[tuple[str, str]] = set()


def _warn_flag_error(fn_name: str, exc: Exception) -> None:
    key = (fn_name, f"{type(exc).__name__}: {exc}")
    if key not in _flag_warnings_seen:
        _flag_warnings_seen.add(key)
        log.warning(f"{fn_name} raised {type(exc).__name__}: {exc} - returning 'No Play'")


def flag_ev(row, threshold=EV_THRESHOLDS["ml"]):
    if not MONEYLINE_ENABLED:
        return "No Play"
    try:
        our_prob = american_to_prob(row["our_odds"])
        book_prob = american_to_prob(row["moneyline"])
        if pd.isna(book_prob):
            return "No Play"
        edge = our_prob - book_prob
        return row["team"] if edge >= threshold else "No Play"
    except Exception as e:
        _warn_flag_error("flag_ev", e)
        return "No Play"


def flag_runline_ev(row, threshold=EV_THRESHOLDS["rl"]):
    if not RUNLINE_ENABLED:
        return "No Play"
    try:
        book_prob = american_to_prob(row["spread_odds"])
        model_prob = row.get("p_cover")
        if pd.isna(book_prob) or pd.isna(model_prob):
            return "No Play"
        edge = model_prob - book_prob
        return row["team"] if edge >= threshold else "No Play"
    except Exception as e:
        _warn_flag_error("flag_runline_ev", e)
        return "No Play"


def flag_total_play(row, threshold=EV_THRESHOLDS["totals"]):
    try:
        over_prob_book = american_to_prob(row.get("total_over_odds"))
        under_prob_book = american_to_prob(row.get("total_under_odds"))
        p_over = row.get("p_over")
        p_under = row.get("p_under")
        if pd.notna(p_over) and pd.notna(over_prob_book) and (p_over - over_prob_book) >= threshold:
            return "Over"
        if pd.notna(p_under) and pd.notna(under_prob_book) and (p_under - under_prob_book) >= threshold:
            return "Under"
        # Book over/under odds missing: fall back to runs-diff heuristic.
        if pd.isna(over_prob_book) and pd.isna(under_prob_book) and pd.notna(row.get("total_diff")):
            if row["total_diff"] >= 1:
                return "Over"
            if row["total_diff"] <= -1:
                return "Under"
        return "No Play"
    except Exception as e:
        _warn_flag_error("flag_total_play", e)
        return "No Play"


def apply_kelly_sizing(df):
    """Add kelly_{full,quarter}_{ml,rl,total} columns."""
    df = df.copy()

    # Moneyline Kelly: model prob vs book moneyline
    ml_kelly = df.apply(
        lambda row: compute_kelly_row(row["win_prob"], row["moneyline"]), axis=1
    )
    df["kelly_full_ml"] = ml_kelly.apply(lambda x: x[0])
    df["kelly_quarter_ml"] = ml_kelly.apply(lambda x: x[1])

    # Run line Kelly: model p_cover vs book spread odds
    rl_kelly = df.apply(
        lambda row: compute_kelly_row(row.get("p_cover"), row.get("spread_odds")), axis=1
    )
    df["kelly_full_rl"] = rl_kelly.apply(lambda x: x[0])
    df["kelly_quarter_rl"] = rl_kelly.apply(lambda x: x[1])

    # Totals Kelly: model p_over vs book over odds, model p_under vs book under odds
    df["kelly_full_total"] = df.apply(
        lambda row: (
            kelly_fraction(row.get("p_over"), american_to_decimal(row.get("total_over_odds")))
            if row.get("total_play") == "Over"
            else kelly_fraction(row.get("p_under"), american_to_decimal(row.get("total_under_odds")))
            if row.get("total_play") == "Under"
            else 0.0
        ),
        axis=1,
    )
    df["kelly_quarter_total"] = df["kelly_full_total"].apply(
        lambda x: round(x * 0.25, 6) if pd.notna(x) else np.nan
    )

    return df
