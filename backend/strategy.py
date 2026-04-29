"""+EV flagging and Kelly sizing.

Reads model probabilities (win_prob, p_cover, p_over, p_under) and book odds,
emits per-row EV flags and Kelly stake fractions. Single source of truth for
the +EV thresholds.
"""
import pandas as pd
import numpy as np
from backend.simulation import american_to_prob
from backend.kelly import american_to_decimal, kelly_fraction, compute_kelly_row


# Single source of truth for +EV thresholds. Used by flag_ev / flag_runline_ev / flag_total_play.
# Totals bar is higher because total-runs markets are noisier than sides.
EV_THRESHOLDS = {
    "ml": 0.045,
    "rl": 0.045,
    "totals": 0.065,
}


# Dedupe warnings so a systematic issue doesn't spam stdout once per row.
_flag_warnings_seen: set[tuple[str, str]] = set()


def _warn_flag_error(fn_name: str, exc: Exception) -> None:
    key = (fn_name, f"{type(exc).__name__}: {exc}")
    if key not in _flag_warnings_seen:
        _flag_warnings_seen.add(key)
        print(f"  WARNING: {fn_name} raised {type(exc).__name__}: {exc} — returning 'No Play'")


def flag_ev(row, threshold=EV_THRESHOLDS["ml"]):
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
    """Flag a run line play if our model's cover probability (from the joint
    negative-binomial distribution, accounting for the book's actual spread)
    beats the book's implied cover probability by at least `threshold`.
    """
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
    """Flag Over/Under based on joint-distribution probabilities vs book odds."""
    try:
        over_prob_book = american_to_prob(row.get("total_over_odds"))
        under_prob_book = american_to_prob(row.get("total_under_odds"))
        p_over = row.get("p_over")
        p_under = row.get("p_under")
        if pd.notna(p_over) and pd.notna(over_prob_book) and (p_over - over_prob_book) >= threshold:
            return "Over"
        if pd.notna(p_under) and pd.notna(under_prob_book) and (p_under - under_prob_book) >= threshold:
            return "Under"
        # Fallback when book over/under odds missing: use diff heuristic so the UI
        # still surfaces directional model disagreement with the line.
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
    """Add Kelly full + quarter columns for moneyline, run line, and totals.

    Expects df to already have win_prob, p_cover, p_over, p_under,
    moneyline, spread_odds, total_over_odds, total_under_odds, total_play.
    """
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
