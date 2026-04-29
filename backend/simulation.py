"""Negative-binomial run-distribution simulation and market probability math.

Pure math: takes expected runs (lambdas) and book lines (total, spread) and
returns the joint-distribution probabilities used for moneyline, run line,
and totals EV. No DB access, no model artifacts.
"""
import numpy as np
import pandas as pd
from scipy.stats import nbinom


# Negative binomial dispersion parameter. Variance = lambda + lambda^2 / r.
# r=6 is calibrated from MLB historical run distributions; it adds realistic
# overdispersion compared to plain Poisson, producing less extreme win probs.
NBINOM_R = 6.0


def compute_game_probs(lambda_home, lambda_away, total_line=None, spread_home=None,
                       max_runs=25, r=NBINOM_R):
    """Full game-level probability dict from the joint negative-binomial run distribution.

    Replaces the old magic `win_prob - 0.10` heuristic for run line and the
    `±1 run` heuristic for totals — both are now derived directly from the joint
    distribution, which naturally tightens in high-scoring environments and
    widens in low-scoring ones.

    Args:
        lambda_home: expected runs for home team
        lambda_away: expected runs for away team
        total_line: book total runs line (for over/under). None → skip.
        spread_home: signed spread from home team's perspective
            (e.g. -1.5 = home favored by 1.5, +1.5 = home underdog). None → skip.
        max_runs: truncation point for the run distribution (25 covers 99.9%+ of games)
        r: negative binomial dispersion parameter (higher r → less overdispersion)

    Returns dict with keys: p_home_win, p_away_win, p_home_cover, p_away_cover,
    p_over, p_under. Keys whose input is missing are set to None.
    Ties on the moneyline are allocated proportionally to expected runs
    (extra-innings approximation). Pushes on run line / totals are split 50/50.
    """
    runs = np.arange(max_runs + 1)
    h_probs = nbinom.pmf(runs, r, r / (r + lambda_home))
    a_probs = nbinom.pmf(runs, r, r / (r + lambda_away))
    # joint[h, a] = P(home scores h, away scores a)
    joint = np.outer(h_probs, a_probs)

    # Moneyline with proportional tie allocation
    p_home_outright = float(np.tril(joint, k=-1).sum())  # h > a
    p_away_outright = float(np.triu(joint, k=1).sum())   # a > h
    p_tie = float(np.trace(joint))
    total_lambda = lambda_home + lambda_away
    home_tie_share = lambda_home / total_lambda if total_lambda > 0 else 0.5
    p_home_win = p_home_outright + p_tie * home_tie_share
    p_away_win = p_away_outright + p_tie * (1 - home_tie_share)

    result = {
        "p_home_win": round(p_home_win, 4),
        "p_away_win": round(p_away_win, 4),
        "p_home_cover": None,
        "p_away_cover": None,
        "p_over": None,
        "p_under": None,
    }

    h_mat = runs[:, None]
    a_mat = runs[None, :]

    # Run line: home "covers" if home_runs + spread_home > away_runs. Pushes split 50/50.
    if spread_home is not None and not pd.isna(spread_home):
        margin = h_mat - a_mat
        p_home_strict = float(joint[margin > -spread_home].sum())
        p_push = float(joint[margin == -spread_home].sum())
        p_away_strict = float(joint[margin < -spread_home].sum())
        result["p_home_cover"] = round(p_home_strict + 0.5 * p_push, 4)
        result["p_away_cover"] = round(p_away_strict + 0.5 * p_push, 4)

    # Totals: over/under the book's total_line. Pushes split 50/50.
    if total_line is not None and not pd.isna(total_line):
        total_runs = h_mat + a_mat
        p_over_strict = float(joint[total_runs > total_line].sum())
        p_under_strict = float(joint[total_runs < total_line].sum())
        p_push_total = float(joint[total_runs == total_line].sum())
        result["p_over"] = round(p_over_strict + 0.5 * p_push_total, 4)
        result["p_under"] = round(p_under_strict + 0.5 * p_push_total, 4)

    return result


def win_prob(lambda_a, lambda_b, max_runs=15, r=NBINOM_R):
    """Backward-compatible: returns P(team A beats team B) as a float.

    Thin wrapper around compute_game_probs so existing callers (backtest.py,
    compute_predictions) continue to work. See compute_game_probs for the full
    joint-distribution output.
    """
    probs = compute_game_probs(lambda_a, lambda_b, max_runs=max_runs, r=r)
    return probs["p_home_win"]


# Backward-compatible alias used by backtest.py
poisson_win_prob = win_prob


def convert_to_odds(p):
    """Convert win probability to American odds."""
    if p < 0.5:
        return round(((1 - p) / p) * 100)
    elif p > 0.5:
        return round(-(p / (1 - p)) * 100) if p < 1 else -1000
    else:
        return 100


def american_to_prob(odds):
    """Convert American odds to implied probability."""
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def apply_market_probs(df):
    """Compute p_cover / p_over / p_under per row from the joint NB distribution,
    using the book's actual total_line and spread (pulled from the row's odds columns).

    Expected columns on df: game_pk, team, is_home, xR, total, spread.
    Adds columns: p_cover, p_over, p_under.
    """
    df = df.copy()
    df["p_cover"] = np.nan
    df["p_over"] = np.nan
    df["p_under"] = np.nan

    for game_pk, group in df.groupby("game_pk"):
        if len(group) != 2:
            continue
        rows = group.sort_values("is_home", ascending=True)
        away_row = rows.iloc[0]
        home_row = rows.iloc[1]
        lambda_away = max(float(away_row["xR"]), 0.5)
        lambda_home = max(float(home_row["xR"]), 0.5)

        total_line = home_row.get("total")
        if pd.isna(total_line):
            total_line = None
        # spread is stored per-row, signed from that row's team perspective.
        # We want the spread from the HOME team's perspective.
        spread_home = home_row.get("spread")
        if pd.isna(spread_home):
            spread_home = None

        probs = compute_game_probs(
            lambda_home, lambda_away,
            total_line=total_line,
            spread_home=spread_home,
        )

        home_idx = rows.index[rows["is_home"] == 1]
        away_idx = rows.index[rows["is_home"] == 0]

        if probs["p_home_cover"] is not None:
            df.loc[home_idx, "p_cover"] = probs["p_home_cover"]
            df.loc[away_idx, "p_cover"] = probs["p_away_cover"]
        if probs["p_over"] is not None:
            df.loc[group.index, "p_over"] = probs["p_over"]
            df.loc[group.index, "p_under"] = probs["p_under"]

    return df
