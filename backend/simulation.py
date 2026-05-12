"""Negative-binomial run-distribution math: takes lambdas + book lines, returns market probs."""
import numpy as np
import pandas as pd
from scipy.stats import nbinom


# r=6 calibrated to MLB historical run distributions. Variance = lambda + lambda^2 / r.
NBINOM_R = 6.0


def compute_game_probs(lambda_home, lambda_away, total_line=None, spread_home=None,
                       max_runs=25, r=NBINOM_R):
    """Joint NB distribution → {p_home_win, p_away_win, p_home_cover, p_away_cover, p_over, p_under}.

    Keys whose input is missing are set to None. Moneyline ties allocated proportionally
    to expected runs (extra-innings approximation); run-line / totals pushes split 50/50.
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
    """P(team A beats team B). Thin wrapper around compute_game_probs."""
    probs = compute_game_probs(lambda_a, lambda_b, max_runs=max_runs, r=r)
    return probs["p_home_win"]


poisson_win_prob = win_prob


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


def apply_market_probs(df):
    """Add p_cover, p_over, p_under columns per row from the joint NB distribution.

    Reads game_pk, team, is_home, xR, total, spread from df.
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
