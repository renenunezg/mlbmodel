"""Comprehensive model evaluation metrics: regression, probabilistic, financial.

All functions are pure (no DB access) — they receive arrays/DataFrames and return
scalars or dicts. The caller (evaluate_model.py) is responsible for persistence.
"""
import numpy as np
import pandas as pd
from scipy.stats import nbinom

# ---------------------------------------------------------------------------
# Regression metrics (predicted runs vs actual runs)
# ---------------------------------------------------------------------------

def _finite_pair(y_true, y_pred):
    """Return (y_true, y_pred) filtered to rows where both values are finite."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return y_true[mask], y_pred[mask]


def mae(y_true, y_pred):
    yt, yp = _finite_pair(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp))) if yt.size else np.nan


def rmse(y_true, y_pred):
    yt, yp = _finite_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2))) if yt.size else np.nan


def mape(y_true, y_pred):
    yt, yp = _finite_pair(y_true, y_pred)
    nonzero = yt != 0
    if not nonzero.any():
        return np.nan
    return float(np.mean(np.abs((yt[nonzero] - yp[nonzero]) / yt[nonzero])) * 100)


def r_squared(y_true, y_pred):
    yt, yp = _finite_pair(y_true, y_pred)
    if yt.size < 2:
        return np.nan
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def residual_stats(y_true, y_pred):
    yt, yp = _finite_pair(y_true, y_pred)
    residuals = yp - yt
    if len(residuals) < 2:
        return {"mean": np.nan, "std": np.nan, "skew": np.nan}
    return {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals, ddof=1)),
        "skew": float(pd.Series(residuals).skew()),
    }


def regression_summary(y_true, y_pred):
    """All regression metrics in one dict."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
        **{f"residual_{k}": v for k, v in residual_stats(y_true, y_pred).items()},
    }


# ---------------------------------------------------------------------------
# Probabilistic metrics (win_prob vs binary outcome)
# ---------------------------------------------------------------------------

def brier_score(probs, outcomes):
    """Mean squared error between predicted probabilities and binary outcomes."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    mask = ~(np.isnan(probs) | np.isnan(outcomes))
    if not mask.any():
        return np.nan
    return float(np.mean((probs[mask] - outcomes[mask]) ** 2))


def log_loss(probs, outcomes, eps=1e-7):
    """Binary cross-entropy with safe clipping to avoid log(0)."""
    probs = np.clip(np.asarray(probs, dtype=float), eps, 1 - eps)
    outcomes = np.asarray(outcomes, dtype=float)
    mask = ~(np.isnan(probs) | np.isnan(outcomes))
    if not mask.any():
        return np.nan
    p, y = probs[mask], outcomes[mask]
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def sharpness(probs):
    """Variance of predicted probabilities — measures decisiveness."""
    probs = np.asarray(probs, dtype=float)
    valid = probs[~np.isnan(probs)]
    return float(np.var(valid)) if len(valid) > 0 else np.nan


def calibration_curve(probs, outcomes, n_bins=10):
    """Bin predictions into deciles and return predicted vs observed rates.

    Returns list of dicts: [{bin_mid, predicted_mean, observed_rate, count}, ...]
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    mask = ~(np.isnan(probs) | np.isnan(outcomes))
    probs, outcomes = probs[mask], outcomes[mask]
    if len(probs) == 0:
        return []

    bins = np.linspace(0, 1, n_bins + 1)
    result = []
    for i in range(n_bins):
        in_bin = (probs >= bins[i]) & (probs < bins[i + 1])
        if i == n_bins - 1:
            in_bin |= (probs == bins[i + 1])
        count = int(in_bin.sum())
        if count == 0:
            continue
        result.append({
            "bin_mid": round(float((bins[i] + bins[i + 1]) / 2), 2),
            "predicted_mean": round(float(probs[in_bin].mean()), 4),
            "observed_rate": round(float(outcomes[in_bin].mean()), 4),
            "count": count,
        })
    return result


def prediction_interval_coverage(xr_preds, actual_runs, level=0.80, r=6.0):
    """Fraction of actual outcomes inside the predicted interval from the NB distribution.

    For each xR (expected runs), compute the [lower, upper] bounds of the NB
    distribution that contain `level` probability mass. Then check what fraction
    of actual runs fall inside.
    """
    xr = np.asarray(xr_preds, dtype=float)
    actual = np.asarray(actual_runs, dtype=float)
    mask = ~(np.isnan(xr) | np.isnan(actual))
    xr, actual = xr[mask], actual[mask]
    if len(xr) == 0:
        return np.nan

    alpha = (1 - level) / 2
    inside = 0
    for mu, y in zip(xr, actual):
        mu = max(mu, 0.5)
        p = r / (r + mu)
        lo = nbinom.ppf(alpha, r, p)
        hi = nbinom.ppf(1 - alpha, r, p)
        if lo <= y <= hi:
            inside += 1
    return round(inside / len(xr), 4)


def probabilistic_summary(probs, outcomes, xr_preds=None, actual_runs=None):
    """All probabilistic metrics in one dict."""
    result = {
        "brier_score": brier_score(probs, outcomes),
        "log_loss": log_loss(probs, outcomes),
        "sharpness": sharpness(probs),
    }
    if xr_preds is not None and actual_runs is not None:
        result["interval_coverage_80"] = prediction_interval_coverage(xr_preds, actual_runs)
    return result


# ---------------------------------------------------------------------------
# Financial / betting metrics
# ---------------------------------------------------------------------------

def roi(ledger):
    """Net profit / total staked. ledger: DataFrame with columns [stake, payout]."""
    if ledger.empty:
        return np.nan
    total_staked = ledger["stake"].sum()
    if total_staked == 0:
        return np.nan
    net = ledger["payout"].sum() - total_staked
    return round(float(net / total_staked), 4)


def _daily_returns(ledger):
    """Compute daily P&L returns from a bet ledger with 'date', 'stake', 'payout'."""
    if ledger.empty:
        return pd.Series(dtype=float)
    daily = ledger.groupby("date").apply(
        lambda g: (g["payout"].sum() - g["stake"].sum()) / g["stake"].sum()
        if g["stake"].sum() > 0 else 0.0
    )
    return daily


def sharpe_ratio(ledger, annual_factor=252):
    """Annualized Sharpe ratio from daily returns."""
    dr = _daily_returns(ledger)
    if len(dr) < 2:
        return np.nan
    mu = dr.mean()
    sigma = dr.std()
    if sigma == 0:
        return np.nan
    return round(float(mu / sigma * np.sqrt(annual_factor)), 4)


def sortino_ratio(ledger, annual_factor=252):
    """Annualized Sortino ratio (only penalizes downside volatility)."""
    dr = _daily_returns(ledger)
    if len(dr) < 2:
        return np.nan
    mu = dr.mean()
    downside = dr[dr < 0]
    if len(downside) < 1:
        return np.nan if mu == 0 else np.inf
    downside_std = downside.std()
    if downside_std == 0:
        return np.nan
    return round(float(mu / downside_std * np.sqrt(annual_factor)), 4)


def max_drawdown(equity_curve):
    """Maximum peak-to-trough percentage decline.

    equity_curve: array-like of cumulative equity values.
    """
    eq = np.asarray(equity_curve, dtype=float)
    if len(eq) < 2:
        return np.nan
    running_max = np.maximum.accumulate(eq)
    drawdowns = (eq - running_max) / running_max
    return round(float(drawdowns.min()), 4)


def equity_curve_from_ledger(ledger, initial=1.0):
    """Build a cumulative equity curve from a bet ledger.

    Returns a DataFrame with columns [date, equity].
    """
    if ledger.empty:
        return pd.DataFrame(columns=["date", "equity"])
    daily = ledger.groupby("date").agg(
        staked=("stake", "sum"),
        returned=("payout", "sum"),
    ).sort_index()
    daily["pnl"] = daily["returned"] - daily["staked"]
    daily["equity"] = initial + daily["pnl"].cumsum()
    return daily.reset_index()[["date", "equity"]]


def hit_rate_by_edge_bucket(ledger, buckets=(0.03, 0.05, 0.10, 0.20)):
    """Hit rate and ROI per edge bucket.

    ledger must have columns: edge, won (bool), stake, payout.
    Returns list of dicts: [{bucket_label, n_bets, hit_rate, roi}, ...]
    """
    if ledger.empty or "edge" not in ledger.columns:
        return []

    edges = sorted(buckets)
    results = []
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else np.inf
        label = f"{int(lo*100)}-{int(hi*100)}%" if hi != np.inf else f"{int(lo*100)}%+"
        mask = (ledger["edge"] >= lo) & (ledger["edge"] < hi)
        subset = ledger[mask]
        if subset.empty:
            continue
        total_staked = subset["stake"].sum()
        bucket_roi = (subset["payout"].sum() - total_staked) / total_staked if total_staked > 0 else 0
        results.append({
            "bucket_label": label,
            "n_bets": int(len(subset)),
            "hit_rate": round(float(subset["won"].mean()), 4),
            "roi": round(float(bucket_roi), 4),
        })
    return results


def financial_summary(ledger):
    """All financial metrics in one dict."""
    eq = equity_curve_from_ledger(ledger)
    return {
        "roi": roi(ledger),
        "sharpe": sharpe_ratio(ledger),
        "sortino": sortino_ratio(ledger),
        "max_drawdown": max_drawdown(eq["equity"].values) if not eq.empty else np.nan,
        "total_staked_units": round(float(ledger["stake"].sum()), 4) if not ledger.empty else 0,
        "net_profit_units": round(float(ledger["payout"].sum() - ledger["stake"].sum()), 4) if not ledger.empty else 0,
    }
