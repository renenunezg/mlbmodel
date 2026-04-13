"""Tests for regression, probabilistic, and financial metrics."""
import numpy as np
import pandas as pd
from backend.metrics import (
    mae, rmse, mape, r_squared, brier_score, log_loss,
    sharpness, calibration_curve, roi, max_drawdown,
    equity_curve_from_ledger, hit_rate_by_edge_bucket,
)


# --- Regression ---

def test_mae_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0


def test_mae_known():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    assert mae(y_true, y_pred) == 1.0


def test_rmse_known():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    assert abs(rmse(y_true, y_pred) - 1.0) < 1e-9


def test_r_squared_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(r_squared(y, y) - 1.0) < 1e-9


def test_r_squared_mean_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.full_like(y, y.mean())
    assert abs(r_squared(y, y_pred)) < 1e-9  # R² = 0 when predicting mean


# --- Probabilistic ---

def test_brier_score_perfect():
    probs = np.array([1.0, 0.0, 1.0])
    outcomes = np.array([1.0, 0.0, 1.0])
    assert brier_score(probs, outcomes) == 0.0


def test_brier_score_baseline():
    # Always predicting 0.5 on fair coin: Brier = 0.25
    probs = np.full(1000, 0.5)
    outcomes = np.random.choice([0.0, 1.0], size=1000, p=[0.5, 0.5])
    bs = brier_score(probs, outcomes)
    assert abs(bs - 0.25) < 0.02  # within noise


def test_log_loss_perfect():
    probs = np.array([0.99, 0.01, 0.99])
    outcomes = np.array([1.0, 0.0, 1.0])
    ll = log_loss(probs, outcomes)
    assert ll < 0.05  # near zero for near-perfect


def test_sharpness_all_same():
    probs = np.full(100, 0.5)
    assert sharpness(probs) == 0.0


def test_sharpness_spread():
    probs = np.array([0.1, 0.9])
    assert sharpness(probs) > 0


def test_calibration_curve_basic():
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    outcomes = np.array([0, 0, 1, 1])
    bins = calibration_curve(probs, outcomes, n_bins=2)
    assert len(bins) == 2
    assert bins[0]["observed_rate"] == 0.0  # both low-prob bets lost
    assert bins[1]["observed_rate"] == 1.0  # both high-prob bets won


# --- Financial ---

def test_roi_breakeven():
    ledger = pd.DataFrame({"stake": [1.0, 1.0], "payout": [2.0, 0.0]})
    assert roi(ledger) == 0.0  # net = 0


def test_roi_profit():
    ledger = pd.DataFrame({"stake": [1.0, 1.0], "payout": [2.5, 0.0]})
    assert roi(ledger) == 0.25  # net = 0.5 / 2.0 staked


def test_max_drawdown_flat():
    eq = np.array([1.0, 1.0, 1.0])
    assert max_drawdown(eq) == 0.0


def test_max_drawdown_known():
    eq = np.array([1.0, 1.1, 0.9, 1.0])
    # Peak = 1.1, trough = 0.9 → DD = (0.9-1.1)/1.1 ≈ -0.1818
    dd = max_drawdown(eq)
    assert abs(dd - (-0.1818)) < 0.001


def test_equity_curve_from_ledger():
    ledger = pd.DataFrame({
        "date": pd.to_datetime(["2025-04-01", "2025-04-01", "2025-04-02"]),
        "stake": [0.05, 0.05, 0.05],
        "payout": [0.10, 0.0, 0.10],
    })
    eq = equity_curve_from_ledger(ledger)
    assert len(eq) == 2
    assert eq.iloc[0]["equity"] == 1.0  # day 1: +0.10 - 0.10 = 0 net → 1.0
    assert eq.iloc[1]["equity"] == 1.05  # day 2: +0.05 → 1.05


def test_hit_rate_by_edge_bucket():
    ledger = pd.DataFrame({
        "edge": [0.04, 0.06, 0.12, 0.25],
        "won": [True, False, True, True],
        "stake": [1, 1, 1, 1],
        "payout": [2, 0, 2, 2],
    })
    buckets = hit_rate_by_edge_bucket(ledger, buckets=(0.03, 0.05, 0.10, 0.20))
    assert len(buckets) >= 2
    labels = [b["bucket_label"] for b in buckets]
    assert "3-5%" in labels
