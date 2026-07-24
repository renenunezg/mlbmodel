import numpy as np
import pandas as pd
from backend.metrics import (
    brier_score, calibration_curve, equity_curve_from_ledger,
    hit_rate_by_edge_bucket, log_loss, mae, max_drawdown, r_squared, rmse,
    roi, sharpness,
)


def test_regression_metrics_known():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    assert mae(y_true, y_pred) == 1.0
    assert abs(rmse(y_true, y_pred) - 1.0) < 1e-9
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.full_like(y, y.mean())
    assert abs(r_squared(y, y_pred)) < 1e-9
    probs = np.array([0.1, 0.9])
    assert sharpness(probs) > 0


def test_probability_metrics_and_calibration():
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    outcomes = np.array([0, 0, 1, 1])
    assert brier_score(probs, outcomes) < 0.03
    assert log_loss(probs, outcomes) < 0.2
    bins = calibration_curve(probs, outcomes, n_bins=2)
    assert len(bins) == 2
    assert bins[0]["observed_rate"] == 0.0
    assert bins[1]["observed_rate"] == 1.0


def test_roi():
    breakeven = pd.DataFrame({"stake": [1.0, 1.0], "payout": [2.0, 0.0]})
    profit = pd.DataFrame({"stake": [1.0, 1.0], "payout": [2.5, 0.0]})
    assert roi(breakeven) == 0.0
    assert roi(profit) == 0.25


def test_max_drawdown_known():
    eq = np.array([1.0, 1.1, 0.9, 1.0])
    # Peak 1.1 to trough 0.9 = -0.2u (absolute, since stakes are flat 1u-based).
    dd = max_drawdown(eq)
    assert abs(dd - (-0.2)) < 1e-9


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
