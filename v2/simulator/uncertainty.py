"""Separate between-posterior variability from finite-simulation error."""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def win_probability_uncertainty(probabilities: list[float], n_per_draw: int) -> dict:
    """Moment-corrected posterior spread with a Monte Carlo resolution gate.

    Each independent draw's win fraction has conditional binomial variance.
    Subtract its unbiased estimate from the observed between-draw variance.
    Do not publish parameter bands when that variance cannot be distinguished
    from simulation noise at the 95% level. These are estimated posterior
    quantiles, not an interval covering market or structural model error.
    """
    p = np.asarray(probabilities, dtype=float)
    if len(p) < 3 or n_per_draw < 2 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("At least three valid posterior estimates and two simulations per draw are required")
    observed = float(np.var(p, ddof=1))
    noise = float(np.mean(p * (1 - p) / (n_per_draw - 1)))
    signal = max(0.0, observed - noise)
    resolved = noise > 0 and observed > noise * chi2.ppf(0.95, len(p) - 1) / (len(p) - 1)
    lo = hi = None
    if resolved:
        adjusted = p.mean() + (p - p.mean()) * np.sqrt(signal / observed)
        lo, hi = np.clip(np.quantile(adjusted, [.1, .9]), 0, 1).tolist()
    return {
        "method": "binomial-variance-correction-v1",
        "status": "resolved" if resolved else "below_mc_resolution",
        "p10": lo, "p90": hi,
        "between_draw_variance": observed,
        "mc_variance_per_draw": noise,
        "parameter_variance_estimate": signal,
        "conditional_simulation_mcse": float(np.sqrt(noise / len(p))),
        "posterior_integration_mcse": float(np.sqrt(signal / len(p))),
        "raw_probability_mcse": float(np.sqrt((noise + signal) / len(p))),
        "draws": len(p), "sims_per_draw": n_per_draw,
    }
