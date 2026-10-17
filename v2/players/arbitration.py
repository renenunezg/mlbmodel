"""Unsigned salaries fitted to reported arbitration agreements, not FA fractions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, minimize

from v2.players import market_inputs, pricing, production

FEATURES = ["bat_war", "bat_pa_600", "pit_war", "pit_bf_600", "saves_10"]
METHOD = "arbitration_service_workload_v1"


def _features(wars: dict, profile: dict, service: float) -> np.ndarray:
    stage = int(np.floor(service))
    if stage not in range(2, 6):
        raise ValueError("Arbitration estimate requires two to five completed service years")
    inputs = [wars.get("bat", 0), profile["bat_pa"]/600,
              sum(w for role, w in wars.items() if role in {"SP", "RP"}),
              profile["pit_pa"]/600, profile["expected_saves"]/10]
    if not np.isfinite(inputs).all() or min(inputs) < 0:
        raise ValueError("Invalid arbitration production inputs")
    x = np.zeros(20)
    x[(stage-2)*5:(stage-1)*5] = inputs
    return x


def estimate(fitted: dict, wars: dict, profile: dict, service: float, season: int) -> float:
    """Estimate a future tender salary; the caller preserves quoted cash and floors."""
    if fitted.get("method") != METHOD or fitted.get("features") != FEATURES:
        raise ValueError("Missing or incompatible arbitration model")
    if season <= fitted["last_signing_season"]:
        raise ValueError("Arbitration estimate overlaps salary training")
    beta = np.asarray(fitted["coefficients"], dtype=float)
    if beta.shape != (20,) or not np.isfinite(beta).all() or (beta < 0).any():
        raise ValueError("Invalid arbitration salary coefficients")
    return float(pricing.league_minimum(season) + _features(wars, profile, service)@beta
                 * (1+fitted["inflation"])**(season-fitted["base_season"]))


def calibration_rows(models: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Build every eligible signing using that winter's frozen forecast models."""
    rows, coverage = [], []
    for first, model in sorted(models.items()):
        cohort = json.loads((Path(__file__).parent / "data" / f"arbitration_signings_{first}.json").read_text())
        if model.production["last_target_season"] >= first:
            raise ValueError("Arbitration forecast overlaps training outcomes")
        pt = market_inputs.historical_profiles(model.workload, first, first)
        pt = production.add_forecasts(pt, model.production, first, first)
        excluded = list(cohort["excluded"])
        for c in cohort["contracts"]:
            player = pt[pt.player_id.eq(c["player_id"]) & pt.expected_war.notna()]
            if player.empty or player.role.nunique() != 1:
                excluded.append({"name": c["name"], "reason": "No supported single-role MLB production forecast"})
                continue
            p = player.iloc[0]
            profile = {k: float(p[k]) for k in market_inputs.PROFILE_COLUMNS}
            wars = {p.pricing_role: max(0.0, float(p.expected_war))}
            rows.append({**c, "war_by_role": wars, "profile": profile,
                         "design": _features(wars, profile, c["service_time"]),
                         "training_through": model.production["last_target_season"]})
        coverage.append({"first_season": first, "source": cohort["source"], "selection": cohort["selection"],
                         "included": sum(r["first_season"] == first for r in rows), "excluded": excluded})
    return pd.DataFrame(rows), coverage


def fit(frame: pd.DataFrame, base_season: int) -> dict:
    """Pool service-year salary coefficients with player-grouped validation.

    All coefficients are nonnegative. Pooling borrows information across service
    groups while retaining their different salary scales. The target is actual
    one-year agreed salary above the minimum. The input WAR and workload are
    forecasts frozen before the signing winter, exactly as used for future costs.
    """
    if len(frame) < 100 or frame.duplicated(["player_id", "first_season"]).any():
        raise ValueError("Arbitration calibration requires unique, identified salary agreements")
    x = np.stack(frame.design)
    minimum = np.array([pricing.league_minimum(y) for y in frame.first_season])
    y = (frame.salary_musd.to_numpy()-minimum)/(1+pricing.FALLBACK_INFLATION)**(frame.first_season.to_numpy()-base_season)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite arbitration calibration data")
    ids = frame.player_id.to_numpy()
    contrast = np.kron(np.eye(4)-np.ones((4, 4))/4, np.eye(5))

    def fit_mask(mask, pars):
        reg = np.vstack([np.sqrt(np.exp(pars[0]))*contrast, np.sqrt(np.exp(pars[1]))*np.eye(20)])
        result = lsq_linear(np.vstack([x[mask], reg]), np.r_[y[mask], np.zeros(40)], bounds=(0, np.inf))
        if not result.success:
            raise ValueError("Arbitration salary fit did not converge")
        return result.x

    def errors(pars):
        result = np.empty(len(y))
        for pid in np.unique(ids):
            held = ids == pid
            result[held] = x[held]@fit_mask(~held, pars)-y[held]
        return result

    optimum = minimize(lambda pars: np.mean(errors(pars)**2), [0, 0], method="Powell", bounds=[(-10, 15)]*2,
                       options={"xtol": 1e-3, "ftol": 1e-5})
    if not optimum.success:
        raise ValueError("Arbitration regularization did not converge")
    beta = fit_mask(np.ones(len(y), dtype=bool), optimum.x)
    return {"method": METHOD, "features": FEATURES, "coefficients": beta.tolist(),
            "base_season": base_season, "last_signing_season": int(frame.first_season.max()),
            "inflation": pricing.FALLBACK_INFLATION, "n": len(frame),
            "regularization": np.exp(optimum.x).tolist(),
            "loo_rmse_musd": float(np.sqrt(np.mean(errors(optimum.x)**2))),
            "selection": "Leave-one-player-out penalties, all signings by held player excluded"}


def calibrate(models: dict) -> tuple[dict, dict]:
    frame, coverage = calibration_rows(models)
    fitted = fit(frame, int(frame.first_season.max()))
    return fitted, {"cohorts": coverage, "fit": fitted,
                    "limitations": ["One-year reported agreements only; unresolved hearings remain excluded",
                                    "Future service assumes full seasons; salary floors and non-tenders apply in the contract path"]}
