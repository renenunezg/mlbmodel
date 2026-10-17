"""Dollar calibration from verified signing events, plus unsigned salary rules."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, minimize, minimize_scalar

# Fallbacks, each cited in the tip when used.
FALLBACK_PRICE_MUSD = 11.0
FALLBACK_PRICE_SOURCE = "fangraphs_2025_26"
FALLBACK_INFLATION = 0.03
FALLBACK_ARB_FRACTIONS = {1: 0.25, 2: 0.40, 3: 0.60, 4: 0.70}
# CBA minimum salary by season, in millions; later seasons use the latest known.
LEAGUE_MINIMUM_MUSD: dict[int, float] = {2022: 0.70, 2023: 0.72, 2024: 0.74, 2025: 0.76, 2026: 0.78}
ARBITRATION_SERVICE = 3.0


@dataclass(frozen=True)
class PriceModel:
    price_musd: float
    se: float | None
    inflation: float
    n: int
    source: str
    base_season: int | None = None
    above_minimum: bool = False
    role_prices: dict[str, float] = field(default_factory=dict)
    profile_prices: dict[str, float] = field(default_factory=dict)

    def price_for(self, season: int, first_season: int, role: str | None = None) -> float:
        base = first_season if self.base_season is None else self.base_season
        if self.role_prices and role not in self.role_prices:
            raise ValueError("Role-calibrated dollars require a supported production role")
        rate = self.role_prices[role] if self.role_prices else self.price_musd
        return rate * (1.0 + self.inflation) ** (season - base)

    def value_for(self, war: float, season: int, first_season: int, role: str | None = None) -> float:
        if self.profile_prices:
            raise ValueError("Market pricing requires value_for_roles with a dated usage profile")
        return war * self.price_for(season, first_season, role) + (league_minimum(season) if self.above_minimum else 0.0)

    def value_for_roles(self, wars: dict[str, float], season: int, first_season: int,
                        profile: dict[str, float] | None = None) -> float:
        """Price both sides of a two-way player, paying the minimum salary once."""
        amount = sum(w * self.price_for(season, first_season, r) for r, w in wars.items())
        if self.profile_prices:
            if profile is None or "expected_saves" not in profile or not np.isfinite(list(profile.values())).all():
                raise ValueError("Market pricing requires a complete dated usage profile")
            base = first_season if self.base_season is None else self.base_season
            extra = (self.profile_prices["expected_saves"] * profile["expected_saves"] / 10 if "RP" in wars else 0)
            amount = max(0.0, amount + extra * (1+self.inflation)**(season-base))
        return amount + (league_minimum(season) if self.above_minimum else 0.0)


def replacement_present_value(first: int, years: int, discount: float) -> float:
    """Replacement salaries cover playing years, excluding later bonus payments."""
    return sum(league_minimum(first+i) * (1-discount)**i for i in range(years))


def contract_roles(contract: dict, wars: dict[str, float]) -> dict[str, float]:
    """Use a documented signing role consistently in calibration and replay.

    A short injury-return relief stint does not turn a new rotation contract
    into a relief contract. This changes pricing context, never projected WAR.
    Historical workload remains the default when the signing report is silent.
    """
    role = contract.get("signing_role")
    if role is None:
        return wars
    if role not in {"SP", "RP"} or not contract.get("role_source") or not contract.get("date"):
        raise ValueError("Signing role requires a dated source and a supported pitching role")
    if len(wars) != 1 or not set(wars) <= {"SP", "RP"}:
        raise ValueError("Signing role conflicts with projected production roles")
    return {role: sum(wars.values())}


def league_minimum(season: int) -> float:
    known = max(s for s in LEAGUE_MINIMUM_MUSD if s <= season) if any(s <= season for s in LEAGUE_MINIMUM_MUSD) else min(LEAGUE_MINIMUM_MUSD)
    return LEAGUE_MINIMUM_MUSD[known]


def contract_present_values(contract: dict, wars: list[float], discount: float, inflation: float) -> tuple[float, float]:
    """Compare the same guaranteed production horizon and dated cash obligations.

    Bonuses and buyouts belong in their payment years, including years after play
    ends. Opt-outs and exercisable options must be valued separately, not flattened
    into AAV. The caller supplies actual-workload WAR, never standardized WAR.
    """
    if contract.get("transaction_type") != "free_agent_signing":
        raise ValueError("Dollar calibration requires verified signing events, not how_acquired")
    if contract.get("has_exercisable_option", False):
        raise ValueError("Option contracts cannot enter the guaranteed-cash price regression")
    if not contract.get("source") or not contract.get("date"):
        raise ValueError("Signing event needs source and date")
    years = int(contract["years"])
    cash = np.asarray(contract["cash_musd"], dtype=float)
    war = np.asarray(wars, dtype=float)
    if years < 1 or len(war) != years or not len(cash):
        raise ValueError("Contract and production horizons do not match")
    if not np.isfinite(cash).all() or not np.isfinite(war).all() or (cash < 0).any() or (war < 0).any():
        raise ValueError("Non-finite or negative contract inputs")
    if not np.isclose(cash.sum(), contract["guarantee_musd"]):
        raise ValueError("Cash does not reconcile to the guarantee")
    cost = float(cash @ (1 - discount) ** np.arange(len(cash)))
    production = float(war @ ((1 - discount) * (1 + inflation)) ** np.arange(years))
    return cost, production


def fit_price_per_win(present_values: pd.DataFrame, base_season: int, inflation: float = FALLBACK_INFLATION) -> PriceModel:
    """Fit salary above replacement against production PV in base-season dollars.

    One row per verified signing: player_id, first_season, salary_pv_musd,
    production_pv_wins, replacement_pv_musd. Zero-WAR contracts stay in the errors
    and sample count. The minimum salary is an offset, not a fitted intercept.
    No silent fallback: an unidentified or degenerate fit is an error, and a
    successful fit still requires independent market validation before publication.
    """
    keys = ["player_id", "first_season"]
    if present_values.duplicated(keys).any():
        raise ValueError("Duplicate contract in dollar calibration")
    x = present_values["production_pv_wins"].to_numpy(float)
    y = (present_values["salary_pv_musd"] - present_values["replacement_pv_musd"]).to_numpy(float)
    y = y / (1 + inflation) ** (present_values["first_season"].to_numpy(int) - base_season)
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all() or (x < 0).any() or (y < 0).any() or x @ x <= 0:
        raise ValueError("Insufficient identifiable contract data for dollar calibration")
    slope = float((x @ y) / (x @ x))
    resid = y - slope * x
    se = float(np.sqrt((resid @ resid) / (len(x) - 1) / (x @ x)))
    if slope <= 0:
        raise ValueError("Dollar calibration must be positive")
    return PriceModel(slope, se, inflation, len(x), "verified_contract_pv_above_minimum", base_season, True)


def fit_role_prices(frame: pd.DataFrame, base_season: int, inflation: float = FALLBACK_INFLATION) -> tuple[PriceModel, dict]:
    """Partially pool batting, starter and relief prices using verified cash PV.

    Annualize each contract's production and cash PV by its playing-year annuity.
    This retains the full guarantee without allowing an eleven-year contract to
    receive 121 times a one-year contract's weight solely because of its length.
    Choose pooling by leave-one-player-out error, excluding every signing by the
    held player. Subsequent seasons, never this fit, establish market accuracy.
    """
    if frame.duplicated(["player_id", "first_season"]).any():
        raise ValueError("Duplicate contract in dollar calibration")
    roles = ["bat", "SP", "RP"]
    if set(frame.role) != set(roles):
        raise ValueError("Calibration requires contracts for every production role")
    annuity = frame.annuity.to_numpy(float)
    x = frame.production_pv_wins.to_numpy(float) / annuity
    x *= (1+inflation)**(frame.first_season.to_numpy(int)-base_season)
    y = (frame.salary_pv_musd-frame.replacement_pv_musd).to_numpy(float) / annuity
    if len(x) < 10 or not np.isfinite([x, y, annuity]).all() or (annuity <= 0).any() or (x < 0).any() or (y < 0).any() or x@x <= 0:
        raise ValueError("Invalid or insufficient role-calibration contracts")
    X = x[:, None] * (frame.role.to_numpy()[:, None] == np.array(roles)[None, :])
    ids = frame.player_id.to_numpy()

    def fit(mask: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
        a, b, z = x[mask], y[mask], X[mask]
        if a@a <= 0:
            raise ValueError("Unidentified role-price fold")
        common = a@b/(a@a)
        beta = np.linalg.solve(z.T@z+alpha*np.eye(len(roles)), z.T@b+alpha*common)
        return beta, float(common)

    def errors(log_alpha: float) -> np.ndarray:
        prediction = np.empty(len(x))
        for pid in np.unique(ids):
            held = ids == pid
            beta, _ = fit(~held, np.exp(log_alpha))
            prediction[held] = X[held] @ beta
        return prediction-y

    optimum = minimize_scalar(lambda a: np.mean(errors(a)**2), bounds=(-10, 20), method="bounded")
    if not optimum.success:
        raise ValueError("Role-price pooling did not converge")
    alpha = float(np.exp(optimum.x))
    beta, common = fit(np.ones(len(x), dtype=bool), alpha)
    if (beta <= 0).any():
        raise ValueError("Role prices must be positive")
    residual = errors(optimum.x)
    price = PriceModel(common, float(np.sqrt(np.mean(residual**2)/(x@x))), inflation, len(frame),
                       "verified_contract_pv_role_pooled", base_season, True, dict(zip(roles, beta.tolist())))
    return price, {"pooling_alpha": alpha, "loo_annualized_rmse_musd": float(np.sqrt(np.mean(residual**2))),
                   "weighting": "One contract per row, cash and production divided by playing-year discounted annuity",
                   "selection": "Leave-one-player-out pooling; later seasons excluded from fitting"}


def fit_market_prices(frame: pd.DataFrame, base_season: int,
                      discount: float, inflation: float = FALLBACK_INFLATION) -> tuple[PriceModel, dict]:
    """Jointly fit role WAR prices and the relief usage premium to actual cash.

    Role prices pool toward each other, not toward a gross WAR price that already
    includes closer compensation. The usage premium shrinks toward zero.
    Leave-one-player-out selection excludes every signing by the held player.
    Subsequent signing winters and trades never select these coefficients.
    """
    roles = ["bat", "SP", "RP"]
    if frame.duplicated(["player_id", "first_season"]).any() or set(frame.role) != set(roles) or len(frame) < 10:
        raise ValueError("Market calibration needs unique signings covering all roles")
    x = np.zeros((len(frame), 4))
    for j, row in enumerate(frame.itertuples()):
        if len(row.projected_war) != row.years or len(row.projected_profiles) != row.years:
            raise ValueError("Market profile and contract horizons differ")
        for i, (war, profile) in enumerate(zip(row.projected_war, row.projected_profiles)):
            factor = (1+inflation)**(row.first_season+i-base_season)*(1-discount)**i/row.annuity
            x[j, roles.index(row.role)] += war*factor
            x[j, 3] += profile["expected_saves"]/10*factor if row.role == "RP" else 0
    y = (frame.salary_pv_musd-frame.replacement_pv_musd).to_numpy(float)/frame.annuity.to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (x < 0).any() or (y < 0).any() or (frame.annuity <= 0).any():
        raise ValueError("Non-finite or invalid cash calibration data")
    ids = frame.player_id.to_numpy()

    def fit(mask, pars):
        reg = np.zeros((4, 4))
        reg[:3, :3] = np.sqrt(np.exp(pars[0]))*(np.eye(3)-np.ones((3, 3))/3)
        reg[3, 3] = np.sqrt(np.exp(pars[1]))
        result = lsq_linear(np.vstack([x[mask], reg]), np.r_[y[mask], np.zeros(4)], bounds=(0, np.inf))
        if not result.success:
            raise ValueError("Contract price fit did not converge")
        return result.x

    def errors(pars):
        prediction = np.empty(len(y))
        for pid in np.unique(ids):
            held = ids == pid
            prediction[held] = x[held]@fit(~held, pars)
        return prediction-y

    optimum = minimize(lambda p: np.mean(errors(p)**2), [3, 0], method="Powell", bounds=[(-10, 20)]*2,
                       options={"xtol": 1e-4, "ftol": 1e-7})
    if not optimum.success:
        raise ValueError("Market price penalty selection did not converge")
    beta = fit(np.ones(len(y), dtype=bool), optimum.x)
    residual = errors(optimum.x)
    price = PriceModel(float(np.mean(beta[:3])), None, inflation, len(frame),
                       "verified_contract_war_and_relief_usage", base_season, True,
                       dict(zip(roles, beta[:3].tolist())), {"expected_saves": float(beta[3])})
    return price, {"pooling_alpha": float(np.exp(optimum.x[0])),
                   "profile_alpha": float(np.exp(optimum.x[1])),
                   "loo_annualized_rmse_musd": float(np.sqrt(np.mean(residual**2))),
                   "selection": "Leave-one-player-out penalties; subsequent signing winters excluded",
                   "weighting": "One signing per row, annualized discounted cash",
                   "profile_units": {"expected_saves": "Ten expected saves"}}


def arbitration_start(service_time: float, service_season: int, arbitration_seasons: Iterable[int]) -> float:
    """Use an explicit early arbitration marker to preserve all four Super Two tiers."""
    return 2.0 if any(2.0 <= service_time + season - service_season < 3.0 for season in arbitration_seasons) else ARBITRATION_SERVICE


def arbitration_year(service_time: float, first_service: float = ARBITRATION_SERVICE) -> int:
    """1 to 4 from service before the season and the schedule's arbitration start."""
    if service_time != service_time:
        return 1
    return int(max(1, min(4, np.floor(service_time - first_service) + 1)))
