"""What a win costs and what control years pay: fitted from the contracts table.

Every quantity has a documented fallback that the fit report names when used.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Fallbacks, each cited in the tip when used.
FALLBACK_PRICE_MUSD = 11.0
FALLBACK_PRICE_SOURCE = "fangraphs_2025_26"
FALLBACK_INFLATION = 0.03
INFLATION_CAP = 0.05
FALLBACK_ARB_FRACTIONS = {1: 0.25, 2: 0.40, 3: 0.60, 4: 0.70}
MIN_FREE_AGENT_CONTRACTS = 60
MIN_ARBITRATION_PLAYERS = 15
# CBA minimum salary by season, in millions; later seasons use the latest known.
LEAGUE_MINIMUM_MUSD: dict[int, float] = {2024: 0.74, 2025: 0.76, 2026: 0.78}
FREE_AGENT_OFFSEASONS = 3
ARBITRATION_SERVICE = 3.0


@dataclass(frozen=True)
class PriceModel:
    price_musd: float
    se: float
    inflation: float
    n: int
    source: str

    def price_for(self, season: int, first_season: int) -> float:
        return self.price_musd * (1.0 + self.inflation) ** max(0, season - first_season)


def league_minimum(season: int) -> float:
    known = max(s for s in LEAGUE_MINIMUM_MUSD if s <= season) if any(s <= season for s in LEAGUE_MINIMUM_MUSD) else min(LEAGUE_MINIMUM_MUSD)
    return LEAGUE_MINIMUM_MUSD[known]


def free_agent_signings(contracts: pd.DataFrame, current_season: int) -> pd.DataFrame:
    """One row per free-agent contract from the last offseasons: player_id, first_season, aav_musd, years.

    The contract summary on the page gives the length, total value, and first season, so a
    deal signed two winters ago is dated to its own first season, not the page's.
    """
    c = contracts[(contracts["how_acquired"] == "Free Agency") & contracts["contract_start"].notna() & contracts["contract_years"].notna()]
    c = c.drop_duplicates("player_id")
    c = c[c["contract_start"].between(current_season - FREE_AGENT_OFFSEASONS + 1, current_season)]
    c = c[c["contract_total_musd"].notna() & (c["contract_years"] > 0)]
    return pd.DataFrame({
        "player_id": c["player_id"].astype("int64").to_numpy(), "first_season": c["contract_start"].astype("int64").to_numpy(),
        "years": c["contract_years"].astype("int64").to_numpy(),
        "aav_musd": (c["contract_total_musd"].astype("float64") / c["contract_years"].astype("float64")).to_numpy(dtype="float64"),
    })


def fit_price_per_win(signings: pd.DataFrame, projected_war: pd.DataFrame) -> PriceModel:
    """Dollars per projected win from free-agent average annual values.

    ``projected_war`` has player_id, first_season, war (our projection for the first
    contract season, made with inputs through the prior season). The fit is through the
    origin on contracts with positive projected WAR; the fallback applies with fewer than
    60 contracts or when a free-intercept fit's intercept is negative beyond two standard errors.
    """
    data = signings.merge(projected_war, on=["player_id", "first_season"], how="inner")
    data = data[data["war"] > 0].astype({"war": "float64", "aav_musd": "float64"})
    n = len(data)
    if n < MIN_FREE_AGENT_CONTRACTS:
        log.info(f"price per win: {n} contracts, using the fallback")
        return PriceModel(FALLBACK_PRICE_MUSD, float("nan"), FALLBACK_INFLATION, n, FALLBACK_PRICE_SOURCE)
    x, y = data["war"].values, data["aav_musd"].values
    slope = float((x @ y) / (x @ x))
    resid = y - slope * x
    se = float(np.sqrt((resid @ resid) / (n - 1) / (x @ x)))
    design = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    free_resid = y - design @ coef
    cov = np.linalg.inv(design.T @ design) * (free_resid @ free_resid) / (n - 2)
    if coef[0] < -2 * np.sqrt(cov[0, 0]):
        log.info("price per win: negative intercept beyond two standard errors, using the fallback")
        return PriceModel(FALLBACK_PRICE_MUSD, float("nan"), FALLBACK_INFLATION, n, FALLBACK_PRICE_SOURCE)
    by_season = data.groupby("first_season").apply(lambda g: float((g["war"] @ g["aav_musd"]) / (g["war"] @ g["war"])), include_groups=False)
    inflation = FALLBACK_INFLATION
    if len(by_season) >= 2:
        growth = by_season.sort_index().pct_change().dropna()
        if not growth.empty:
            fitted = float(growth.mean())
            # A fit outside the cap is not evidence of a trend; the fallback stands.
            inflation = fitted if 0.0 <= fitted <= INFLATION_CAP else FALLBACK_INFLATION
    return PriceModel(slope, se, inflation, n, f"fit_{int(data['first_season'].min())}_{int(data['first_season'].max())}")


def arbitration_start(service_time: float, service_season: int, arbitration_seasons: Iterable[int]) -> float:
    """Use an explicit early arbitration marker to preserve all four Super Two tiers."""
    return 2.0 if any(2.0 <= service_time + season - service_season < 3.0 for season in arbitration_seasons) else ARBITRATION_SERVICE


def arbitration_year(service_time: float, first_service: float = ARBITRATION_SERVICE) -> int:
    """1 to 4 from service before the season and the schedule's arbitration start."""
    if service_time != service_time:
        return 1
    return int(max(1, min(4, np.floor(service_time - first_service) + 1)))


def fit_arbitration_fractions(contracts: pd.DataFrame, projected_war: pd.DataFrame, price: PriceModel, season: int) -> dict[int, float]:
    """Median share of market value paid in each arbitration year, from this season's arbitration salaries.

    ``projected_war`` has player_id and war for ``season``. Years with fewer than 15 players fall back.
    """
    arb = contracts[(contracts["season"] == season) & (contracts["salary_kind"] == "guaranteed") & contracts["service_time"].between(2.0, 6.0)
                    & (contracts["how_acquired"] != "Free Agency")]
    data = arb.merge(projected_war[["player_id", "war"]], on="player_id", how="inner")
    data = data[data["war"] > 0.5]
    starts = {
        pid: arbitration_start(float(g["service_time"].iloc[0]), int(g["season"].min()),
                               g.loc[g["salary_kind"].eq("arbitration") | g.get("declined_to", pd.Series(index=g.index, dtype=object)).eq("arbitration"), "season"])
        for pid, g in contracts.groupby("player_id")
    }
    data["arb_year"] = [arbitration_year(r.service_time, starts[r.player_id]) for r in data.itertuples()]
    data["fraction"] = data["salary_musd"] / (data["war"] * price.price_musd)
    out = dict(FALLBACK_ARB_FRACTIONS)
    for year, g in data.groupby("arb_year"):
        if len(g) >= MIN_ARBITRATION_PLAYERS:
            out[int(year)] = float(np.clip(g["fraction"].median(), 0.05, 1.0))
    return out
