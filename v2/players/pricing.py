"""What a win costs and what control years pay: fitted from the contracts table.

Every quantity has a documented fallback that the fit report names when used.
"""
from __future__ import annotations

import logging
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

    A contract is a run of guaranteed seasons for a player acquired by free agency whose
    first guaranteed season is within FREE_AGENT_OFFSEASONS of the current season.
    """
    c = contracts[(contracts["how_acquired"] == "Free Agency") & (contracts["salary_kind"].isin(["guaranteed", "buyout"]))]
    c = c.sort_values(["player_id", "season"])
    rows = []
    for player_id, g in c.groupby("player_id"):
        guaranteed = g[g["salary_kind"] == "guaranteed"]
        if guaranteed.empty:
            continue
        first = int(guaranteed["season"].min())
        if first < current_season - FREE_AGENT_OFFSEASONS + 1 or first > current_season:
            continue
        buyouts = g.loc[g["salary_kind"] == "buyout", "salary_musd"].fillna(0).sum()
        total = guaranteed["salary_musd"].fillna(0).sum() + buyouts
        rows.append({"player_id": int(player_id), "first_season": first, "years": len(guaranteed), "aav_musd": total / len(guaranteed)})
    return pd.DataFrame(rows, columns=["player_id", "first_season", "years", "aav_musd"])


def fit_price_per_win(signings: pd.DataFrame, projected_war: pd.DataFrame) -> PriceModel:
    """Dollars per projected win from free-agent average annual values.

    ``projected_war`` has player_id, first_season, war (our projection for the first
    contract season, made with inputs through the prior season). The fit is through the
    origin on contracts with positive projected WAR; the fallback applies with fewer than
    60 contracts or when a free-intercept fit's intercept is negative beyond two standard errors.
    """
    data = signings.merge(projected_war, on=["player_id", "first_season"], how="inner")
    data = data[data["war"] > 0]
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
            inflation = float(np.clip(growth.mean(), -INFLATION_CAP, INFLATION_CAP))
    return PriceModel(slope, se, inflation, n, f"fit_{int(data['first_season'].min())}_{int(data['first_season'].max())}")


def arbitration_year(service_time: float) -> int:
    """1 to 4 from service time before the season; a Super Two (under 3.000) counts as year 1."""
    if service_time != service_time:
        return 1
    return int(max(1, min(4, np.floor(service_time - ARBITRATION_SERVICE) + 1)))


def fit_arbitration_fractions(contracts: pd.DataFrame, projected_war: pd.DataFrame, price: PriceModel, season: int) -> dict[int, float]:
    """Median share of market value paid in each arbitration year, from this season's arbitration salaries.

    ``projected_war`` has player_id and war for ``season``. Years with fewer than 15 players fall back.
    """
    arb = contracts[(contracts["season"] == season) & (contracts["salary_kind"] == "guaranteed") & contracts["service_time"].between(2.0, 6.0)
                    & (contracts["how_acquired"] != "Free Agency")]
    data = arb.merge(projected_war[["player_id", "war"]], on="player_id", how="inner")
    data = data[data["war"] > 0.5]
    data["arb_year"] = data["service_time"].apply(arbitration_year)
    data["fraction"] = data["salary_musd"] / (data["war"] * price.price_musd)
    out = dict(FALLBACK_ARB_FRACTIONS)
    for year, g in data.groupby("arb_year"):
        if len(g) >= MIN_ARBITRATION_PLAYERS:
            out[int(year)] = float(np.clip(g["fraction"].median(), 0.05, 1.0))
    return out
