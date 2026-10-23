"""Prospect trade value: expected control-years surplus at the player model's price of a win.

The six control seasons start at the expected arrival. Each season's expected WAR is a sixth of
the expected control WAR (which already carries the probability of never arriving); pre-arbitration
seasons pay the league minimum (weighted by the probability of being in the majors), arbitration
seasons the player model's fitted share of market value. Surplus is discounted at the player
model's rate per year from today, so a distant arrival is worth less. The band repeats the sum
on the control-WAR quartiles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from v2.players.pricing import PriceModel, league_minimum
from v2.players.value import DISCOUNT_PER_YEAR

CONTROL_SEASONS = 6
PRE_ARB_SEASONS = 3


def market_value(war: float, p_mlb: float, year: int, season: int, role: str, price: PriceModel) -> float:
    """Market salary for one season's expected WAR: the role's price per win, plus the minimum when the player is there."""
    return war * price.price_for(year, season, role) + (p_mlb * league_minimum(year) if price.above_minimum else 0.0)


def _surplus(control_war: float, p_mlb: float, eta_years: float, role: str, price: PriceModel, arb: dict[int, float],
             season: int) -> tuple[float, float]:
    """Discounted surplus and undiscounted market value over the control seasons for one player."""
    per_season = max(control_war, 0.0) / CONTROL_SEASONS
    first = season + int(round(max(eta_years, 0.5)))
    surplus = 0.0
    market = 0.0
    for k in range(CONTROL_SEASONS):
        year = first + k
        value = market_value(per_season, p_mlb, year, season, role, price)
        if k < PRE_ARB_SEASONS:
            salary = p_mlb * league_minimum(year)
        else:
            salary = arb.get(k - PRE_ARB_SEASONS + 1, 0.0) * value
        discount = (1.0 - DISCOUNT_PER_YEAR) ** (year - season)
        surplus += (value - salary) * discount
        market += value
    return surplus, market


def price_role(role: str, start_share: float) -> str:
    """The player model's price role: hitters, starters (half their games or more) and relievers."""
    if role == "bat":
        return "bat"
    return "SP" if start_share == start_share and start_share >= 0.5 else "RP"


def trade_values(pred: pd.DataFrame, roles: pd.Series, price: PriceModel, arb: dict[int, float], season: int) -> pd.DataFrame:
    """Trade value, band and market value per row of ``pred`` (needs control_war, control_q25, control_q75, p_mlb, eta_years)."""
    out = pd.DataFrame(index=pred.index)
    vals, lows, highs, markets = [], [], [], []
    for row, role in zip(pred.itertuples(index=False), roles):
        v, m = _surplus(row.control_war, row.p_mlb, row.eta_years, role, price, arb, season)
        lo, _ = _surplus(row.control_q25, row.p_mlb, row.eta_years, role, price, arb, season)
        hi, _ = _surplus(row.control_q75, row.p_mlb, row.eta_years, role, price, arb, season)
        vals.append(v)
        lows.append(min(lo, v))
        highs.append(max(hi, v))
        markets.append(m)
    out["trade_value_musd"] = np.round(vals, 2)
    out["trade_value_low"] = np.round(lows, 2)
    out["trade_value_high"] = np.round(highs, 2)
    out["market_value_musd"] = np.round(markets, 2)
    out["eta_season"] = season + np.round(np.maximum(pred["eta_years"].to_numpy(float), 0.5)).astype(int)
    return out


def outlook(control_war: float, p_mlb: float, eta_years: float, age: float, role: str, price: PriceModel, arb: dict[int, float],
            season: int) -> list[dict]:
    """Season-by-season expected line for the detail table."""
    per_season = max(control_war, 0.0) / CONTROL_SEASONS
    first = season + int(round(max(eta_years, 0.5)))
    rows = []
    for k in range(CONTROL_SEASONS):
        year = first + k
        value = market_value(per_season, p_mlb, year, season, role, price)
        kind = "pre_arb" if k < PRE_ARB_SEASONS else "arbitration"
        salary = p_mlb * league_minimum(year) if kind == "pre_arb" else arb.get(k - PRE_ARB_SEASONS + 1, 0.0) * value
        rows.append({"season": year, "age": int(round(age + (year - season))) if age == age else None, "war": round(per_season, 2),
                     "salary_musd": round(salary, 2), "salary_kind": kind, "surplus_musd": round(value - salary, 2),
                     "in_control": True, "p_active": round(p_mlb, 3)})
    return rows
