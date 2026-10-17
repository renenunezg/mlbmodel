"""Trade value: projected production over control years at the price of a win, minus salary.

Pre-arbitration seasons pay the league minimum, arbitration seasons a fitted
share of market value, guaranteed seasons their salary. A club option is
exercised when the projected value exceeds its price, otherwise the buyout is
paid for a season the player does not play; a player option is exercised when
the salary exceeds his market value. Control ends at free agency or a declined
option. Each season's surplus is discounted for distance.
"""
from __future__ import annotations

from dataclasses import dataclass

from v2.players.pricing import PriceModel, arbitration_year, league_minimum

DISCOUNT_PER_YEAR = 0.08
PLAYER_OPTIONS = {"player", "mutual"}


@dataclass(frozen=True)
class ContractYear:
    season: int
    salary_kind: str  # guaranteed, option, buyout, arbitration, pre_arb, free_agent
    salary_musd: float | None
    option_musd: float | None
    option_kind: str | None = "team"


@dataclass(frozen=True)
class Outlook:
    season: int
    age: int
    war: float
    salary_musd: float
    salary_kind: str
    surplus_musd: float
    in_control: bool
    value_musd: float = 0.0  # war × price for the season, undiscounted


def project_outlook(
    war_by_season: dict[int, float],
    ages: dict[int, int],
    contract: list[ContractYear],
    price: PriceModel,
    arb_fractions: dict[int, float],
    service_time: float,
    first_season: int,
    discount: float = DISCOUNT_PER_YEAR,
) -> list[Outlook]:
    """One Outlook per contract season from ``first_season`` on, until control ends."""
    out: list[Outlook] = []
    control = True
    by_season = {c.season: c for c in contract}
    for season in sorted(s for s in by_season if s >= first_season):
        c = by_season[season]
        war = float(war_by_season.get(season, 0.0))
        price_y = price.price_for(season, first_season)
        value = war * price_y
        years_ahead = season - first_season
        service_before = (service_time if service_time == service_time else 0.0) + years_ahead
        if not control or c.salary_kind == "free_agent":
            out.append(Outlook(season, ages.get(season, 0), war, 0.0, "free_agent", 0.0, False, value))
            control = False
            continue
        kind = c.salary_kind
        if kind == "pre_arb":
            salary = league_minimum(season)
        elif kind == "arbitration":
            salary = arb_fractions.get(arbitration_year(service_before), arb_fractions[1]) * value
        elif kind == "buyout":
            option_price = float(c.option_musd or 0.0)
            buyout = float(c.salary_musd or 0.0)
            if (c.option_kind or "team") in PLAYER_OPTIONS:
                exercised = option_price > value
            else:
                exercised = value > option_price
            if exercised:
                salary, kind = option_price, "option"
            else:
                # The buyout is paid for a season he does not play for the team.
                salary, kind, war, value, control = buyout, "buyout", 0.0, 0.0, False
        else:
            salary = float(c.salary_musd or 0.0)
            kind = "guaranteed"
        surplus = (value - salary) * (1.0 - discount) ** years_ahead
        out.append(Outlook(season, ages.get(season, 0), war, salary, kind, surplus, control, value))
    return out


def trade_value(outlook: list[Outlook]) -> float:
    """Discounted surplus over control years, including a declined option's buyout."""
    return float(sum(o.surplus_musd for o in outlook if o.in_control or o.salary_kind == "buyout"))


def market_value(outlook: list[Outlook]) -> float:
    """What the first projected season is worth on the open market, undiscounted."""
    return float(outlook[0].value_musd) if outlook else 0.0
