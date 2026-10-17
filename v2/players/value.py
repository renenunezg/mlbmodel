"""Trade value: projected production over control years at the price of a win, minus salary.

Pre-arbitration seasons pay the league minimum, arbitration seasons a fitted
share of market value, guaranteed seasons their salary. A club option is
exercised when keeping the player is worth more than declining it, including
the avoided buyout and any retained arbitration rights. Otherwise the buyout is
paid for a season the player does not play; a player option is exercised when
the salary exceeds his market value. Control ends at free agency, including an
option declined into free agency. Each season's surplus is discounted for distance.
"""
from __future__ import annotations

from dataclasses import dataclass

from v2.players.pricing import PriceModel, arbitration_start, arbitration_year, league_minimum

DISCOUNT_PER_YEAR = 0.08
PLAYER_OPTIONS = {"player"}
# A mutual option is almost never exercised by both sides; it prices as its buyout.
MUTUAL_OPTIONS = {"mutual"}
# An arbitration estimate never falls below the league minimum or this share of the prior salary.
ARBITRATION_PRIOR_SALARY_FLOOR = 0.8


@dataclass(frozen=True)
class ContractYear:
    season: int
    salary_kind: str  # guaranteed, option, buyout, arbitration, pre_arb, free_agent
    salary_musd: float | None
    option_musd: float | None
    option_kind: str | None = "team"
    declined_to: str | None = None  # free_agent or arbitration when an option is declined


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
    service_season: int | None = None,
) -> list[Outlook]:
    """One Outlook per contract season from ``first_season`` on, until control ends.

    ``service_time`` is measured before ``service_season`` (the page's season); it
    advances one year per season after that.
    """
    out: list[Outlook] = []
    control = True
    by_season = {c.season: c for c in contract}
    base_service = service_time if service_time == service_time else 0.0
    service_season = first_season if service_season is None else service_season
    first_arb_service = arbitration_start(base_service, service_season,
                                        (c.season for c in contract if c.salary_kind == "arbitration" or c.declined_to == "arbitration"))
    prior_salary = None
    for season in sorted(s for s in by_season if s >= first_season):
        c = by_season[season]
        war = float(war_by_season.get(season, 0.0))
        price_y = price.price_for(season, first_season)
        value = war * price_y
        years_ahead = season - first_season
        service_before = base_service + (season - service_season)
        if prior_salary is None:
            earlier = [by_season[s] for s in by_season if s == season - 1 and by_season[s].salary_musd is not None]
            prior_salary = float(earlier[0].salary_musd) if earlier else None
        if not control or c.salary_kind == "free_agent":
            out.append(Outlook(season, ages.get(season, 0), war, 0.0, "free_agent", 0.0, False, value))
            control = False
            continue
        kind = c.salary_kind
        buyout_paid = 0.0
        if kind == "arbitration" or (kind == "buyout" and c.declined_to == "arbitration"):
            estimate = arb_fractions.get(arbitration_year(service_before, first_arb_service), arb_fractions[1]) * value
            floor = max(league_minimum(season), ARBITRATION_PRIOR_SALARY_FLOOR * prior_salary if prior_salary else 0.0)
            arbitration_salary = max(estimate, floor)
        if kind == "pre_arb":
            salary = league_minimum(season)
        elif kind == "arbitration":
            salary = arbitration_salary
        elif kind == "buyout":
            option_price = float(c.option_musd or 0.0)
            buyout = float(c.salary_musd or 0.0)
            option_kind = c.option_kind or "team"
            if option_kind in MUTUAL_OPTIONS:
                exercised = False
            elif option_kind in PLAYER_OPTIONS:
                exercised = option_price > value
            elif c.declined_to == "arbitration":
                exercised = option_price <= arbitration_salary + buyout
            else:
                exercised = value - option_price >= -buyout
            if exercised:
                salary, kind = option_price, "option"
            elif c.declined_to == "arbitration":
                # Declined into arbitration: the team keeps him at an arbitration salary.
                salary, kind = arbitration_salary + buyout, "arbitration"
                buyout_paid = buyout
            else:
                # The buyout is paid for a season he does not play for the team.
                salary, kind, war, value, control = buyout, "buyout", 0.0, 0.0, False
        else:
            salary = float(c.salary_musd or 0.0)
            kind = "guaranteed"
        surplus = (value - salary) * (1.0 - discount) ** years_ahead
        out.append(Outlook(season, ages.get(season, 0), war, salary, kind, surplus, control, value))
        prior_salary = salary - buyout_paid
    return out


def trade_value(outlook: list[Outlook]) -> float:
    """Discounted surplus over control years, including a declined option's buyout."""
    return float(sum(o.surplus_musd for o in outlook if o.in_control or o.salary_kind == "buyout"))


def market_value(outlook: list[Outlook]) -> float:
    """What the first projected season is worth on the open market, undiscounted."""
    return float(outlook[0].value_musd) if outlook else 0.0
