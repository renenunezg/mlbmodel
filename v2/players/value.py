"""Trade value: projected production over control years at the price of a win, minus salary.

Pre-arbitration seasons pay the league minimum, arbitration seasons a fitted
share of market value, guaranteed seasons their salary. A club option is
exercised when keeping the player is worth more than declining it, including
the avoided buyout and any retained arbitration rights. Otherwise the buyout is
paid for a season the player does not play; a player option is exercised when
the salary exceeds his market value. Control ends at free agency, including an
option declined into free agency or a non-tender. Tender and club-option decisions
compare the entire remaining surplus, not just the current season.
Each season's surplus is discounted for distance.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cache
from math import isclose, isfinite

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
    tendered: bool = False  # dated replay after the non-tender deadline
    cashflows: tuple[tuple[int, float], ...] | None = None  # payment year and cash earned by this service year
    retention_raise_musd: float = 0.0  # annual club counteroffer for a multi-year player opt-out

    def __post_init__(self) -> None:
        if self.cashflows is not None and (self.salary_kind != "guaranteed" or not self.cashflows
                or self.salary_musd is None or not isfinite(self.salary_musd)
                or not isclose(sum(amount for _, amount in self.cashflows), self.salary_musd)
                or any(paid < self.season or amount < 0 or not isfinite(amount) for paid, amount in self.cashflows)):
            raise ValueError("Invalid deferred contract cash or earning-year allocation")


@dataclass(frozen=True)
class Outlook:
    season: int
    age: int
    war: float
    salary_musd: float
    salary_kind: str
    surplus_musd: float
    in_control: bool
    value_musd: float = 0.0  # Undiscounted market salary, including the replacement offset.


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
    allow_non_tender: bool = True,
    war_by_role: dict[int, dict[str, float]] | None = None,
    market_profiles: dict[int, dict[str, float]] | None = None,
    arbitration_model: dict | None = None,
) -> list[Outlook]:
    """One Outlook per contract season from ``first_season`` on, until control ends.

    ``service_time`` is measured before ``service_season`` (the page's season); it
    advances one year per season after that.
    Unsigned seasons are tendered only when their remaining discounted surplus
    exceeds releasing the player. Guaranteed salaries and buyouts remain owed.
    Disable non-tender decisions when production cannot be estimated.
    """
    by_season = {c.season: c for c in contract}
    seasons = sorted(s for s in by_season if s >= first_season)
    base_service = service_time if service_time == service_time else 0.0
    service_season = first_season if service_season is None else service_season
    first_arb_service = arbitration_start(base_service, service_season,
                                        (c.season for c in contract if c.salary_kind == "arbitration" or c.declined_to == "arbitration"))

    def market_value_for(year: int) -> float:
        if war_by_role is not None:
            return price.value_for_roles(war_by_role[year], year, first_season,
                                         market_profiles[year] if market_profiles is not None else None)
        return price.value_for(float(war_by_season.get(year, 0.0)), year, first_season)

    def released(index: int) -> tuple[Outlook, ...]:
        return tuple(Outlook(y, ages.get(y, 0), float(war_by_season.get(y, 0.0)),
                             0.0, "free_agent", 0.0, False,
                             market_value_for(y))
                     for y in seasons[index:])

    @cache
    def keep(index: int, prior_salary: float | None) -> tuple[Outlook, ...]:
        if index == len(seasons):
            return ()
        season = seasons[index]
        c = by_season[season]
        war = float(war_by_season.get(season, 0.0))
        market = market_value_for(season)
        factor = (1.0 - discount) ** (season - first_season)
        if c.salary_kind == "free_agent":
            return released(index)

        def continuing(salary: float, kind: str, buyout: float = 0.0) -> tuple[Outlook, ...]:
            cash_cost = ((salary + buyout) * factor if c.cashflows is None else
                         sum(amount * (1-discount)**(paid-first_season) for paid, amount in c.cashflows) + buyout*factor)
            row = Outlook(season, ages.get(season, 0), war, salary + buyout, kind,
                          market * factor - cash_cost, True, market)
            return (row,) + keep(index + 1, salary)

        def exit_with_buyout(buyout: float) -> tuple[Outlook, ...]:
            row = Outlook(season, ages.get(season, 0), 0.0, buyout, "buyout", -buyout * factor, False, 0.0)
            return (row,) + released(index + 1)

        def tender(salary: float, kind: str, buyout: float = 0.0) -> tuple[Outlook, ...]:
            continued = continuing(salary, kind, buyout)
            # A non-tender forfeits ALL subsequent control, not just a bad season.
            # Signed guarantees and buyouts are never cancellable this way.
            exited = exit_with_buyout(buyout) if buyout else released(index)
            return max((continued, exited), key=trade_value) if allow_non_tender and not c.tendered else continued

        service_before = base_service + season - service_season
        fraction = arb_fractions.get(arbitration_year(service_before, first_arb_service), arb_fractions[1])
        salary_estimate = fraction * market
        if arbitration_model and (c.salary_kind == "arbitration" or c.declined_to == "arbitration"):
            from v2.players.arbitration import estimate

            if war_by_role is None or market_profiles is None:
                raise ValueError("Arbitration pricing requires dated WAR and workload profiles")
            salary_estimate = estimate(arbitration_model, war_by_role[season], market_profiles[season],
                                       service_before, season)
        arb_salary = max(salary_estimate, league_minimum(season),
                         ARBITRATION_PRIOR_SALARY_FLOOR * (prior_salary or 0.0))
        if c.salary_kind == "pre_arb":
            return tender(league_minimum(season), "pre_arb")
        if c.salary_kind == "arbitration":
            # A dated arbitration estimate is not a guarantee: retain the tender
            # decision, but do not replace the quoted salary with a WAR fraction.
            salary = max(float(c.salary_musd), league_minimum(season)) if c.salary_musd is not None else arb_salary
            return tender(salary, "arbitration")
        if c.salary_kind == "buyout":
            option_price = float(c.option_musd or 0.0)
            buyout = float(c.salary_musd or 0.0)
            exercised = continuing(option_price, "option")
            declined = (tender(arb_salary, "arbitration", buyout) if c.declined_to == "arbitration"
                        else exit_with_buyout(buyout))
            if c.option_kind in MUTUAL_OPTIONS:
                return declined
            if c.option_kind == "player_block":
                # The player compares the entire remaining guarantee with the
                # projected alternative, not only the first option year's salary.
                if trade_value(exercised) <= trade_value(declined):
                    return exercised
                if c.retention_raise_musd <= 0:
                    return declined
                counter = tuple(replace(o, salary_musd=o.salary_musd+c.retention_raise_musd,
                    surplus_musd=o.surplus_musd-c.retention_raise_musd*(1-discount)**(o.season-first_season))
                    if o.in_control else o for o in exercised)
                return max((counter, declined), key=trade_value)
            if c.option_kind in PLAYER_OPTIONS:
                return exercised if option_price > market else declined
            return max((exercised, declined), key=trade_value)
        return continuing(float(c.salary_musd or 0.0), "guaranteed")

    previous = by_season.get(first_season - 1)
    prior_salary = previous.salary_musd if previous is not None else None
    return list(keep(0, prior_salary))


def trade_value(outlook: list[Outlook]) -> float:
    """Discounted surplus over control years, including a declined option's buyout."""
    return float(sum(o.surplus_musd for o in outlook if o.in_control or o.salary_kind == "buyout"))


def market_value(outlook: list[Outlook]) -> float:
    """What the first projected season is worth on the open market, undiscounted."""
    return float(outlook[0].value_musd) if outlook else 0.0
