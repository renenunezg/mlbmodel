"""Trade value arithmetic and the scoring guards that keep wrong dollar figures off the site."""
import pytest

from v2.players.pricing import PriceModel
from v2.players.value import ContractYear, market_value, project_outlook, trade_value

PRICE = PriceModel(price_musd=10.0, se=0.5, inflation=0.0, n=100, source="test")
ARB = {1: 0.25, 2: 0.40, 3: 0.60, 4: 0.70}


def test_outlook_prices_arbitration_and_option():
    contract = [
        ContractYear(2027, "arbitration", None, None),
        ContractYear(2028, "buyout", 2.0, 15.0),
        ContractYear(2029, "free_agent", None, None),
    ]
    out = project_outlook({2027: 3.0, 2028: 2.0, 2029: 1.0}, {2027: 29, 2028: 30, 2029: 31},
                          contract, PRICE, ARB, service_time=3.1, first_season=2027)
    assert out[0].salary_musd == 7.5 and out[0].salary_kind == "arbitration"  # 0.25 × 3.0 × 10
    assert out[1].salary_musd == 15.0 and out[1].in_control                   # 20 > 15: option exercised
    assert out[2].in_control is False
    assert trade_value(out) == pytest.approx((30 - 7.5) + (20 - 15) * 0.92)


def test_declined_option_ends_control():
    contract = [ContractYear(2027, "buyout", 2.0, 15.0), ContractYear(2028, "guaranteed", 20.0, None)]
    out = project_outlook({2027: 1.0, 2028: 1.0}, {2027: 33, 2028: 34}, contract, PRICE, ARB, 10.0, 2027)
    assert out[0].salary_musd == 2.0 and out[0].in_control is False and out[1].in_control is False
    assert out[0].war == 0.0 and trade_value(out) == pytest.approx(-2.0)


def test_free_agent_has_market_value_only():
    out = project_outlook({2027: 4.0}, {2027: 30}, [ContractYear(2027, "free_agent", None, None)], PRICE, ARB, 7.0, 2027)
    assert trade_value(out) == 0.0 and market_value(out) == 40.0
