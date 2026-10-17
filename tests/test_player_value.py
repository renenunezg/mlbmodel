"""Trade value arithmetic and the scoring guards that keep wrong dollar figures off the site."""
import pandas as pd
import pytest

from v2.players import bio, score
from v2.players import inputs as inputs_mod
from v2.players.contracts import parse_team_page
from v2.players.model import ValueModel
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
    # A buyout makes exercise preferable even when the option has negative surplus.
    out = project_outlook({2027: 1.4}, {2027: 29}, [ContractYear(2027, "buyout", 2.0, 15.0)], PRICE, ARB, 7.0, 2027)
    assert out[0].salary_kind == "option" and trade_value(out) == pytest.approx(-1.0)


def test_declined_option_ends_control():
    contract = [ContractYear(2027, "buyout", 2.0, 15.0), ContractYear(2028, "guaranteed", 20.0, None)]
    out = project_outlook({2027: 1.0, 2028: 1.0}, {2027: 33, 2028: 34}, contract, PRICE, ARB, 10.0, 2027)
    assert out[0].salary_musd == 2.0 and out[0].in_control is False and out[1].in_control is False
    assert out[0].war == 0.0 and trade_value(out) == pytest.approx(-2.0)


def test_free_agent_has_market_value_only():
    out = project_outlook({2027: 4.0}, {2027: 30}, [ContractYear(2027, "free_agent", None, None)], PRICE, ARB, 7.0, 2027)
    assert trade_value(out) == 0.0 and market_value(out) == 40.0


# Scoring guards.
EMPTY = ValueModel.empty(dev_seasons=[2021, 2022, 2023, 2024], holdout_pair=(2025, 2026))
NO_CONTRACTS = pd.DataFrame(columns=["player_id", "season", "salary_kind", "salary_musd", "option_musd", "option_kind", "service_time", "fa_year", "how_acquired"])


def test_statcast_third_base_and_shortstop_reach_scoring_correctly(monkeypatch):
    # Statcast fielding slots: 5 is third base, 6 is shortstop.
    pitches = pd.DataFrame([
        {"game_pk": 1, "inning": inning, "inning_topbot": "Top", "home_team": "SD", "away_team": "KC",
         "batter": 99, "pitcher": 100, "events": "field_out",
         **{f"fielder_{slot}": None for slot in range(2, 10)},
         "fielder_5": 592518 if inning < 3 else 677951,
         "fielder_6": 677951 if inning < 3 else 592518}
        for inning in (1, 2, 3)
    ])
    monkeypatch.setattr(bio, "_load_pitches", lambda season, columns: pitches)
    groups = bio.primary_position(bio.positions([2026]), {2026: 1.0}).set_index("player_id")
    assert groups.loc[592518, "position_group"] == "3B"
    assert groups.loc[677951, "position_group"] == "SS"
    assert groups.loc[592518, "secondary_positions"] == ["SS"]
    assert groups.loc[677951, "secondary_positions"] == ["3B"]
    data = one_hitter("forty_man", pa=600)
    data["groups"] = groups.loc[[592518]].reset_index().assign(player_id=1)
    values, _ = score.assemble_values(**data, contracts=NO_CONTRACTS, model=EMPTY, next_season=2027)
    assert values.iloc[0].position_group == "3B"
    assert values.iloc[0].secondary_positions == ["SS"]
    assert values.iloc[0].tip["positional_adjustment"] == EMPTY.positional_adjustment["3B"]


def one_hitter(status: str, pa: float) -> dict:
    population = pd.DataFrame([{
        "player_id": 1, "player_name": "Test Hitter", "team": "NYY" if status != "free_agent" else "", "team_id": 147,
        "status": status, "age": 28, "bats": "R", "throws": "R", "birth_date": "1998-06-30", "listed_position": "2B",
    }])
    groups = pd.DataFrame([{"player_id": 1, "position_group": "2B", "secondary_positions": []}])
    rates = pd.DataFrame([{"player_id": 1, "role": "bat", "group": "2B", "n": pa,
                           "K": 0.22, "BB": 0.08, "HBP": 0.01, "1B": 0.15, "2B": 0.045, "3B": 0.004, "HR": 0.03, "OUT": 0.461}])
    inputs = pd.DataFrame([{"player_id": 1, "season": 2026, "role": "bat", "input": "contact.k_rate", "value": 0.22, "sample": pa}])
    return {"population": population, "groups": groups, "rates": rates, "inputs": inputs, "il_days": pd.DataFrame(columns=["player_id", "season", "il_days"]),
            "playing_time": pd.DataFrame([{"player_id": 1, "role": "bat", "pa": pa}])}


def test_model_refuses_its_own_development_season():
    with pytest.raises(ValueError, match="trained on"):
        EMPTY.validate(2024)
    EMPTY.validate(2027)


def test_insufficient_sample_has_no_grade():
    values, _ = score.assemble_values(**one_hitter("forty_man", pa=20), contracts=NO_CONTRACTS, model=EMPTY, next_season=2027)
    assert values.grade.isna().all() and (values.grade_reason == "insufficient_sample").all()
    assert values.trade_value_musd.isna().all()


def test_pending_free_agent_rows():
    values, _ = score.assemble_values(**one_hitter("free_agent", pa=600), contracts=NO_CONTRACTS, model=EMPTY, next_season=2027)
    assert (values.control_years == 0).all() and (values.trade_value_musd == 0).all() and values.market_value_musd.notna().all()


# Fix pass after review.
def test_window_counts_stay_in_plate_appearance_units():
    counts = {2026: pd.DataFrame([{"player_id": 1, "role": "bat", "K": 5, "BB": 2, "HBP": 0, "1B": 4, "2B": 1, "3B": 0, "HR": 1, "OUT": 7, "season": 2026}])}
    wc = inputs_mod.window_counts(counts, {2026: 5.0, 2025: 4.0, 2024: 3.0})
    assert wc[["K", "BB", "HBP", "1B", "2B", "3B", "HR", "OUT"]].sum(axis=1).iloc[0] == pytest.approx(20.0)


def test_arbitration_year_counts_seasons_since_the_page():
    # Service 3.000 before 2026; the 2027 season is his second arbitration year.
    contract = [ContractYear(2026, "guaranteed", 3.48, None), ContractYear(2027, "arbitration", None, None), ContractYear(2028, "free_agent", None, None)]
    out = project_outlook({2027: 3.0}, {2027: 26}, contract, PRICE, ARB, service_time=3.0, first_season=2027, service_season=2026)
    assert out[0].salary_kind == "arbitration" and out[0].salary_musd == pytest.approx(0.40 * 30.0)
    contract = [ContractYear(s, "arbitration", None, None) for s in range(2026, 2030)]
    out = project_outlook(dict.fromkeys(range(2026, 2030), 3.0), {}, contract, PRICE, ARB, 2.144, 2026)
    assert [o.salary_musd for o in out] == pytest.approx([7.5, 12.0, 18.0, 21.0])
    # Starting a projection after the first arb year must not reset its tier.
    out = project_outlook({2027: 3.0}, {}, contract, PRICE, ARB, 2.144, 2027, service_season=2026)
    assert out[0].salary_musd == pytest.approx(12.0)


def test_arbitration_estimate_never_falls_below_prior_salary_floor():
    contract = [ContractYear(2026, "guaranteed", 3.48, None), ContractYear(2027, "arbitration", None, None), ContractYear(2028, "free_agent", None, None)]
    out = project_outlook({2027: 0.2}, {2027: 26}, contract, PRICE, ARB, service_time=3.0, first_season=2027, service_season=2026)
    assert out[0].salary_musd == pytest.approx(0.8 * 3.48)


def test_mutual_option_defaults_to_the_buyout():
    contract = [ContractYear(2027, "buyout", 2.0, 15.0, "mutual")]
    out = project_outlook({2027: 3.0}, {2027: 30}, contract, PRICE, ARB, 8.0, 2027)
    assert out[0].salary_kind == "buyout" and out[0].in_control is False


def test_declined_option_to_arbitration_keeps_control():
    contract = [ContractYear(2027, "buyout", 0.5, 15.0, "team", "arbitration"), ContractYear(2028, "arbitration", None, None), ContractYear(2029, "free_agent", None, None)]
    out = project_outlook({2027: 0.5, 2028: 0.1}, {2027: 27, 2028: 28}, contract, PRICE, ARB, 4.0, 2027, service_season=2026)
    assert out[0].salary_kind == "arbitration" and out[0].in_control and out[1].in_control
    assert out[0].salary_musd == pytest.approx(3.5)  # third-year arb plus the buyout
    assert out[1].salary_musd == pytest.approx(2.4)  # prior salary floor excludes the one-time buyout
    rows = pd.DataFrame([
        {"player_id": 1, "season": c.season, "salary_kind": c.salary_kind, "salary_musd": c.salary_musd,
         "option_musd": c.option_musd, "option_kind": c.option_kind, "declined_to": c.declined_to,
         "service_time": 5.0, "fa_year": 2029, "how_acquired": "Traded"}
        for c in contract
    ])
    scored, _ = score.assemble_values(**one_hitter("forty_man", 600), contracts=rows, model=EMPTY, next_season=2027)
    assert scored.iloc[0].control_years == 2
    assert scored.iloc[0].salary_current_musd == scored.iloc[0].outlook[0]["salary_musd"]
    assert scored.iloc[0].outlook[0]["salary_kind"] == "arbitration"
    # Retained arbitration can be cheaper than exercising a positive-surplus option.
    out = project_outlook({2027: 3.0}, {}, [ContractYear(2027, "buyout", 1.0, 15.0, "team", "arbitration")], PRICE, ARB, 3.0, 2027)
    assert out[0].salary_kind == "arbitration" and out[0].salary_musd == pytest.approx(8.5)


def test_long_contract_reaches_final_scored_season():
    cells = "".join(f'<td data-stat="contract_y{i}">$10M</td>' for i in range(14))
    html = (
        '<table id="payroll"><thead><tr><th data-stat="contract_y0" aria-label="2026"></th></tr></thead><tbody><tr>'
        '<th data-stat="player"><a href="/players/x/testpl01.shtml">Test Hitter</a></th>'
        '<td data-stat="service_time">7.134</td><td data-stat="how_acquired">Free Agency</td>'
        '<td data-stat="contract_summary">14 yrs/$140M (26-39)</td>' + cells + '</tr></tbody></table>'
    )
    contract = parse_team_page(html).assign(player_id=1)
    draws = pd.DataFrame([[-1.0, 0.0, 1.0]], index=[1])
    scored, _ = score.assemble_values(**one_hitter("forty_man", 600), contracts=contract, model=EMPTY, next_season=2027, draws=draws)
    row = scored.iloc[0]
    assert row.fa_year == 2040 and row.control_years == 13 and row.owed_through_control_musd == 130.0
    final = next(o for o in row.outlook if o["season"] == 2039)
    assert final["war"] == row.projected_war and final["war"] > 0
    assert row.trade_value_low < row.trade_value_musd < row.trade_value_high


def test_scored_season_is_the_latest_that_has_started():
    from datetime import date as d

    assert score.season_for(d(2027, 1, 15), {2027: d(2027, 3, 26), 2026: d(2026, 3, 25)}) == 2026
    assert score.season_for(d(2027, 4, 1), {2027: d(2027, 3, 26), 2026: d(2026, 3, 25)}) == 2027
