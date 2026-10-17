"""Trade value arithmetic and the scoring guards that keep wrong dollar figures off the site."""
import pandas as pd
import pytest

from v2.players import aging, bio, production, score, workload
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
    # Deferred salary is discounted on payment, while nominal debt stays visible.
    deferred = [ContractYear(2027, "guaranteed", 70, None, cashflows=((2027, 2), (2037, 68)))]
    out = project_outlook({2027: 5}, {}, deferred, PRICE, ARB, 8, 2027)
    assert trade_value(out) == pytest.approx(50-2-68*.92**10)
    assert out[0].salary_musd == 70 and out[0].in_control
    # A block opt-out compares all remaining years. The club's counteroffer
    # raises every retained year's salary, not only the opt-out season.
    block = [ContractYear(2027, "buyout", 0, 46, option_kind="player_block", retention_raise_musd=4),
             ContractYear(2028, "guaranteed", 46, None)]
    out = project_outlook({2027: 6, 2028: 4}, {}, block, PRICE, ARB, 8, 2027)
    assert [o.salary_musd for o in out] == [50, 50]
    assert trade_value(out) == pytest.approx(10-10*.92)
    out = project_outlook({2027: 4.9, 2028: 4.9}, {}, block, PRICE, ARB, 8, 2027)
    assert not any(o.in_control for o in out)  # Player leaves; counteroffer costs more than retaining is worth.
    out = project_outlook({2027: 6, 2028: 0}, {}, block, PRICE, ARB, 8, 2027)
    assert [o.salary_musd for o in out] == [46, 46]  # Valuable first year cannot erase the remaining guarantee.
    from v2.players import cash

    source = pd.DataFrame([{"player_id": 665742, "contract_start": 2025, "contract_years": 15,
                            "contract_total_musd": 765}])
    from datetime import date

    payroll = [ContractYear(2027, "guaranteed", 47.5, None)]
    corrected, info = cash.apply(source, payroll, date(2026, 10, 17))
    assert corrected[0].salary_musd == 42.5 and info["verified"]  # Bonus already paid in 2025.
    unchanged, info = cash.apply(source, payroll, date(2024, 1, 1))
    assert unchanged == payroll and not info["verified"]


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
    from v2.players.fit import fit

    with pytest.raises(ValueError, match="overlap"):
        fit(last_dev_season=2025, holdout_pair=(2025, 2026))


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
    assert out[0].salary_kind == "free_agent" and trade_value(out) == 0
    # A bad arb year can still buy more valuable later control; the salary floor remains.
    contract.insert(2, ContractYear(2028, "arbitration", None, None))
    contract[-1] = ContractYear(2029, "free_agent", None, None)
    out = project_outlook({2027: 0.2, 2028: 3.0}, {}, contract, PRICE, ARB, 3.0, 2027, service_season=2026)
    assert out[0].in_control and out[0].salary_musd == pytest.approx(0.8 * 3.48)
    # The same weak production under a signed contract still costs money.
    signed = [ContractYear(2027, "guaranteed", 3.48, None)]
    assert trade_value(project_outlook({2027: 0.2}, {}, signed, PRICE, ARB, 3.0, 2027)) == pytest.approx(-1.48)
    # A historical trade after the deadline cannot retroactively non-tender year one.
    tendered = [ContractYear(2027, "arbitration", 12.0, None, tendered=True)]
    out = project_outlook({2027: 0.2}, {}, tendered, PRICE, ARB, 5.0, 2027)
    assert out[0].in_control and out[0].salary_musd == 12.0 and trade_value(out) == -10.0



def test_mutual_option_defaults_to_the_buyout():
    contract = [ContractYear(2027, "buyout", 2.0, 15.0, "mutual")]
    out = project_outlook({2027: 3.0}, {2027: 30}, contract, PRICE, ARB, 8.0, 2027)
    assert out[0].salary_kind == "buyout" and out[0].in_control is False


def test_declined_option_to_arbitration_keeps_control():
    contract = [ContractYear(2027, "buyout", 0.5, 15.0, "team", "arbitration"), ContractYear(2028, "arbitration", None, None), ContractYear(2029, "free_agent", None, None)]
    out = project_outlook({2027: 0.5, 2028: 0.1}, {2027: 27, 2028: 28}, contract, PRICE, ARB, 4.0, 2027, service_season=2026)
    assert out[0].salary_kind == "arbitration" and out[0].in_control and not out[1].in_control
    assert out[0].salary_musd == pytest.approx(3.5)  # third-year arb plus the buyout
    assert out[1].salary_kind == "free_agent"  # non-tender the loss-making final arb year
    out = project_outlook({2027: 0.5, 2028: 0.25}, {}, contract, PRICE, ARB, 4.0, 2027, service_season=2026)
    assert out[1].salary_musd == pytest.approx(2.4)  # floor excludes the one-time buyout
    out = project_outlook({2027: 0.0, 2028: 0.0}, {}, contract, PRICE, ARB, 4.0, 2027, service_season=2026)
    assert trade_value(out) == -0.5 and out[0].salary_kind == "buyout"
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


def test_workload_replay_uses_completed_history_and_separate_roles(monkeypatch):
    rows = [{"player_id": 1, "role": role, "season": year, "age": year-1999, "volume": n, "quality": 0.0}
            for year, bat, pit in [(2022, 600, 700), (2023, 0, 0), (2024, 600, 700), (2025, 0, 300), (2026, 400, 0), (2027, 9999, 9999)]
            for role, n in [("bat", bat), ("pit", pit)]]
    fitted = {"method": "direct_horizon_v1", "last_target_season": 2025, "features": list(workload.FEATURES),
              "models": {str(h): {role: {"coefficients": [0, 420/h, 120, 60, *([0]*(len(workload.FEATURES)-3))],
                                         "max_volume": 1000} for role in ("bat", "pit")} for h in range(1, 5)}}
    h = workload.history(pd.DataFrame(rows), 2027, {1})
    assert h.set_index("role").loc["bat", "lag2"] == 0
    forecast = workload.predict(h, pd.Series({1: 28}), fitted, 2027, 2030)
    assert forecast[forecast.season.eq(2027)].set_index("role").pa.to_dict() == pytest.approx({"bat": 340.0, "pit": 130.0})
    # Year four reads the original history, not its own year-three expectation.
    assert forecast[forecast.season.eq(2030)].set_index("role").pa.to_dict() == pytest.approx({"bat": 130.0, "pit": 130.0})
    with pytest.raises(ValueError, match="overlaps"):
        workload.predict(h, pd.Series({1: 28}), fitted, 2025, 2027)
    with pytest.raises(ValueError, match="Unsupported workload horizon"):
        workload.predict(h, pd.Series({1: 28}), fitted, 2027, 2031)
    age_fit = {"method": "direct_volume_weighted_rate_v1", "last_target_season": 2025,
               "models": {str(year): {role: {"coefficients": [11-year, 0, 0, 0]}
                                      for role in ("bat", "pit")} for year in range(1, 5)}}
    forecast = aging.add_projection_shifts(forecast, h, pd.Series({1: 28}), age_fit, 2027)
    assert forecast.loc[forecast.season.eq(2027), "run_shift"].eq(0).all()
    assert forecast.loc[forecast.season.eq(2030), "run_shift"].eq(-3).all()
    with pytest.raises(ValueError, match="overlaps"):
        aging.add_projection_shifts(forecast, h, pd.Series({1: 28}), age_fit, 2025)
    data = one_hitter("forty_man", 600)
    data["playing_time"] = forecast
    c = pd.DataFrame([{"player_id": 1, "season": y, "salary_kind": "guaranteed", "salary_musd": 1,
                       "option_musd": None, "option_kind": None, "service_time": 7,
                       "fa_year": 2031, "how_acquired": "Free Agency"} for y in range(2027, 2031)])
    v, _ = score.assemble_values(**data, contracts=c, model=EMPTY, next_season=2027)
    assert v.iloc[0].playing_time == 340
    assert v.iloc[0].outlook[-1]["war"] != v.iloc[0].outlook[0]["war"]
    # Changing a two-way player's pitching workload must not scale his batting production.
    pit = data["rates"].iloc[0].copy()
    pit["role"], pit["group"], pit["n"] = "pit", "SP", 600
    for k in inputs_mod.OUTCOMES:
        pit[k] = 1.0 if k == "OUT" else 0.0
    data["rates"] = pd.concat([data["rates"], pd.DataFrame([pit.to_dict()])], ignore_index=True)
    with_pitching, _ = score.assemble_values(**data, contracts=c, model=EMPTY, next_season=2027)
    data["playing_time"] = forecast.assign(pa=forecast.pa.where(forecast.role.eq("bat"), 0))
    without_pitching, _ = score.assemble_values(**data, contracts=c, model=EMPTY, next_season=2027)
    assert with_pitching.iloc[0].playing_time == without_pitching.iloc[0].playing_time
    assert with_pitching.iloc[0].grade == without_pitching.iloc[0].grade
    assert with_pitching.iloc[0].projected_war > without_pitching.iloc[0].projected_war

    # The economic forecast is total WAR, not a grade/runs estimate rescaled twice.
    model = ValueModel.from_payload(EMPTY.to_payload())
    model.production = {"method": production.METHOD}
    data["playing_time"] = forecast.assign(expected_war=forecast.role.map({"bat": 3.0, "pit": 2.0}))
    details = []
    direct, _ = score.assemble_values(**data, contracts=c, model=model, next_season=2027, projection_details=details)
    assert direct.iloc[0].projected_war == 5.0
    assert direct.iloc[0].grade == with_pitching.iloc[0].grade
    assert all(w == 5.0 for w in details[0]["war_by_season"].values())
    assert pd.isna(direct.iloc[0].trade_value_low)  # Grade draws are not WAR uncertainty.
    data["playing_time"].loc[data["playing_time"].role.eq("pit"), "expected_war"] = float("nan")
    unknown, _ = score.assemble_values(**data, contracts=c, model=model, next_season=2027)
    assert pd.isna(unknown.iloc[0].trade_value_musd)
    assert all(o["war"] is None for o in unknown.iloc[0].outlook)
    # Calibrated batting and pitching dollars must stay separate for two-way
    # players, with one replacement salary, throughout the serving path.
    model.price.update(role_prices={"bat": 8, "SP": 12, "RP": 15}, above_minimum=True, inflation=0)
    data["playing_time"] = forecast.assign(expected_war=forecast.role.map({"bat": 3.0, "pit": 2.0}),
                                           pricing_role=forecast.role.map({"bat": "bat", "pit": "SP"}))
    priced, _ = score.assemble_values(**data, contracts=c, model=model, next_season=2027)
    assert priced.iloc[0].market_value_musd == pytest.approx(48.8)
    assert priced.iloc[0].projected_war == 5
    assert priced.iloc[0].tip["prices_by_role_musd"] == {"bat": 8, "SP": 12}
    assert priced.iloc[0].tip["price_per_win_musd"] == 9.6

    # Authentic source field and source completeness are load-bearing boundaries.
    # FIP-based fWAR must never be replaced by pitching RA9-WAR or a missing zero.
    payload = {"stats": [{"type": {"displayName": "sabermetrics"},
        "group": {"displayName": group}, "totalSplits": 1,
        "splits": [{"season": "2024", "sport": {"id": 1}, "player": {"id": 1},
                    "stat": {"war": war, "ra9War": 99}}]}
        for group, war in (("hitting", -0.5), ("pitching", 3.42056))]}
    parsed = production.parse_war(payload, 2024)
    assert parsed.war.tolist() == [-0.5, 3.42056]
    with pytest.raises(ValueError, match="Wrong fWAR season"):
        production.parse_war(payload, 2025)
    payload["stats"][0]["totalSplits"] = 2
    with pytest.raises(ValueError, match="Incomplete fWAR"):
        production.parse_war(payload, 2024)
    payload["stats"][0]["totalSplits"] = 1
    del payload["stats"][1]["splits"][0]["stat"]["war"]
    with pytest.raises(ValueError, match="missing fWAR"):
        production.parse_war(payload, 2024)
    model.production = {"method": "direct_total_bwar_v1"}
    with pytest.raises(ValueError, match="requires a fitted fWAR"):
        score.assemble_values(**data, contracts=c, model=model, next_season=2027)

    # No target-year values, synthetic recursive history, or survivor-only targets.
    war_lines = pd.DataFrame(rows).assign(war=2.0, skill=0.0)
    war_lines["volume"] /= 600
    h = production.history(war_lines, 2027)
    monkeypatch.setattr(bio, "october_roster", lambda season: pd.DataFrame({"player_id": [1], "status": ["D60"]}) if season == 2026 else pytest.fail("Roster looked past forecast cutoff"))
    monkeypatch.setattr(bio, "surgery_dates", lambda: pd.DataFrame({"player_id": [1, 1], "date": pd.to_datetime(["2026-06-01", "2027-06-01"])}))
    known = production.history(war_lines, 2027, include_roster=True, include_surgery=True)
    assert known.tj_count.eq(1).all() and known.tj_years.between(.4, .5).all()
    assert known.injured.eq(1).all() and known.rostered.eq(1).all()
    monkeypatch.setattr(production, "aaa_hitting", lambda season: pd.DataFrame({"player_id": [1], "role": ["bat"],
        **{k: [.25] for k in production.AAA_FEATURES[1:]}}) if season == 2026 else pytest.fail("Triple-A inputs looked past forecast cutoff"))
    with_aaa = production.history(war_lines, 2027, include_aaa=True)
    assert with_aaa.loc[with_aaa.role.eq("bat"), "aaa_volume"].eq(.25).all()
    assert with_aaa.loc[with_aaa.role.eq("pit"), "aaa_volume"].eq(0).all()
    assert h.volume1.max() < 1
    coefficients = [0, 1, *([0]*(len(production.FEATURES)-1))]
    fitted = {"method": production.METHOD, "features": production.FEATURES, "last_target_season": 2025,
              "models": {str(i): {r: {"coefficients": coefficients} for r in ("bat", "pit")} for i in range(1, 5)}}
    prediction = production.predict(h, fitted, 2027, 2030)
    assert prediction.expected_war.eq(2.0).all()
    with pytest.raises(ValueError, match="overlaps"):
        production.predict(h, fitted, 2025, 2027)
    with pytest.raises(ValueError, match="Unsupported WAR horizon"):
        production.predict(h, fitted, 2027, 2031)


def test_market_dollars_and_package_accounting_fail_closed():
    """Protect the calibration units, asset coverage, and release gate together."""
    from dataclasses import asdict

    from v2.players import market, pricing
    from v2.players.model import publish_model

    signing = {"player_id": 1, "first_season": 2025, "transaction_type": "free_agent_signing",
               "date": "2024-12-01", "source": "https://example.org/signing", "years": 2,
               "cash_musd": [8, 10, 2], "guarantee_musd": 20}
    cost, wins = pricing.contract_present_values(signing, [2, 1], .08, .03)
    assert cost == pytest.approx(8 + 10*.92 + 2*.92**2)  # bonus/buyout cash after play ends
    assert wins == pytest.approx(2 + .92*1.03)          # both production years, actual workload
    with pytest.raises(ValueError, match="signing events"):
        pricing.contract_present_values({**signing, "transaction_type": "extension", "how_acquired": "Free Agency"}, [2, 1], .08, .03)
    with pytest.raises(ValueError, match="horizons"):
        pricing.contract_present_values(signing, [2], .08, .03)
    with pytest.raises(ValueError, match="reconcile"):
        pricing.contract_present_values({**signing, "cash_musd": [8, 10]}, [2, 1], .08, .03)
    # The same documented rotation role must price calibration and later replay.
    rotation = {**signing, "signing_role": "SP", "role_source": signing["source"]}
    assert pricing.contract_roles(rotation, {"RP": .5}) == {"SP": .5}
    assert pricing.contract_roles(signing, {"RP": .5}) == {"RP": .5}
    with pytest.raises(ValueError, match="dated source"):
        pricing.contract_roles({**signing, "signing_role": "SP"}, {"RP": .5})
    with pytest.raises(ValueError, match="conflicts"):
        pricing.contract_roles(rotation, {"bat": 2, "SP": 1})
    frame = pd.DataFrame({"player_id": [1, 2, 3], "first_season": [2025]*3,
                          "salary_pv_musd": [cost, cost*2, 5], "production_pv_wins": [wins, wins*2, 0],
                          "replacement_pv_musd": [.76+.78*.92, .76+.78*.92, .76]})
    price = pricing.fit_price_per_win(frame, 2025)
    assert price.price_musd == pytest.approx((5*wins*cost-3*wins*(.76+.78*.92))/(5*wins*wins)) and price.n == 3
    assert price.price_for(2027, 2027) == pytest.approx(price.price_musd*1.03**2)
    with pytest.raises(ValueError, match="Duplicate"):
        pricing.fit_price_per_win(pd.concat([frame, frame.iloc[[0]]]), 2025)

    model = ValueModel.empty([2024], (2025, 2026))
    model.price = asdict(price)
    assert model.price_model().value_for(0, 2025, 2025) == .76
    replacement = project_outlook({2025: 0}, {}, [ContractYear(2025, "pre_arb", None, None)],
                                  model.price_model(), {1: .25}, 0, 2025)
    assert replacement[0].in_control and replacement[0].surplus_musd == 0
    # Contract and trade paths must use the same role premiums without adding
    # fWAR, doubling the minimum, or silently reverting to the old dollar slope.
    profile_price = pricing.PriceModel(10, None, 0, 50, "test", 2025, True,
                                      {"bat": 10, "SP": 10, "RP": 10},
                                      {"expected_saves": 2})
    profile = {"expected_saves": 20}
    assert profile_price.value_for_roles({"bat": 1, "RP": .5}, 2025, 2025, profile) == pytest.approx(19.76)
    with pytest.raises(ValueError, match="usage profile"):
        profile_price.value_for_roles({"RP": .5}, 2025, 2025)
    with pytest.raises(ValueError, match="usage profile"):
        profile_price.value_for(.5, 2025, 2025, "RP")
    forecast = project_outlook({2025: .5}, {}, [ContractYear(2025, "guaranteed", 8, None)],
                               profile_price, {1: .25}, 6, 2025, war_by_role={2025: {"RP": .5}},
                               market_profiles={2025: profile})
    assert forecast[0].war == .5 and forecast[0].surplus_musd == pytest.approx(1.76)
    from v2.players.arbitration import FEATURES, METHOD

    salary_model = {"method": METHOD, "features": FEATURES, "coefficients": [0.0]*20,
                    "last_signing_season": 2024, "base_season": 2024, "inflation": .03}
    profile = {"expected_saves": 20, "bat_pa": 0, "pit_pa": 200}
    forecast = project_outlook({2025: .5, 2026: .5}, {},
        [ContractYear(2025, "arbitration", 8, None), ContractYear(2026, "arbitration", None, None)],
        profile_price, {1: .25}, 3, 2025, war_by_role={2025: {"RP": .5}, 2026: {"RP": .5}},
        market_profiles={2025: profile, 2026: profile}, arbitration_model=salary_model)
    assert [r.salary_musd for r in forecast] == [8, 6.4]  # quoted salary, then prior-salary floor
    # None/NaN assets must not disappear from a total. Cash belongs to its recipient once.
    cash = {"name": "cash", "kind": "cash", "cash_musd": [2.5, 2.5]}
    prospect = {"name": "prospect", "kind": "prospect", "fv": "50", "role": "bat"}
    missing = {"name": "missing", "kind": "mlb", "player_id": 999}
    result = market.package_value([cash, prospect, missing], pd.DataFrame(), model, 2025)
    assert not result["complete_accounting"] and result["scenario_bounds_musd"] is None
    assert result["unpriced_assets"] == ["missing"]
    result = market.package_value([cash, prospect], pd.DataFrame(), model, 2025)
    assert result["complete_accounting"] and result["valued_count"] == 2
    assert result["scenario_bounds_musd"][0] == pytest.approx(4.8 + 28)
    model.price["price_musd"] *= 2
    assert market.package_value([prospect], pd.DataFrame(), model, 2025)["scenario_bounds_musd"][0] == 28
    # A successful price fit alone must fail before the DB or scoring inputs are touched.
    with pytest.raises(ValueError, match="not passed"):
        publish_model(model)
    with pytest.raises(ValueError, match="not passed"):
        score.score(None, model, write=True)
    from v2.players.model import MARKET_VALIDATION_CHECKS

    model.evaluation["market"] = {"status": "validated", "checks": {k: True for k in MARKET_VALIDATION_CHECKS},
                                  "model_fingerprint": model.validation_fingerprint()}
    model.require_market_validation()
    model.price["price_musd"] += .1
    with pytest.raises(ValueError, match="does not match"):
        model.require_market_validation()
