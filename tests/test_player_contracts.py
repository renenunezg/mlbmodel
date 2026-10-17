"""Baseball-Reference contract parsing: the one external format the valuation prices from."""
from pathlib import Path

from v2.players.contracts import parse_team_page

FIXTURE = (Path(__file__).parent / "fixtures" / "bbref" / "nyy-salaries-and-contracts.html").read_text(encoding="utf-8")


def test_parse_guaranteed_row():
    rows = parse_team_page(FIXTURE, season=2026)
    judge = rows[rows.player_name == "Aaron Judge"].sort_values("season")
    assert judge.bbref_id.iloc[0] == "judgeaa01"
    assert judge.service_time.iloc[0] == 9.051
    assert list(judge.season) == [2026, 2027, 2028, 2029, 2030, 2031, 2032]
    assert judge.salary_kind.tolist()[:6] == ["guaranteed"] * 6 and judge.salary_kind.iloc[6] == "free_agent"
    assert judge.salary_musd.iloc[0] == 40.0 and judge.fa_year.iloc[0] == 2032
    assert judge.contract_start.iloc[0] == 2023 and judge.contract_years.iloc[0] == 9 and judge.contract_total_musd.iloc[0] == 360.0


def test_parse_arbitration_and_pre_arb():
    rows = parse_team_page(FIXTURE, season=2026)
    volpe = rows[rows.player_name == "Anthony Volpe"].sort_values("season")
    assert volpe.salary_kind.tolist() == ["guaranteed", "arbitration", "arbitration", "free_agent"]
    rice = rows[rows.player_name == "Ben Rice"].sort_values("season")
    assert rice.salary_kind.tolist()[:2] == ["pre_arb", "pre_arb"]
    assert rice.salary_musd.isna().iloc[0]


def test_parse_buyout_marks_option():
    rows = parse_team_page(FIXTURE, season=2026)
    stanton = rows[(rows.player_name == "Giancarlo Stanton") & (rows.season == 2028)].iloc[0]
    assert stanton.salary_kind == "buyout" and stanton.salary_musd == 10.0 and stanton.option_musd == 25.0
    assert stanton.option_kind == "team"


def test_money_cells_in_thousands():
    from v2.players.contracts import _money_musd

    assert _money_musd("$750k") == 0.75 and _money_musd("$750,000") == 0.75 and _money_musd("$0") == 0.0


def test_page_season_comes_from_the_header():
    rows = parse_team_page(FIXTURE)
    assert rows.season.min() == 2026


def test_blank_after_arbitration_is_arbitration():
    html = (
        '<table id="payroll"><thead><tr><th data-stat="contract_y0" aria-label="2026"></th></tr></thead><tbody><tr>'
        '<th data-stat="player"><a href="/players/x/testpl01.shtml">Test Player</a></th>'
        '<td data-stat="service_time">2.100</td><td data-stat="how_acquired">Amateur Draft</td>'
        '<td data-stat="contract_y0">Arb</td><td data-stat="contract_y1"></td><td data-stat="contract_y2"></td>'
        '</tr></tbody></table>'
    )
    rows = parse_team_page(html)
    assert rows.salary_kind.tolist()[:3] == ["arbitration", "arbitration", "arbitration"]
