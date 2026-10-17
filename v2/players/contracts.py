"""Contracts, service time, and control years from Baseball-Reference.

Thirty team pages a week, three seconds apart, cached for seven days. Each
page's payroll table gives every 40-man player's service time, how he was
acquired, and a cell per future season holding a guaranteed salary, a buyout
with its option, an arbitration or pre-arbitration marker, or free agency.
Baseball-Reference ids map to MLBAM ids through the Chadwick register.

Usage:
    python -m v2.players.contracts --season 2026 [--no-write]
"""
from __future__ import annotations

import argparse
import logging
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from sqlalchemy import text

from backend.db import engine, writes_allowed
from backend.log import setup_logging
from backend.team_mappings import TEAM_ID_BY_CODE

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "players"
BBREF = "https://www.baseball-reference.com"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
CRAWL_DELAY_SECONDS = 3.0
PAGE_CACHE_DAYS = 7
CHADWICK_CACHE_DAYS = 30
CHADWICK_URL = "https://raw.githubusercontent.com/chadwickbureau/register/master/data/people-{shard}.csv"
CHADWICK_SHARDS = [f"{i:x}" for i in range(16)]
# The share of parsed rows allowed to lack an MLBAM id before the run fails.
MAX_UNMATCHED_SHARE = 0.02
# Future-season cells on the payroll table, contract_y0 being the page's season.
# A page shows only as many columns as its longest contract, so a cheap player's
# arbitration and free-agent seasons may fall past the last column and are then
# extended from his service time (arbitration at 3.000, free agency at 6.000).
CONTRACT_COLUMNS = [f"contract_y{i}" for i in range(10)]
ARBITRATION_SERVICE = 3.0
FREE_AGENT_SERVICE = 6.0

# Baseball-Reference franchise codes equal this repo's team codes except the
# Athletics, whose franchise page stays under OAK (ATH there is the 1901-54 Philadelphia club).
BBREF_FRANCHISE = {"ATH": "OAK", "LAA": "ANA"}
BBREF_TEAMS: dict[str, str] = {
    "ARI": "arizona-diamondbacks", "ATH": "athletics", "ATL": "atlanta-braves",
    "BAL": "baltimore-orioles", "BOS": "boston-red-sox", "CHC": "chicago-cubs",
    "CHW": "chicago-white-sox", "CIN": "cincinnati-reds", "CLE": "cleveland-guardians",
    "COL": "colorado-rockies", "DET": "detroit-tigers", "HOU": "houston-astros",
    "KCR": "kansas-city-royals", "LAA": "los-angeles-angels", "LAD": "los-angeles-dodgers",
    "MIA": "miami-marlins", "MIL": "milwaukee-brewers", "MIN": "minnesota-twins",
    "NYM": "new-york-mets", "NYY": "new-york-yankees", "PHI": "philadelphia-phillies",
    "PIT": "pittsburgh-pirates", "SDP": "san-diego-padres", "SEA": "seattle-mariners",
    "SFG": "san-francisco-giants", "STL": "st-louis-cardinals", "TBR": "tampa-bay-rays",
    "TEX": "texas-rangers", "TOR": "toronto-blue-jays", "WSN": "washington-nationals",
}
assert set(BBREF_TEAMS) == set(TEAM_ID_BY_CODE)

CONTRACT_FRAME_COLUMNS = [
    "bbref_id", "player_name", "season", "salary_musd", "salary_kind", "option_musd",
    "option_kind", "service_time", "how_acquired", "fa_year",
]

_MONEY = re.compile(r"\$([\d.]+)\s*([MK]?)")
_OPTION_KIND = re.compile(r"(team|club|player|mutual|vesting)", re.IGNORECASE)


def _money_musd(text_value: str) -> float | None:
    match = _MONEY.search(text_value)
    if not match:
        return None
    amount = float(match.group(1))
    return amount / 1000.0 if match.group(2) == "K" else amount


def _franchise(team_abbr: str) -> str:
    return BBREF_FRANCHISE.get(team_abbr, team_abbr)


def team_page_url(team_abbr: str) -> str:
    return f"{BBREF}/teams/{_franchise(team_abbr)}/{BBREF_TEAMS[team_abbr]}-salaries-and-contracts.shtml"


def _get(url: str) -> requests.Response:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    time.sleep(CRAWL_DELAY_SECONDS)
    return response


def fetch_team_page(team_abbr: str) -> str:
    """Fetch one team's salaries-and-contracts page, cached on disk for a week."""
    path = CACHE_DIR / f"bbref_{team_abbr}.html"
    if path.exists():
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        if age < timedelta(days=PAGE_CACHE_DAYS):
            return path.read_text(encoding="utf-8")
    response = _get(team_page_url(team_abbr))
    if response.status_code == 404:
        # The slug changed; discover it from the team's index page.
        franchise = _franchise(team_abbr)
        index = _get(f"{BBREF}/teams/{franchise}/{date.today().year}.shtml")
        index.raise_for_status()
        match = re.search(r'href="(/teams/[A-Z]+/[^"]*salaries-and-contracts\.shtml)"', index.text)
        if not match:
            raise RuntimeError(f"No salaries-and-contracts link found for {team_abbr}")
        response = _get(BBREF + match.group(1))
    response.raise_for_status()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(response.text, encoding="utf-8")
    return response.text


def _cell_text(cell) -> str:
    return " ".join(cell.get_text(" ", strip=True).split())


def _parse_cell(cell) -> tuple[str, float | None, float | None, str | None]:
    """One future-season cell -> (salary_kind, salary_musd, option_musd, option_kind)."""
    raw = _cell_text(cell)
    if raw == "":
        return "blank", None, None, None
    if raw.startswith("FA"):
        return "free_agent", None, None, None
    if raw.startswith("Arb"):
        return "arbitration", None, None, None
    if raw.startswith("Pre-Arb"):
        return "pre_arb", None, None, None
    salary = _money_musd(raw)
    if salary is None and re.fullmatch(r"[\d.]+", raw):
        salary = float(raw)  # a bare "0": a season with nothing owed
    if salary is None:
        raise ValueError(f"Unrecognised contract cell {raw!r}")
    tooltip = cell.find(attrs={"data-tip": True})
    if "[" in raw and tooltip is not None:
        tip = tooltip["data-tip"]
        kind_match = _OPTION_KIND.search(tip)
        kind = kind_match.group(1).lower() if kind_match else "team"
        return "buyout", salary, _money_musd(tip), "club" if kind == "club" else kind
    return "guaranteed", salary, None, None


def _extend_schedule(parsed: list[tuple], service_time: float, season: int) -> list[tuple]:
    """Fill blank cells and seasons past the last column from service time.

    Blank cells before free agency are pre-arbitration seasons. When no free-agent
    marker is visible, the schedule continues one season at a time, each adding a
    full year of service, until free agency.
    """
    visible = [row for row in parsed if row[1] != "blank" or any(r[1] != "blank" for r in parsed[parsed.index(row):])]
    out: list[tuple] = []
    for yr, kind, salary, option, option_kind in visible:
        if kind == "free_agent":
            out.append((yr, kind, salary, option, option_kind))
            return out
        if kind == "blank":
            kind = "pre_arb"
        out.append((yr, kind, salary, option, option_kind))
    if service_time != service_time:  # NaN: nothing to extend from
        return out
    last_year = out[-1][0] if out else season - 1
    projected_service = service_time + (last_year - season + 1)
    year = last_year + 1
    while projected_service < FREE_AGENT_SERVICE:
        kind = "arbitration" if projected_service >= ARBITRATION_SERVICE else "pre_arb"
        out.append((year, kind, None, None, None))
        projected_service += 1.0
        year += 1
    out.append((year, "free_agent", None, None, None))
    return out


def parse_team_page(html: str, season: int) -> pd.DataFrame:
    """One row per player per contract season from a team's payroll table."""
    soup = BeautifulSoup(html.replace("<!--", "").replace("-->", ""), "lxml")
    table = soup.find("table", id="payroll")
    if table is None:
        raise ValueError("No payroll table in page")
    rows: list[dict] = []
    for tr in table.find("tbody").find_all("tr"):
        cells = {td["data-stat"]: td for td in tr.find_all(["td", "th"]) if td.has_attr("data-stat")}
        name_cell = cells.get("player")
        if name_cell is None or name_cell.find("a") is None:
            continue
        link = name_cell.find("a")["href"]
        bbref_id = link.rsplit("/", 1)[-1].replace(".shtml", "")
        player_name = _cell_text(name_cell).rstrip("*#+ ").strip()
        service_time = float(_cell_text(cells["service_time"]) or "nan") if "service_time" in cells else float("nan")
        how_acquired = _cell_text(cells["how_acquired"]) if "how_acquired" in cells else ""
        parsed = []
        for offset, column in enumerate(CONTRACT_COLUMNS):
            if column not in cells:
                break
            parsed.append((season + offset, *_parse_cell(cells[column])))
        parsed = _extend_schedule(parsed, service_time, season)
        fa_year = next((yr for yr, kind, *_ in parsed if kind == "free_agent"), None)
        for yr, kind, salary, option, option_kind in parsed:
            rows.append({
                "bbref_id": bbref_id, "player_name": player_name, "season": yr,
                "salary_musd": salary, "salary_kind": kind, "option_musd": option,
                "option_kind": option_kind, "service_time": service_time,
                "how_acquired": how_acquired, "fa_year": fa_year,
            })
            if kind == "free_agent":
                break
    frame = pd.DataFrame(rows, columns=CONTRACT_FRAME_COLUMNS)
    frame["fa_year"] = frame["fa_year"].astype("Int64")
    return frame


def chadwick_map() -> dict[str, int]:
    """Baseball-Reference id -> MLBAM id from the Chadwick register, cached for a month."""
    path = CACHE_DIR / "chadwick.parquet"
    if path.exists() and datetime.now() - datetime.fromtimestamp(path.stat().st_mtime) < timedelta(days=CHADWICK_CACHE_DAYS):
        frame = pd.read_parquet(path)
    else:
        frames = []
        for shard in CHADWICK_SHARDS:
            frames.append(pd.read_csv(
                CHADWICK_URL.format(shard=shard), usecols=["key_mlbam", "key_bbref"],
                dtype={"key_bbref": "string"}, low_memory=False,
            ))
        frame = pd.concat(frames, ignore_index=True).dropna()
        frame["key_mlbam"] = frame["key_mlbam"].astype("int64")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    return dict(zip(frame["key_bbref"], frame["key_mlbam"], strict=True))


def build_contracts(season: int, teams: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All teams' contract rows with MLBAM ids, plus the rows that could not be matched."""
    mapping = chadwick_map()
    frames = []
    for team in teams or sorted(BBREF_TEAMS):
        frame = parse_team_page(fetch_team_page(team), season)
        frame["team"] = team
        frame["source_url"] = team_page_url(team)
        frames.append(frame)
        log.info(f"contracts: {team} {frame.bbref_id.nunique()} players, {len(frame)} rows")
    rows = pd.concat(frames, ignore_index=True)
    rows["player_id"] = rows["bbref_id"].map(mapping).astype("Int64")
    unmatched = rows[rows["player_id"].isna()][["bbref_id", "player_name", "team"]].drop_duplicates()
    share = len(rows[rows["player_id"].isna()]) / max(1, len(rows))
    if share > MAX_UNMATCHED_SHARE:
        raise RuntimeError(f"{share:.1%} of contract rows lack an MLBAM id: {unmatched.player_name.tolist()}")
    matched = rows.dropna(subset=["player_id"]).copy()
    matched["player_id"] = matched["player_id"].astype("int64")
    # A player traded during the week can appear on two pages; keep the later fetch order's last row.
    matched = matched.drop_duplicates(subset=["player_id", "season"], keep="last")
    return matched, unmatched


def publish_contracts(frame: pd.DataFrame) -> None:
    """Replace the contracts table with this fetch, in one transaction."""
    columns = [
        "player_id", "season", "salary_musd", "salary_kind", "option_musd", "option_kind",
        "service_time", "how_acquired", "fa_year", "team", "player_name", "source_url",
    ]
    payload = frame[columns].astype(object).where(frame[columns].notna(), None)
    fetched_at = datetime.utcnow()
    records = [dict(zip(columns, row, strict=True), fetched_at=fetched_at) for row in payload.itertuples(index=False)]
    insert = text(
        "INSERT INTO player_contracts (" + ", ".join(columns) + ", fetched_at, updated_at) VALUES ("
        + ", ".join(f":{c}" for c in columns) + ", :fetched_at, now())"
    )
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM player_contracts"))
        conn.execute(insert, records)
    log.info(f"contracts: published {len(records)} rows")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch Baseball-Reference contracts into mlb.player_contracts.")
    ap.add_argument("--season", type=int, default=date.today().year)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    probe = requests.get(f"{BBREF}/robots.txt", headers={"User-Agent": USER_AGENT}, timeout=30)
    if probe.status_code != 200:
        log.warning(
            f"Baseball-Reference returned {probe.status_code}; blocked from this runner. "
            "Run the weekly fetch locally with MLBMODEL_DB_WRITES=1."
        )
        return
    contracts, unmatched = build_contracts(args.season)
    log.info(f"contracts: {contracts.player_id.nunique()} players, {len(contracts)} rows, {len(unmatched)} unmatched")
    for row in unmatched.itertuples(index=False):
        log.info(f"unmatched: {row.player_name} ({row.team}, {row.bbref_id})")
    if args.no_write or not writes_allowed():
        log.info("contracts: no write")
        return
    publish_contracts(contracts)


if __name__ == "__main__":
    setup_logging()
    main()
