"""
FanGraphs data fetcher.

Replaces: scrape_starting_pitchers.py, scrape_bullpen.py

Scrapes FanGraphs legacy endpoint for proprietary pitcher metrics not
available from the MLB Stats API: xFIP, FIP, SIERA.
Uses requests + BeautifulSoup (no Selenium).
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO
import time
from datetime import datetime, date, timedelta
from backend.team_mappings import TEAM_NAME_MAP

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
FANGRAPHS_LEGACY_URL = "https://www.fangraphs.com/leaders-legacy.aspx"

PITCHER_COLUMNS = ["pitcher_name", "team", "ip", "era", "fip", "xfip", "siera", "whip", "k_9", "bb_9", "hr_9"]


def _fetch_fangraphs_pitching(season: int, stats: str = "sta", qual: int = 0,
                               start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Fetch pitching data from FanGraphs legacy endpoint.

    Args:
        season: MLB season year
        stats: 'sta' for starters, 'rel' for relievers
        qual: Minimum IP qualifier (0 = no minimum)
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns raw DataFrame from FanGraphs table.
    """
    all_dfs = []
    page = 1
    max_pages = 10

    if start_date is None:
        # Default to last 30 days
        end_dt = datetime.today()
        start_dt = end_dt - timedelta(days=30)
        start_date = start_dt.strftime("%Y-%m-%d")
        end_date = end_dt.strftime("%Y-%m-%d")

    while page <= max_pages:
        url = (
            f"{FANGRAPHS_LEGACY_URL}?pos=all&stats={stats}&lg=all&qual={qual}"
            f"&type=1&season={season}&month=1000&season1={season}&ind=0&team=0"
            f"&rost=0&age=0&filter=&players=0"
            f"&startdate={start_date}&enddate={end_date}"
            f"&page={page}_50"
        )

        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"FanGraphs request failed on page {page}: {e}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        tables = soup.find_all("table", class_="rgMasterTable")

        if not tables:
            break

        df_page = pd.read_html(StringIO(str(tables[0])))[0]
        if df_page.empty:
            break

        all_dfs.append(df_page)
        page += 1
        time.sleep(1)  # Rate limit

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    # Flatten multi-level columns if present
    df.columns = [col[1] if isinstance(col, tuple) else str(col).strip() for col in df.columns]
    return df


def fetch_pitcher_stats(season: int = None) -> pd.DataFrame:
    """Fetch individual starting pitcher stats from FanGraphs.

    Returns DataFrame with columns:
        pitcher_name, team, season, role, ip, era, fip, xfip, siera, whip, k_9, bb_9, hr_9
    """
    if season is None:
        season = date.today().year

    df = _fetch_fangraphs_pitching(season, stats="sta")

    if df.empty:
        print("No starting pitcher data from FanGraphs.")
        return pd.DataFrame(columns=PITCHER_COLUMNS + ["season", "role"])

    # Select and rename columns
    col_map = {
        "Name": "pitcher_name",
        "Team": "team",
        "IP": "ip",
        "ERA": "era",
        "FIP": "fip",
        "xFIP": "xfip",
        "SIERA": "siera",
        "WHIP": "whip",
        "K/9": "k_9",
        "BB/9": "bb_9",
        "HR/9": "hr_9",
    }

    available = {k: v for k, v in col_map.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available)

    # Clean numeric columns
    numeric_cols = ["ip", "era", "fip", "xfip", "siera", "whip", "k_9", "bb_9", "hr_9"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("%", "").str.replace(",", "").str.strip()
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize team names
    if "team" in df.columns:
        df["team"] = df["team"].map(TEAM_NAME_MAP).fillna(df["team"])

    df["season"] = season
    df["role"] = "starter"

    return df


def fetch_bullpen_stats(season: int = None) -> pd.DataFrame:
    """Fetch team-level bullpen stats from FanGraphs.

    Returns DataFrame with columns:
        team, season, ip, era, fip, xfip, siera, whip, k_9, bb_9, hr_9
    """
    if season is None:
        season = date.today().year

    # team=0,ts means aggregate by team
    all_dfs = []
    page = 1

    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=30)
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")

    url = (
        f"{FANGRAPHS_LEGACY_URL}?pos=all&stats=rel&lg=all&qual=0"
        f"&type=1&season={season}&month=1000&season1={season}&ind=0&team=0%2Cts"
        f"&rost=0&age=0&filter=&players=0"
        f"&startdate={start_date}&enddate={end_date}"
        f"&page=1_50"
    )

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"FanGraphs bullpen request failed: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.find_all("table", class_="rgMasterTable")

    if not tables:
        print("No bullpen tables found on FanGraphs.")
        return pd.DataFrame()

    df = pd.read_html(StringIO(str(tables[0])))[0]
    df.columns = [col[1] if isinstance(col, tuple) else str(col).strip() for col in df.columns]

    col_map = {
        "Team": "team",
        "IP": "ip",
        "ERA": "era",
        "FIP": "fip",
        "xFIP": "xfip",
        "SIERA": "siera",
        "WHIP": "whip",
        "K/9": "k_9",
        "BB/9": "bb_9",
        "HR/9": "hr_9",
    }

    available = {k: v for k, v in col_map.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available)

    # Clean numeric columns
    numeric_cols = ["ip", "era", "fip", "xfip", "siera", "whip", "k_9", "bb_9", "hr_9"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("%", "").str.replace(",", "").str.strip()
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize team names
    if "team" in df.columns:
        df["team"] = df["team"].map(TEAM_NAME_MAP).fillna(df["team"])

    df["season"] = season

    return df


if __name__ == "__main__":
    season = date.today().year

    print(f"=== Starting Pitcher Stats ({season}) ===")
    pitchers = fetch_pitcher_stats(season)
    if not pitchers.empty:
        print(pitchers.head(10).to_string(index=False))
        print(f"\nTotal pitchers: {len(pitchers)}")
    else:
        print("No data (season may not have started yet).")

    print(f"\n=== Bullpen Stats ({season}) ===")
    bullpen = fetch_bullpen_stats(season)
    if not bullpen.empty:
        print(bullpen.to_string(index=False))
    else:
        print("No data (season may not have started yet).")
