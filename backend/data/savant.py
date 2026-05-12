"""Park factors from Baseball Savant. CSV endpoint first, HTML scrape fallback."""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO
from datetime import date
from backend.team_mappings import TEAM_NAME_MAP

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

SAVANT_PARK_FACTORS_URL = "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"


def fetch_park_factors(season: int = None) -> pd.DataFrame:
    """Park factors per (team, venue, season). Defaults to last year."""
    if season is None:
        # Park factors use previous year's data since current season may not have enough
        season = date.today().year - 1

    # Try the CSV/JSON download endpoint first
    csv_url = (
        f"{SAVANT_PARK_FACTORS_URL}"
        f"?type=year&year={season}&batSide=&stat=index_wOBA&condition=All&rolling=3&csvType=statcast"
    )

    try:
        resp = requests.get(csv_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()

        # Check if response is CSV
        content_type = resp.headers.get("Content-Type", "")
        if "csv" in content_type or "text/plain" in content_type:
            df = pd.read_csv(StringIO(resp.text))
            return _process_park_factors(df, season)
    except Exception as e:
        print(f"CSV endpoint failed, trying HTML: {e}")

    # Fallback: HTML scraping (no Selenium)
    html_url = (
        f"{SAVANT_PARK_FACTORS_URL}"
        f"?type=year&year={season}&batSide=&stat=index_wOBA&condition=All&rolling=3&parks=mlb"
    )

    try:
        resp = requests.get(html_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()

        tables = pd.read_html(StringIO(resp.text))
        if tables:
            for table in tables:
                cols_lower = [str(c).lower() for c in table.columns]
                if any("team" in c for c in cols_lower) and any("venue" in c for c in cols_lower):
                    return _process_park_factors(table, season)

        # If pd.read_html doesn't find it, try BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table")
        if table:
            df = pd.read_html(StringIO(str(table)))[0]
            return _process_park_factors(df, season)

    except Exception as e:
        print(f"HTML scraping failed: {e}")

    print("Could not fetch park factors. Baseball Savant may require JavaScript rendering.")
    print("Consider using cached/static park factors as fallback.")
    return _static_park_factors(season)


def _process_park_factors(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Clean and normalize park factors DataFrame."""
    # Normalize column names
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]

    # Find team and venue columns
    team_col = next((c for c in df.columns if "team" in c), None)
    venue_col = next((c for c in df.columns if "venue" in c or "park" in c), None)
    pf_col = next((c for c in df.columns if c in ("park_factor", "pf", "index_woba", "woba")), None)

    if team_col is None:
        print(f"Warning: no team column found in columns: {df.columns.tolist()}")
        return pd.DataFrame()

    result = pd.DataFrame()
    result["team"] = df[team_col].map(TEAM_NAME_MAP).fillna(df[team_col])
    result["venue"] = df[venue_col] if venue_col else ""
    result["season"] = season

    if pf_col:
        result["park_factor"] = pd.to_numeric(df[pf_col], errors="coerce").fillna(100).astype(int)
    else:
        result["park_factor"] = 100

    return result


def _static_park_factors(season: int) -> pd.DataFrame:
    """Fallback static park factors based on historical averages.
    These are approximate 3-year rolling averages and should be updated
    if the dynamic fetch consistently fails.
    """
    factors = {
        "COL": ("Coors Field", 116),
        "ARI": ("Chase Field", 106),
        "BOS": ("Fenway Park", 105),
        "CIN": ("Great American Ball Park", 104),
        "TEX": ("Globe Life Field", 103),
        "CHC": ("Wrigley Field", 102),
        "TOR": ("Rogers Centre", 102),
        "PHI": ("Citizens Bank Park", 101),
        "ATL": ("Truist Park", 101),
        "MIL": ("American Family Field", 101),
        "BAL": ("Camden Yards", 100),
        "MIN": ("Target Field", 100),
        "NYY": ("Yankee Stadium", 100),
        "LAA": ("Angel Stadium", 100),
        "CHW": ("Guaranteed Rate Field", 100),
        "HOU": ("Minute Maid Park", 100),
        "DET": ("Comerica Park", 99),
        "KCR": ("Kauffman Stadium", 99),
        "WSN": ("Nationals Park", 99),
        "STL": ("Busch Stadium", 99),
        "SFG": ("Oracle Park", 98),
        "NYM": ("Citi Field", 98),
        "CLE": ("Progressive Field", 97),
        "PIT": ("PNC Park", 97),
        "SDP": ("Petco Park", 96),
        "SEA": ("T-Mobile Park", 96),
        "LAD": ("Dodger Stadium", 96),
        "TBR": ("Tropicana Field", 96),
        "MIA": ("loanDepot park", 95),
        "ATH": ("Sacramento Sutter Health Park", 100),  # New stadium, default to neutral
    }

    rows = [
        {"team": team, "venue": venue, "season": season, "park_factor": pf}
        for team, (venue, pf) in factors.items()
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=== Park Factors ===")
    pf = fetch_park_factors()
    if not pf.empty:
        print(pf.sort_values("park_factor", ascending=False).to_string(index=False))
    else:
        print("No park factors available.")
