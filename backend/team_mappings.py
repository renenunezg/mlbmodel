"""
Team name normalization.

The MLB Stats API returns canonical abbreviations (LAD, NYY, etc.).
This map handles edge cases from other data sources (FanGraphs, Baseball
Savant, The Odds API) that use full names, nicknames, or variant abbrevs.
"""

TEAM_NAME_MAP = {
    # Canonical abbreviations (identity mappings for safety)
    "LAA": "LAA", "HOU": "HOU", "ATH": "ATH", "TOR": "TOR",
    "ATL": "ATL", "MIL": "MIL", "STL": "STL", "CHC": "CHC",
    "ARI": "ARI", "LAD": "LAD", "SFG": "SFG", "CLE": "CLE",
    "SEA": "SEA", "MIA": "MIA", "NYM": "NYM", "WSN": "WSN",
    "BAL": "BAL", "SDP": "SDP", "PHI": "PHI", "PIT": "PIT",
    "TEX": "TEX", "TBR": "TBR", "BOS": "BOS", "CIN": "CIN",
    "COL": "COL", "KCR": "KCR", "DET": "DET", "MIN": "MIN",
    "CHW": "CHW", "NYY": "NYY",

    # MLB Stats API variants
    "AZ": "ARI", "OAK": "ATH", "KC": "KCR", "TB": "TBR",
    "SD": "SDP", "SF": "SFG", "WAS": "WSN", "WSH": "WSN",
    "CWS": "CHW",

    # Full names (FanGraphs, Odds API)
    "Los Angeles Angels": "LAA", "Houston Astros": "HOU",
    "Oakland Athletics": "ATH", "Toronto Blue Jays": "TOR",
    "Atlanta Braves": "ATL", "Milwaukee Brewers": "MIL",
    "St. Louis Cardinals": "STL", "St Louis Cardinals": "STL",
    "Chicago Cubs": "CHC", "Arizona Diamondbacks": "ARI",
    "Los Angeles Dodgers": "LAD", "San Francisco Giants": "SFG",
    "Cleveland Guardians": "CLE", "Seattle Mariners": "SEA",
    "Miami Marlins": "MIA", "New York Mets": "NYM",
    "Washington Nationals": "WSN", "Baltimore Orioles": "BAL",
    "San Diego Padres": "SDP", "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT", "Texas Rangers": "TEX",
    "Tampa Bay Rays": "TBR", "Boston Red Sox": "BOS",
    "Cincinnati Reds": "CIN", "Colorado Rockies": "COL",
    "Kansas City Royals": "KCR", "Detroit Tigers": "DET",
    "Minnesota Twins": "MIN", "Chicago White Sox": "CHW",
    "New York Yankees": "NYY",

    # Nicknames only (FanGraphs uses these)
    "Angels": "LAA", "Astros": "HOU", "Athletics": "ATH",
    "Blue Jays": "TOR", "Braves": "ATL", "Brewers": "MIL",
    "Cardinals": "STL", "Cubs": "CHC", "Diamondbacks": "ARI",
    "D-backs": "ARI", "Dodgers": "LAD", "Giants": "SFG",
    "Guardians": "CLE", "Mariners": "SEA", "Marlins": "MIA",
    "Mets": "NYM", "Nationals": "WSN", "Orioles": "BAL",
    "Padres": "SDP", "Phillies": "PHI", "Pirates": "PIT",
    "Rangers": "TEX", "Rays": "TBR", "Red Sox": "BOS",
    "Reds": "CIN", "Rockies": "COL", "Royals": "KCR",
    "Tigers": "DET", "Twins": "MIN", "White Sox": "CHW",
    "Yankees": "NYY",

    # City-only variants (various sources)
    "Sacramento": "ATH",
}


def normalize_team(name: str) -> str:
    """Normalize a team name to its canonical 3-letter abbreviation."""
    if not isinstance(name, str):
        return name
    return TEAM_NAME_MAP.get(name.strip(), name.strip())


# MLB Stats API team_id by canonical 3-letter code.
TEAM_ID_BY_CODE: dict[str, int] = {
    "LAA": 108, "HOU": 117, "ATH": 133, "TOR": 141, "ATL": 144, "MIL": 158,
    "STL": 138, "CHC": 112, "ARI": 109, "LAD": 119, "SFG": 137, "CLE": 114,
    "SEA": 136, "MIA": 146, "NYM": 121, "WSN": 120, "BAL": 110, "SDP": 135,
    "PHI": 143, "PIT": 134, "TEX": 140, "TBR": 139, "BOS": 111, "CIN": 113,
    "COL": 115, "KCR": 118, "DET": 116, "MIN": 142, "CHW": 145, "NYY": 147,
}
