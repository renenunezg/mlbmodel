import pandas as pd
from backend.team_mappings import TEAM_NAME_MAP
import re

def clean_odds(value):
    if isinstance(value, str):
        value = value.strip()
        # Replace "even" or "even +" with "+100"
        if value.lower() in ['even', 'even +']:
            return '+100'
        # Remove any trailing extra plus signs (a space followed by a plus at the end)
        value = re.sub(r'\s+\+$', '', value)
    return value

url = "https://www.vegasinsider.com/mlb/odds/las-vegas/"

try:
    tables = pd.read_html(url)
    df = tables[0]

    # Drop unnecessary rows
    df = df[df['Time'].astype(str).str.contains(r'\d+\s+.+?\s+[A-Z][a-z]+\s+\([LR]\)')]

    df = df.copy()
    df['Team_Info'] = df['Time'].str.extract(r'^\d+\s+(.*?)\s+[A-Za-z]+\s+\([LR]\)')[0].str.strip()

    def map_team(name):
        if not isinstance(name, str):
            return None
        for key, abbr in TEAM_NAME_MAP.items():
            if name.lower().startswith(key.lower()):
                return abbr
        return None

    df['Team'] = df['Team_Info'].apply(map_team)

    # Separate the DataFrame into moneyline, totals, and runlines based on the 'Open' column patterns.
    # Moneyline rows: 'Open' is exactly a moneyline number (e.g., +170, -205)
    ml_df = df[df['Open'].astype(str).str.match(r'^(even|even \+|[\+\-]\d+)$', case=False)].copy()
    ml_df.reset_index(drop=True, inplace=True)
    
    # Totals rows: 'Open' starts with 'o' or 'u' (over/under)
    totals_df = df[df['Open'].astype(str).str.match(r'^[ouOU].+')].copy()
    totals_df.reset_index(drop=True, inplace=True)
    
    # Runline rows: 'Open' starts with a plus or minus and contains a decimal (e.g., +1.5 -110)
    runline_df = df[df['Open'].astype(str).str.match(r'^[\+\-]\d+\.\d+.*')].copy()
    runline_df.reset_index(drop=True, inplace=True)
    
    # Assign game_id to each DataFrame (we assume each game has exactly 2 rows)
    ml_df['game_id'] = ml_df.index // 2
    totals_df['game_id'] = totals_df.index // 2
    runline_df['game_id'] = runline_df.index // 2
    
    # Merge totals and runlines into the moneyline DataFrame based on game_id.
    # We assume that the first row in ml_df for a game corresponds to the first row in totals_df/runline_df, and similarly for the second row.
    for gid in ml_df['game_id'].unique():
        # Get indices for the moneyline rows for this game.
        ml_indices = ml_df[ml_df['game_id'] == gid].index
        # Only proceed if there are exactly 2 moneyline rows.
        if len(ml_indices) != 2:
            continue
        # Merge totals if available.
        t_rows = totals_df[totals_df['game_id'] == gid]
        if len(t_rows) >= 1:
            ml_df.loc[ml_indices[0], 'Total'] = t_rows.iloc[0]['Bet365']
        if len(t_rows) >= 2:
            ml_df.loc[ml_indices[1], 'Total'] = t_rows.iloc[1]['Bet365']
        # Merge runlines if available.
        r_rows = runline_df[runline_df['game_id'] == gid]
        if len(r_rows) >= 1:
            ml_df.loc[ml_indices[0], 'RunLine'] = r_rows.iloc[0]['Bet365']
        if len(r_rows) >= 2:
            ml_df.loc[ml_indices[1], 'RunLine'] = r_rows.iloc[1]['Bet365']

    # Now drop rows that only had totals and runlines
    ml_df = ml_df[ml_df['Team'].notna()].reset_index(drop=True)

    # Rename columns and clean odds
    ml_df.rename(columns={"Bet365": "Bet365_ML"}, inplace=True)
    ml_df['Bet365_ML'] = ml_df['Bet365_ML'].apply(clean_odds)
    ml_df['Open'] = ml_df['Open'].apply(clean_odds)
    ml_df['Total'] = ml_df['Total'].apply(clean_odds)
    ml_df['RunLine'] = ml_df['RunLine'].apply(clean_odds)

    ml_df.rename(columns={
        "Team": "team",
        "Open": "open",
        "Bet365_ML": "bet365_ml",
        "Total": "total",
        "RunLine": "run_line"
    }, inplace=True)
    
    # Final DataFrame: we assume ml_df now has exactly 2 rows per game.
    ml_df = ml_df[['game_id', 'team', 'open', 'bet365_ml', 'total', 'run_line']]
    print(ml_df)

except Exception as e:
    print("❌ Error while scraping odds table:", e)
