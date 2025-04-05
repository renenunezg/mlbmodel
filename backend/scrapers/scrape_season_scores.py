import os
from sqlalchemy import create_engine
import pandas as pd
from pybaseball import schedule_and_record
from concurrent.futures import ThreadPoolExecutor
import time
import requests
import random
from bs4 import BeautifulSoup
from ratelimit import limits, sleep_and_retry

from dotenv import load_dotenv
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Limit to 30 calls per minute (Baseball Reference may allow more, but this is safe)
CALLS = 30
PERIOD = 60

# Import your existing mappings (adjust import path as needed)
from backend.team_mappings import TEAM_NAME_MAP

# Add reverse mapping for full team names to abbreviations
TEAM_NAME_MAP.update({
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Yankees": "NYY",
    "New York Mets": "NYM",
    "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP",
    "Seattle Mariners": "SEA",
    "San Francisco Giants": "SFG",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN"
})

load_dotenv()

# Baseball Reference abbreviations for teams
TEAM_ABBREVS = [
    "ARI","ATL","BAL","BOS","CHC","CHW","CIN","CLE","COL","DET",
    "HOU","KCR","LAA","LAD","MIA","MIL","MIN","NYY","NYM","OAK",
    "PHI","PIT","SDP","SEA","SFG","STL","TBR","TEX","TOR","WSN"
]

@sleep_and_retry
@limits(calls=CALLS, period=PERIOD)
def fetch_team_schedule(season, team):
    url = f"https://www.baseball-reference.com/teams/{team}.shtml"
    print(f"Fetching data from: {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.5993.70 Safari/537.36"
    }
    max_retries = 5
    delay = 5

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 429:
                raise requests.exceptions.HTTPError("429 Too Many Requests")
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "lxml")
            time.sleep(random.uniform(5, 8))  # Slight jitter
            tables = soup.find_all("table")
            parsed_tables = pd.read_html(str(tables))
            for i, table in enumerate(parsed_tables):
                if "Date" in table.columns and "W/L" in table.columns:
                    df = table
                    break
            else:
                raise ValueError("No valid table with expected columns found.")
            df["Team_BRef"] = team
            return df
        except requests.exceptions.HTTPError as exc:
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt) + random.uniform(1, 3)
                print(f"⚠ {exc} for {team}, retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                print(f"❌ Could not fetch data for {team} after {max_retries} attempts: {exc}")
                return None
        except Exception as exc:
            print(f"❌ Unexpected error for {team}: {exc}")
            return None

def fetch_all_scores(season=2025):
    all_data = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        results = executor.map(lambda team: fetch_team_schedule(season, team), TEAM_ABBREVS)

    for df in results:
        if df is not None:
            all_data.append(df)
            print(f"✔ Data fetched for {df['Team_BRef'].iloc[0]}")

    if not all_data:
        print("⚠ No game data available for any team. Check if the season has started or if Baseball Reference has published the data.")
        return pd.DataFrame()

    full_df = pd.concat(all_data, ignore_index=True)

    # Rename columns for clarity
    full_df.rename(columns={
        "R": "runs_scored",
        "RA": "runs_allowed",
        "W/L": "win_loss",
        "Date": "game_date",
        "Opp": "opponent"
    }, inplace=True)

    # Map Opponent name to your known abbreviations
    def map_opponent(name):
        return TEAM_NAME_MAP.get(name, name)

    full_df["opponent_abbr"] = full_df["opponent"].apply(map_opponent)

    unmatched = full_df[full_df["opponent_abbr"] == full_df["opponent"]]["opponent"].unique()
    if len(unmatched) > 0:
        print("⚠ Unmatched team names found in opponent column:")
        for name in unmatched:
            print("-", name)

    return full_df

def print_results_to_terminal(df):
    """
    From the raw DataFrame:
    1) Keep only home-team rows (where Home_Away != '@').
    2) Rename columns to date, away, away_runs, home, home_runs.
    3) Add a simple game_id = date_home_away.
    4) Print cleaned results.
    """
    if df.empty or "Home_Away" not in df.columns:
        print("⚠ No valid game data to display.")
        return

    df_home = df[df["Home_Away"] != "@"].copy()
    
    df_home.dropna(subset=["runs_scored", "runs_allowed"], inplace=True)
    
    df_home["parsed_date"] = pd.to_datetime(df_home["game_date"], format="%A, %b %d", errors="coerce")
    df_home = df_home[df_home["parsed_date"] <= pd.to_datetime("today")]
    
    df_home["date"] = df_home["parsed_date"].dt.strftime("%m/%d")
    
    df_home["game_id"] = df_home["parsed_date"].dt.strftime('%m/%d/%Y') + "_" + df_home["opponent_abbr"] + "_" + df_home["Team_BRef"]
    
    df_home["home_runs"] = df_home["runs_scored"].astype(int)
    df_home["away_runs"] = df_home["runs_allowed"].astype(int)
    df_home["home"] = df_home["Team_BRef"]
    df_home["away"] = df_home["opponent_abbr"]

    df_clean = df_home[[
        "game_id", "date", "away", "away_runs", "home", "home_runs"
    ]]

    if df_clean.empty:
        print("⚠ No completed home games to display.")
        return

    print("=== CLEANED GAMES (HOME VIEW) ===")
    print(df_clean.head(20))
    print("TOTAL GAMES:", len(df_clean))
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        engine = create_engine(db_url)
        df_clean.to_sql("game_results", engine, if_exists="replace", index=False)
        print("✅ Saved to database table 'game_results'")
    else:
        print("❌ DATABASE_URL not set. Could not save to DB.")

def run():
    df = fetch_all_scores(season=2025)
    print_results_to_terminal(df)

if __name__ == "__main__":
    run()