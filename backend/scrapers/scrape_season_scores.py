import re
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from backend.team_mappings import TEAM_NAME_MAP
from backend.db import engine
from sqlalchemy.exc import SQLAlchemyError

# Define the season boundaries.
OPENING_DAY = datetime.strptime("March 27, 2025", "%B %d, %Y")
TODAY = datetime.today()

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")
    return webdriver.Chrome(options=chrome_options)

def normalize_team(name):
    return TEAM_NAME_MAP.get(name.strip(), name.strip())

def extract_games_for_date(date_str, games_text):
    games = []
    lines = games_text.splitlines()

    for line in lines:
        if "Boxscore" not in line:
            continue
        try:
            # Remove "Boxscore" text and try to parse the game line.
            parts = line.split("Boxscore")[0].strip()
            score_pattern = re.findall(r"(.+?)\s\((\d+)\)\s@\s(.+?)\s\((\d+)\)", parts)
            if not score_pattern:
                score_pattern = re.findall(r"(.+?)\s\((\d+)\)\s+at\s+(.+?)\s\((\d+)\)", parts)
            if not score_pattern:
                continue

            team1, score1, team2, score2 = score_pattern[0]
            away_team = normalize_team(team1.strip())
            home_team = normalize_team(team2.strip())
            away_score = int(score1)
            home_score = int(score2)

            games.append({
                'date': datetime.strptime(date_str, "%A, %B %d, %Y").date(),
                'away_team': away_team,
                'away_score': away_score,
                'home_team': home_team,
                'home_score': home_score
            })
        except Exception as e:
            print(f"Skipping line due to error: {e}")
            continue

    return games

def scrape_season_schedule():
    url = 'https://www.baseball-reference.com/leagues/majors/2025-schedule.shtml'
    driver = setup_driver()
    driver.get(url)

    # Get the raw text of the page body.
    text = driver.find_element("tag name", "body").text
    driver.quit()

    # Use regex to capture blocks starting with a full date.
    date_game_blocks = re.findall(
        r"((?:Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday),\s+[A-Za-z]+\s+\d{1,2},\s+2025)(.*?)(?=(?:Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday),\s+[A-Za-z]+\s+\d{1,2},\s+2025|$)",
        text,
        re.DOTALL
    )

    game_data = []
    for date_str, games_text in date_game_blocks:
        if "Boxscore" not in games_text:
            continue
        try:
            game_date = datetime.strptime(date_str, "%A, %B %d, %Y")
            if game_date.date() < OPENING_DAY.date() or game_date.date() > TODAY.date():
                continue
            games = extract_games_for_date(date_str, games_text)
            game_data.extend(games)
        except ValueError as e:
            print(f"Skipping block due to error: {e}")
            continue

    return pd.DataFrame(game_data)

def insert_into_db(df):
    try:
        df.to_sql("game_results", engine, if_exists="replace", index=False)
        print("✅ Inserted into game_results table.")
    except SQLAlchemyError as e:
        print("❌ Error inserting data into DB:", e)

if __name__ == "__main__":
    df = scrape_season_schedule()
    print(df.to_string(index=False))
    print(f"\n✅ Total games scraped: {len(df)}")
    insert_into_db(df)