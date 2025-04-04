import pandas as pd
from io import StringIO
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
from backend.team_mappings import TEAM_NAME_MAP
from sqlalchemy import create_engine, MetaData, Table
from dotenv import load_dotenv
import os

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
metadata = MetaData()
metadata.reflect(bind=engine)

def scrape_park_factors():
    url = "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors?type=year&year=2024&batSide=&stat=index_wOBA&condition=All&rolling=3&parks=mlb"

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)
    driver.get(url)

    time.sleep(3)  # Let the table load

    try:
        html = driver.page_source
        tables = pd.read_html(StringIO(html))
        df = None
        for table in tables:
            if "Team" in table.columns and "Venue" in table.columns:
                df = table
                break
        if df is not None:
            # Clean column names
            df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
            df.insert(0, 'rank', range(1, len(df) + 1))
            if 'hard_hit' in df.columns:
                df = df.rename(columns={'hard_hit': 'hardhit'})
            # Convert numeric columns (except team/venue/year) to numbers
            for col in df.columns:
                if col not in ['team', 'venue', 'year']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            print("Unique team names before mapping:", df['team'].unique())
            mapped = df['team'].map(TEAM_NAME_MAP)
            print("Mapped team values (with NaNs where no match):", mapped.unique())
            # Map team names using TEAM_NAME_MAP
            df['team'] = df['team'].map(TEAM_NAME_MAP).fillna(df['team'])
            
            df = df.rename(columns={
                'year': 'year_range',
                '1b': '_1b',
                '2b': '_2b',
                '3b': '_3b'
            })
            df.columns = [col if col not in ['1b', '2b', '3b'] else f"_{col}" for col in df.columns]

            park_factors_table = metadata.tables['park_factors']
            with engine.begin() as conn:
                conn.execute(park_factors_table.delete())  # Clear old data
                expected_columns = [c.name for c in park_factors_table.columns]
                df = df[expected_columns]
                conn.execute(park_factors_table.insert(), df.to_dict(orient="records"))
                print("Park factors table updated.")
            print("Cleaned and mapped DataFrame preview:")
            print(df.head(30))
            return df
        else:
            print("No tables found.")
            return None
    except Exception as e:
        print(f"Error scraping park factors: {e}")
        return None
    finally:
        driver.quit()

if __name__ == "__main__":
    scrape_park_factors()
