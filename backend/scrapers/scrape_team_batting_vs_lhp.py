import pandas as pd
import time
from io import StringIO
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import os

# Set up headless Chrome
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
prefs = {"profile.managed_default_content_settings.images": 2}
options.add_experimental_option("prefs", prefs)
options.add_argument("--disable-extensions")

driver = webdriver.Chrome(options=options)

url = "https://www.fangraphs.com/leaders/major-league?pos=all&stats=bat&lg=all&qual=0&ind=0&team=0%2Cts&rost=&filter=&players=0&type=1&month=13&startdate=&enddate=&season1=2025&season=2025"
driver.get(url)

try:
    # Wait for the stats table to load
    WebDriverWait(driver, 3).until(
        EC.presence_of_element_located((By.TAG_NAME, "table"))
    )

    # Extract all tables in the page
    html = driver.page_source
    tables = pd.read_html(StringIO(html))

    # FanGraphs typically puts the data in the last table
    df = tables[-1]

    # Clean and normalize column names
    def clean_column(col):
        if not isinstance(col, str) or col.strip() == "":
            return None
        col = col.split(" - ")[0].split("—")[0].split("--")[0].strip()
        if len(col) >= 4 and col[:len(col)//2] == col[len(col)//2:]:
            col = col[:len(col)//2]  # handle BB%BB% → BB%
        return col

    df.columns = [clean_column(col) for col in df.columns]
    df = df.loc[:, df.columns.notnull()]

    desired_columns = ['Team', 'PA', 'BB%', 'K%', 'BB/K', 'SB', 'OBP', 'SLG', 'OPS', 'ISO', 'Spd', 'BABIP', 'wRC', 'wRAA', 'wOBA', 'wRC+']
    df = df[[col for col in desired_columns if col in df.columns]]
        # Rename the 'Team' column to lowercase 'team'
    df = df.rename(columns={"Team": "team"})
    
    # Load environment variables for DB connection
    from dotenv import load_dotenv
    load_dotenv()

    DB_URL = os.getenv("DATABASE_URL")
    engine = create_engine(DB_URL)

    try:
        df.to_sql("team_batting_vs_lhp", engine, if_exists="replace", index=False)
        print("✅ Data inserted into team_batting_vs_lhp table.")
    except SQLAlchemyError as e:
        print("❌ Error inserting data into DB:", e)

    print(df.head())
    print(df.columns.tolist())

except Exception as e:
    print("⚠️ Error while scraping with Selenium:", e)

finally:
    driver.quit()
