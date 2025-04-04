import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO
from datetime import datetime, timedelta
import time
from sqlalchemy import MetaData, Table, Column, Float, insert
from sqlalchemy.orm import Session
from backend.db import engine

end_date = datetime.today()
start_date = end_date - timedelta(days=30)
start = start_date.strftime("%Y-%m-%d")
end = end_date.strftime("%Y-%m-%d")

headers = {"User-Agent": "Mozilla/5.0"}
all_dfs = []
page = 1
max_pages = 10  # Maximum number of pages to scrape

while True:
    paged_url = f"https://www.fangraphs.com/leaders-legacy.aspx?pos=all&stats=sta&lg=all&qual=0&type=1&season=2025&month=1000&season1=2025&ind=0&team=0&rost=0&age=0&filter=&players=0&startdate={start}&enddate={end}&page={page}_50"
    print(f"Fetching page: {paged_url}")
    response = requests.get(paged_url, headers=headers)
    
    if response.status_code != 200:
        print(f"Error fetching page {page}: {response.status_code}")
        break
    
    time.sleep(1)
    soup = BeautifulSoup(response.text, "html.parser")
    tables = soup.find_all("table", class_="rgMasterTable")

    if not tables:
        break

    df_page = pd.read_html(StringIO(str(tables[0])))[0]
    if df_page.empty or df_page.shape[0] == 0:
        break

    all_dfs.append(df_page)
    page += 1

    if page > max_pages:
        print(f"Reached max_pages limit: {max_pages}.")
        break

if not all_dfs:
    print("⚠️  No tables found. Fangraphs may have temporarily blocked access or no data is available.")
    exit()

df = pd.concat(all_dfs, ignore_index=True)
df.columns = [col[1] if isinstance(col, tuple) else str(col).strip() for col in df.columns]

df = df[['Name', 'Team', 'K/9', 'BB/9', 'HR/9', 'ERA', 'FIP', 'xFIP', 'SIERA', 'WHIP']]
df = df.rename(columns={'Name': 'pitcher', 'Team': 'team', 'K/9': 'K_9', 'BB/9': 'BB_9', 'HR/9': 'HR_9'})

for col in df.columns:
    df[col] = df[col].astype(str).str.replace('%', '').str.replace(',', '').str.strip()

for col in ['K_9', 'BB_9', 'HR_9', 'ERA', 'FIP', 'xFIP', 'SIERA', 'WHIP']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(df.head())
print(f"Total rows fetched: {len(df)}")
print(df.columns.tolist())

metadata = MetaData()
metadata.reflect(bind=engine)
starting_pitchers = metadata.tables["starting_pitchers"]

with Session(engine) as session:
    session.execute(starting_pitchers.delete())
    session.execute(insert(starting_pitchers), df.to_dict(orient="records"))
    session.commit()
    print("Starting pitchers inserted into database.")
