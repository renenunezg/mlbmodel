import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO
import time
from sqlalchemy import MetaData, insert
from sqlalchemy.orm import Session
from backend.db import engine
from datetime import datetime

today = datetime.today()
year = today.year
url = f"https://www.fangraphs.com/leaders-legacy.aspx?pos=all&stats=rel&lg=all&qual=10&type=1&season={year}&month=2&season1={year}&ind=0&team=0,ts&rost=0&age=0&filter=&players=0"

headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)
time.sleep(1)
soup = BeautifulSoup(response.text, "html.parser")
tables = soup.find_all("table", class_="rgMasterTable")

if not tables:
    print("⚠️  No tables found. Fangraphs may have temporarily blocked access. Try again in a minute.")
    exit()

df = pd.read_html(StringIO(str(tables[0])))[0]
df.columns = [col[1] if isinstance(col, tuple) else str(col).strip() for col in df.columns]

df = df[['Team', 'K/9', 'BB/9', 'HR/9', 'ERA', 'FIP', 'xFIP', 'SIERA', 'WHIP']]
df = df.rename(columns={
    'Team': 'team',
    'K/9': 'K_9',
    'BB/9': 'BB_9',
    'HR/9': 'HR_9'
})

for col in df.columns:
    df[col] = df[col].astype(str).str.replace('%', '').str.replace(',', '').str.strip()

for col in ['K_9', 'BB_9', 'HR_9', 'ERA', 'FIP', 'xFIP', 'SIERA', 'WHIP']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(df.head())
print(df.columns.tolist())

metadata = MetaData()
metadata.reflect(bind=engine)
bullpen = metadata.tables["bullpen"]

with Session(engine) as session:
    session.execute(bullpen.delete())
    session.execute(insert(bullpen), df.to_dict(orient="records"))
    session.commit()
    print("Bullpen data inserted into database.")
