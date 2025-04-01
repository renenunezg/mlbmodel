import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO
import os

url = "https://www.fangraphs.com/leaders/major-league?pos=all&stats=bat&lg=all&qual=0&season=2025&season1=2025&ind=0&team=0%2Cts&rost=&filter=&players=0&type=8&month=13"

headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)

if response.status_code != 200:
    print(f"❌ Request failed with status code {response.status_code}")
    exit()

soup = BeautifulSoup(response.text, "html.parser")
tables = soup.find_all("table")

if not tables:
    print("⚠️  No valid stats table found.")
    exit()

df = pd.read_html(StringIO(str(tables[-1])))[0]

print(df.head())
print(df.columns.tolist())
