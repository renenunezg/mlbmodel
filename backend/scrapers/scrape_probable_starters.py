import pandas as pd
from sqlalchemy import insert, MetaData
from backend.db import engine
from sqlalchemy.orm import Session
from backend.team_mappings import TEAM_NAME_MAP

# Scrape
url = "https://www.cbssports.com/fantasy/baseball/probable-pitchers/"
tables = pd.read_html(url)

rows = []
game_id = 1

for table in tables:
    for i, (_, row) in enumerate(table.iterrows()):
        try:
            player_col = row["players"]
            parts = player_col.split("  ")
            if len(parts) >= 3:
                short_name, team_abbr, handedness = parts[0:3]
                full_name = parts[3] if len(parts) > 3 else short_name
                rows.append({
                    "game_id": game_id,
                    "team": TEAM_NAME_MAP.get(team_abbr.strip(), team_abbr.strip()),
                    "pitcher_name": full_name.strip(),
                    "handedness": handedness.strip().replace("HP", ""),
                    "is_home": (i % 2 == 1)
                })
                if i % 2 == 1:
                    game_id += 1
        except Exception as e:
            print(f"Skipping row due to error: {e}")

df = pd.DataFrame(rows)
print(df)

# Insert into DB
metadata = MetaData()
metadata.reflect(bind=engine)
probable_starters = metadata.tables["probable_starters"]

with Session(engine) as session:
    session.execute(probable_starters.delete())  # Clear previous entries
    session.execute(insert(probable_starters), df.to_dict(orient="records"))
    session.commit()
    print("Probable starters inserted into database.")