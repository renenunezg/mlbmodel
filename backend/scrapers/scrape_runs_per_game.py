import pandas as pd
from backend.team_mappings import TEAM_NAME_MAP
from backend.db import engine  # Add this import at the top

URL = "https://www.teamrankings.com/mlb/stat/runs-per-game"  # Adjust if it's a different URL

def main():
    tables = pd.read_html(URL)
    print(f"Found {len(tables)} tables.")
    df = tables[0]
    df.columns = [col.lower() for col in df.columns]
    df = df.rename(columns={"2025": "_2025", "2024": "_2024"})
    df["team"] = df["team"].map(TEAM_NAME_MAP)
    df = df.dropna(subset=["team"])
    df = df.replace("--", 0)
    print(df)
    
    df.to_sql("runs_per_game", con=engine, if_exists="replace", index=False)
    print("Inserted into runs_per_game table.")

if __name__ == "__main__":
    main()
