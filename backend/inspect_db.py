import os
import pandas as pd
from sqlalchemy import create_engine, MetaData
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
metadata = MetaData()
metadata.reflect(bind=engine)

for table_name, table in metadata.tables.items():
    print(f"\n--- {table_name} ---")
    try:
        df = pd.read_sql_table(table_name, con=engine)
        print(df.head(10))
    except Exception as e:
        print(f"Error reading {table_name}: {e}")
