import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import os

st.set_page_config(page_title="MLB Model Dashboard", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    [data-testid="stDataFrame"] {
        border: 2px solid #198754 !important; /* Masters green */
        border-radius: 6px;
        background-color: #1e1e1e !important;
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# Connect to the database
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/mlbmodel")
engine = create_engine(DATABASE_URL)

st.title("⚾ MLB Model Predictions")

# Load model output
@st.cache_data
def load_predictions():
    try:
        df = pd.read_sql("SELECT * FROM model_outputs ORDER BY date DESC", con=engine)
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%b %d, %Y')  # e.g. "Apr 09, 2025"
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

df = load_predictions()
st.write("Columns in DataFrame:", df.columns.tolist())

if df.empty:
    st.warning("No predictions available.")
else:
    column_config = {
        "date": st.column_config.TextColumn("Date", help="Game date"),
        "team": st.column_config.TextColumn("Tm", help="Team abbreviation"),
        "starter": st.column_config.TextColumn("SP", help="Starting pitcher"),
        "expected_runs": st.column_config.NumberColumn("xR", help="Expected runs from model"),
        "our_total": st.column_config.NumberColumn("Tot", help="Projected total runs"),
        "total_diff": st.column_config.NumberColumn("ΔTot", help="Difference from sportsbook total"),
        "total_play_flag": st.column_config.TextColumn("TotBet", help="Model total bet (Over/Under)"),
        "our_odds": st.column_config.NumberColumn("ModelOdds", help="Implied odds from model"),
        "bet365_ml": st.column_config.NumberColumn("B365", help="Bet365 moneyline"),
        "ev_flag": st.column_config.TextColumn("+EV?", help="Positive expected value flag"),
        "run_line_ev_flag": st.column_config.TextColumn("RL +EV?", help="Run line +EV flag"),
        "ml_confidence": st.column_config.NumberColumn("ML Conf", help="Confidence in moneyline pick"),
        "run_line_confidence": st.column_config.NumberColumn("RL Conf", help="Confidence in run line pick"),
        "high_variance_flag": st.column_config.TextColumn("HiVar?", help="Flag for high variance games")
    }
    
    grouped = df.groupby("game_id")
    for game_id, group in grouped:
        if len(group) == 2:
            away_team = group.iloc[0]['team']
            home_team = group.iloc[1]['team']
            date = group.iloc[0]['date']
            st.markdown(f"### {away_team} @ {home_team} – {date}")
            st.dataframe(group, use_container_width=True, column_config=column_config)