from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Boolean, Date
from sqlalchemy import insert
from backend.db import engine
import pandas as pd

# Define the metadata
metadata = MetaData()

# Define the probable_starters table
probable_starters = Table('probable_starters', metadata,
    Column('id', Integer, primary_key=True),
    Column('game_id', Integer),
    Column('date', Date),
    Column('team', String),
    Column('pitcher_name', String),
    Column('handedness', String),
    Column('is_home', Boolean),
)

# Define the starting_pitchers table
starting_pitchers = Table('starting_pitchers', metadata,
    Column('pitcher', String, nullable=False),
    Column('team', String, nullable=False),
    Column('K_9', String),
    Column('BB_9', String),
    Column('HR_9', String),
    Column('ERA', String),
    Column('FIP', String),
    Column('xFIP', String),
    Column('SIERA', String),
    Column('WHIP', String),
)

# Define the bullpen table
bullpen = Table('bullpen', metadata,
    Column('team', String, nullable=False),
    Column('K_9', String),
    Column('BB_9', String),
    Column('HR_9', String),
    Column('ERA', String),
    Column('FIP', String),
    Column('xFIP', String),
    Column('SIERA', String),
    Column('WHIP', String),
)

# Define the team_batting_vs_lhp table
team_batting_vs_lhp = Table('team_batting_vs_lhp', metadata,
    Column('team', String, primary_key=True),
    Column('PA', String),
    Column('BB%', String),
    Column('K%', String),
    Column('BB/K', String),
    Column('SB', String),
    Column('OBP', String),
    Column('SLG', String),
    Column('OPS', String),
    Column('ISO', String),
    Column('Spd', String),
    Column('BABIP', String),
    Column('wRC', String),
    Column('wRAA', String),
    Column('wOBA', String),
    Column('wRC+', String),
)

# Define the team_batting_vs_rhp table
team_batting_vs_rhp = Table('team_batting_vs_rhp', metadata,
    Column('team', String, primary_key=True),
    Column('PA', String),
    Column('BB%', String),
    Column('K%', String),
    Column('BB/K', String),
    Column('SB', String),
    Column('OBP', String),
    Column('SLG', String),
    Column('OPS', String),
    Column('ISO', String),
    Column('Spd', String),
    Column('BABIP', String),
    Column('wRC', String),
    Column('wRAA', String),
    Column('wOBA', String),
    Column('wRC+', String),
)

# Define the odds table
odds = Table('odds', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('game_id', Integer),
    Column('global_game_id', Integer),
    Column('date', String),
    Column('team', String),
    Column('open', String),
    Column('bet365_ml', String),
    Column('total', String),
    Column('run_line', String),
    Column('total_raw', String),
    Column('run_line_raw', String),
)

# Define the park_factors table
park_factors = Table('park_factors', metadata,
    Column('rank', Integer),
    Column('team', String, primary_key=True),
    Column('venue', String),
    Column('year_range', String),
    Column('park_factor', Integer),
    Column('wobacon', Integer),
    Column('xwobacon', Integer),
    Column('bacon', Integer),
    Column('xbacon', Integer),
    Column('hardhit', Integer),
    Column('r', Integer),
    Column('obp', Integer),
    Column('h', Integer),
    Column('_1b', Integer),
    Column('_2b', Integer),
    Column('_3b', Integer),
    Column('hr', Integer),
    Column('bb', Integer),
    Column('so', Integer),
    Column('pa', Integer),
)

# Define the model_outputs table
model_outputs = Table('model_outputs', metadata,
    Column('game_id', Integer),
    Column('date', Date),
    Column('team', String),
    Column('starter', String),
    Column('expected_runs', Float),
    Column('win_prob', Float),
    Column('our_odds', Integer),
    Column('bet365_ml', String),
    Column('total', String),
    Column('run_line', Float),
    Column('run_line_raw', String),
    Column('run_line_odds', Float),
    Column('our_total', Float),
    Column('total_diff', Float),
    Column('total_play', String),
    Column('ev_flag', String),
    Column('run_line_ev_flag', String),
    Column('ml_confidence', Float),
    Column('run_line_confidence', Float),
    Column('high_variance_flag', String),
)

# Define the model_outputs_season table
model_outputs_season = Table('model_outputs_season', metadata,
    Column('game_id', Integer),
    Column('date', Date),
    Column('team', String),
    Column('starter', String),
    Column('expected_runs', Float),
    Column('win_prob', Float),
    Column('our_odds', Integer),
    Column('bet365_ml', String),
    Column('total', String),
    Column('run_line', Float),
    Column('run_line_raw', String),
    Column('run_line_odds', Float),
    Column('our_total', Float),
    Column('total_diff', Float),
    Column('total_play', String),
    Column('ev_flag', String),
    Column('run_line_ev_flag', String),
    Column('ml_confidence', Float),
    Column('run_line_confidence', Float),
    Column('high_variance_flag', String),
)

# Define the game_results table
game_results = Table('game_results', metadata,
    Column('game_id', Integer, primary_key=True, autoincrement=True),
    Column('date', String),
    Column('away_team', String),
    Column('away_score', Integer),
    Column('home_team', String),
    Column('home_score', Integer),
    Column('time', String),
)

# Define the runs_per_game table
runs_per_game = Table('runs_per_game', metadata,
    Column('rank', Integer),
    Column('team', String, primary_key=True),
    Column('_2025', Float),
    Column('last_3', Float),
    Column('last_1', Float),
    Column('home', Float),
    Column('away', Float),
    Column('_2024', Float),
)

# Create the table in the database
metadata.create_all(engine)
