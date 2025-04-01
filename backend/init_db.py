from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
from sqlalchemy import insert
from backend.db import engine
import pandas as pd

# Define the metadata
metadata = MetaData()

# Define the probable_starters table
probable_starters = Table('probable_starters', metadata,
    Column('id', Integer, primary_key=True),
    Column('game_id', Integer),
    Column('team_abbr', String),
    Column('pitcher_name', String),
    Column('handedness', String),
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

# Create the table in the database
metadata.create_all(engine)
