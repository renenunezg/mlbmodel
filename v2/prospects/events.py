"""Plate appearances and player-season-level aggregates from the cached feeds.

A play is a plate appearance when its event ends the batter's turn (``PA_EVENTS``); runner events
such as steals and pickoffs are kept in the cache but do not count. Each plate appearance maps to
the eight-outcome layout the player model uses (``v2.data.pa_dataset.OUTCOMES``), and its pitches
contribute call-code counts (balls, called strikes, swinging strikes, fouls, balls in play). Many
games log only the pitch that ended each plate appearance (all of the lower levels before 2016,
rookie ball before 2025); their call-code counts are missing, and ``seq_pa`` counts the plate
appearances from fully logged games. Stringer trajectory and hardness describe the batted ball; the
tracked fields (exit velocity, launch angle, pitch speed) are present only in tracked games and
are aggregated with their own denominators so an untracked player reads as missing, never zero.

Levels are labels derived from sport and league: ``aaa``, ``aa``, ``a+``, ``a``, ``a-`` (short-season,
through 2020), ``rk+`` (Pioneer and Appalachian, through 2020), ``rk`` (complex leagues), ``dsl``
(Dominican and Venezuelan Summer Leagues) and ``afl``.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from v2.data.pa_dataset import OUTCOMES
from v2.prospects import feeds

log = logging.getLogger(__name__)

LEVEL_ORDER = ["dsl", "rk", "rk+", "a-", "a", "a+", "aa", "aaa", "afl"]
LEVEL_RANK = {lvl: i for i, lvl in enumerate(LEVEL_ORDER)}
SPORT_LEVEL = {11: "aaa", 12: "aa", 13: "a+", 14: "a", 15: "a-", 17: "afl"}
ROOKIE_LEAGUE_LEVEL = {121: "rk", 124: "rk", 130: "dsl", 134: "dsl", 128: "rk+", 120: "rk+"}
PA_EVENTS = {
    "strikeout": "K", "strikeout_double_play": "K", "strikeout_triple_play": "K",
    "walk": "BB", "intent_walk": "BB", "hit_by_pitch": "HBP",
    "single": "1B", "double": "2B", "triple": "3B", "home_run": "HR",
    "field_out": "OUT", "force_out": "OUT", "grounded_into_double_play": "OUT", "double_play": "OUT",
    "triple_play": "OUT", "fielders_choice": "OUT", "fielders_choice_out": "OUT", "sac_fly": "OUT",
    "sac_fly_double_play": "OUT", "sac_bunt": "OUT", "sac_bunt_double_play": "OUT", "field_error": "OUT",
    "catcher_interf": "OUT", "batter_interference": "OUT", "fan_interference": "OUT", "other_out": "OUT",
}
TRAJECTORY = {"ground_ball": "gb", "bunt_grounder": "gb", "line_drive": "ld", "fly_ball": "fb", "popup": "pu", "bunt_popup": "pu"}
HARDNESS = {"soft": "soft", "medium": "medium", "hard": "hard"}
EV90_MIN_BALLS = 15  # batted balls needed before a 90th percentile exit velocity is reported
POSITION_CODES = ("C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH")
# Games average either about 1.7 pitches per plate appearance (only the deciding pitch logged) or 3.5 and up;
# almost none fall between 2.5 and 3.0, and a fully logged game below 2.75 is vanishingly rare.
SEQUENCED_PITCHES_PER_PA = 2.75
PITCH_COUNT_COLUMNS = ("n_pitches", "balls", "called_strikes", "swinging_strikes", "fouls", "in_play", "swings", "contacts", "strikes_seen")
PITCH_SHAPE_COLUMNS = ("tracked_pitches", "fb_velo", "fb_velo_max", "fb_ivb", "fb_hb", "fb_spin", "extension", "whiff_tracked")


def level_of(games: pd.DataFrame) -> pd.Series:
    """Level label per game row from its sport and league ids."""
    level = games["sport_id"].map(SPORT_LEVEL)
    rookie = games["sport_id"].eq(16)
    level = level.where(~rookie, games["league_id"].map(ROOKIE_LEAGUE_LEVEL))
    return level


def load_games(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        for sport_id in feeds.SPORT_IDS:
            g = feeds.read_kind(sport_id, season, "games")
            if not g.empty:
                frames.append(g)
    if not frames:
        return pd.DataFrame()
    games = pd.concat(frames, ignore_index=True)
    games = games[games["n_plays"].fillna(0) > 0].copy()
    games["level"] = level_of(games)
    games = games[games["level"].notna()]
    games["date"] = pd.to_datetime(games["date"])
    return games


def _load(kind: str, seasons: list[int], games: pd.DataFrame) -> pd.DataFrame:
    frames = []
    keep = games[["game_pk", "season", "level", "date", "venue_id", "home_team_id", "away_team_id"]]
    for season in seasons:
        for sport_id in feeds.SPORT_IDS:
            f = feeds.read_kind(sport_id, season, kind)
            if not f.empty:
                frames.append(f.merge(keep, on="game_pk", how="inner"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plate_appearances(seasons: list[int], games: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per plate appearance with its outcome, pitch counts, batted ball and tracked fields."""
    games = load_games(seasons) if games is None else games
    plays = _load("plays", seasons, games)
    if plays.empty:
        return plays
    pa = plays[plays["event_type"].isin(PA_EVENTS)].copy()
    pa["outcome"] = pa["event_type"].map(PA_EVENTS)
    pa["bb_type"] = pa["trajectory"].map(TRAJECTORY)
    pa["bb_hard"] = pa["hardness"].map(HARDNESS)
    pa["in_play_ball"] = pa["outcome"].isin(["1B", "2B", "3B", "OUT"]) & pa["bb_type"].notna()
    pa["swings"] = pa["swinging_strikes"] + pa["fouls"] + pa["in_play"]
    pa["contacts"] = pa["fouls"] + pa["in_play"]
    pa["strikes_seen"] = pa["called_strikes"] + pa["swings"]
    pa["sequenced"] = pa.groupby("game_pk")["n_pitches"].transform("mean") >= SEQUENCED_PITCHES_PER_PA
    pa.loc[~pa["sequenced"], list(PITCH_COUNT_COLUMNS)] = np.nan
    return pa


def appearances(seasons: list[int], games: pd.DataFrame | None = None) -> pd.DataFrame:
    games = load_games(seasons) if games is None else games
    app = _load("appearances", seasons, games)
    if app.empty:
        return app
    # The boxscore team block has no parent organization; the game row does.
    orgs = games.set_index("game_pk")[["home_org_id", "away_org_id"]]
    side = app["home"].map({True: "home_org_id", False: "away_org_id"})
    app["org_id"] = pd.Series(orgs.to_numpy()[orgs.index.get_indexer(app["game_pk"]), (side == "away_org_id").astype(int)],
                              index=app.index)
    app["org_id"] = pd.to_numeric(app["org_id"], errors="coerce").astype("Int64")
    return app


def _ev90(values: pd.Series) -> float:
    v = values.dropna()
    return float(np.percentile(v, 90)) if len(v) >= EV90_MIN_BALLS else np.nan


def hitter_season_levels(pa: pd.DataFrame, app: pd.DataFrame) -> pd.DataFrame:
    """One row per batter, season and level: outcome counts, pitch-level counts, batted-ball mix, tracked overlay."""
    g = pa.groupby(["batter", "season", "level"])
    out = pd.DataFrame({"pa": g.size()})
    for o in OUTCOMES:
        out[o] = g["outcome"].agg(lambda s, o=o: int((s == o).sum()))
    out["seq_pa"] = g["sequenced"].sum()
    for col in PITCH_COUNT_COLUMNS:
        out[col] = g[col].sum()
    bip = pa[pa["in_play_ball"]]
    gb = bip.groupby(["batter", "season", "level"])
    out["bip"] = gb.size()
    for t in ("gb", "ld", "fb", "pu"):
        out[f"bb_{t}"] = gb["bb_type"].agg(lambda s, t=t: int((s == t).sum()))
    for h in ("soft", "medium", "hard"):
        out[f"bb_{h}"] = gb["bb_hard"].agg(lambda s, h=h: int((s == h).sum()))
    # Tracked overlay: denominators are tracked opportunities, never all opportunities.
    out["tracked_pa"] = g["tracked"].sum()
    tracked = bip[bip["launch_speed"].notna()]
    gt = tracked.groupby(["batter", "season", "level"])
    out["tracked_bip"] = gt.size()
    out["ev_mean"] = gt["launch_speed"].mean()
    out["ev90"] = gt["launch_speed"].agg(_ev90)
    out["hard_hit"] = gt["launch_speed"].agg(lambda s: float((s >= 95).mean()))
    out["la_mean"] = gt["launch_angle"].mean()
    out = out.fillna({c: 0 for c in out.columns if c.startswith("bb_") or c in ("bip", "tracked_pa", "tracked_bip")})
    out = out.reset_index().rename(columns={"batter": "player_id"})
    if not app.empty:
        a = app.groupby(["player_id", "season", "level"]).agg(games=("game_pk", "nunique"), sb=("sb", "sum"), cs=("cs", "sum"),
                                                              org_id=("org_id", "last"), team_id=("team_id", "last"))
        pos = _position_shares(app)
        out = out.merge(a.reset_index(), on=["player_id", "season", "level"], how="left").merge(pos, on=["player_id", "season", "level"], how="left")
    out["role"] = "bat"
    return out


def _position_shares(app: pd.DataFrame) -> pd.DataFrame:
    """Share of a player's games at each position in a season and level (a game can count at several)."""
    rows = app[app["positions"].fillna("").str.len() > 0][["player_id", "season", "level", "game_pk", "positions"]].copy()
    rows["positions"] = rows["positions"].str.split(",")
    long = rows.explode("positions")
    long = long[long["positions"].isin(POSITION_CODES)]
    counts = long.groupby(["player_id", "season", "level", "positions"])["game_pk"].nunique().unstack(fill_value=0)
    games = app.groupby(["player_id", "season", "level"])["game_pk"].nunique()
    shares = counts.div(games.reindex(counts.index), axis=0)
    shares.columns = [f"pos_{c}" for c in shares.columns]
    shares = shares.reindex(columns=[f"pos_{c}" for c in POSITION_CODES], fill_value=0.0)
    return shares.reset_index()


def pitcher_season_levels(pa: pd.DataFrame, app: pd.DataFrame, pitches: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per pitcher, season and level: batters faced, outcomes allowed, pitch counts, batted balls, role, tracked overlay."""
    g = pa.groupby(["pitcher", "season", "level"])
    out = pd.DataFrame({"bf": g.size()})
    for o in OUTCOMES:
        out[o] = g["outcome"].agg(lambda s, o=o: int((s == o).sum()))
    out["seq_pa"] = g["sequenced"].sum()
    for col in PITCH_COUNT_COLUMNS:
        out[col] = g[col].sum()
    bip = pa[pa["in_play_ball"]]
    gb = bip.groupby(["pitcher", "season", "level"])
    out["bip"] = gb.size()
    for t in ("gb", "ld", "fb", "pu"):
        out[f"bb_{t}"] = gb["bb_type"].agg(lambda s, t=t: int((s == t).sum()))
    for h in ("soft", "medium", "hard"):
        out[f"bb_{h}"] = gb["bb_hard"].agg(lambda s, h=h: int((s == h).sum()))
    out["tracked_pa"] = g["tracked"].sum()
    tracked = bip[bip["launch_speed"].notna()]
    gt = tracked.groupby(["pitcher", "season", "level"])
    out["tracked_bip"] = gt.size()
    out["ev_mean_allowed"] = gt["launch_speed"].mean()
    out["hard_hit_allowed"] = gt["launch_speed"].agg(lambda s: float((s >= 95).mean()))
    out = out.fillna({c: 0 for c in out.columns if c.startswith("bb_") or c in ("bip", "tracked_pa", "tracked_bip")})
    out = out.reset_index().rename(columns={"pitcher": "player_id"})
    if not app.empty:
        p = app[app["pitched"]].groupby(["player_id", "season", "level"]).agg(
            games=("game_pk", "nunique"), starts=("games_started", "sum"), outs=("outs", "sum"),
            org_id=("org_id", "last"), team_id=("team_id", "last"))
        out = out.merge(p.reset_index(), on=["player_id", "season", "level"], how="left")
    if pitches is not None and not pitches.empty:
        out = out.merge(pitch_shapes(pitches), on=["player_id", "season", "level"], how="left")
    for c in PITCH_SHAPE_COLUMNS:
        if c not in out:
            out[c] = np.nan
    out["tracked_pitches"] = out["tracked_pitches"].fillna(0.0)
    out["role"] = "pit"
    return out


def pitch_shapes(pitches: pd.DataFrame) -> pd.DataFrame:
    """Fastball velocity and shape per pitcher, season and level from tracked pitches, with the tracked pitch count."""
    pitches = pitches.copy()
    for c in ("start_speed", "ivb", "hb", "spin_rate", "extension"):
        pitches[c] = pd.to_numeric(pitches[c], errors="coerce")
    fb = pitches[pitches["pitch_type"].isin(["FF", "SI", "FC", "FT", "FA"]) & pitches["start_speed"].notna()].copy()
    fb["hb"] = fb["hb"].abs()
    g = fb.groupby(["pitcher", "season", "level"])
    out = pd.DataFrame({"tracked_pitches": pitches.groupby(["pitcher", "season", "level"]).size()})
    out["fb_velo"] = g["start_speed"].mean()
    out["fb_velo_max"] = g["start_speed"].quantile(0.95)
    out["fb_ivb"] = g["ivb"].mean()
    out["fb_hb"] = g["hb"].mean()
    out["fb_spin"] = g["spin_rate"].mean()
    out["extension"] = g["extension"].mean()
    sw = pitches[pitches["code"].isin(list(feeds.SWINGING_STRIKE_CODES | feeds.FOUL_CODES | feeds.IN_PLAY_CODES))]
    out["whiff_tracked"] = (pitches[pitches["code"].isin(list(feeds.SWINGING_STRIKE_CODES))].groupby(["pitcher", "season", "level"]).size()
                            / sw.groupby(["pitcher", "season", "level"]).size())
    return out.reset_index().rename(columns={"pitcher": "player_id"})


def season_tables(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hitter and pitcher season-level tables for the given seasons from the feed caches."""
    games = load_games(seasons)
    if games.empty:
        raise FileNotFoundError(f"No cached feeds for seasons {seasons}")
    pa = plate_appearances(seasons, games)
    app = appearances(seasons, games)
    pitches = _load("pitches", seasons, games)
    hitters = hitter_season_levels(pa, app)
    pitchers = pitcher_season_levels(pa, app, pitches if not pitches.empty else None)
    log.info(f"events[{seasons}]: {len(games):,} games, {len(pa):,} plate appearances, "
             f"{len(hitters):,} hitter and {len(pitchers):,} pitcher season-levels")
    return hitters, pitchers
