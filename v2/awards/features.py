"""One candidate pool per award, league, and season, with only regular-season inputs.

A pool's features are what the model scores; the columns it keeps beside them
(name, team, the published stats) never enter a fit. Gold Glove prior wins read
seasons strictly before the board's season, so a board cannot see its own winners.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from v2.awards import data

FIRST_BALLOT_SEASON = 2010
FIRST_RELIEVER_SEASON = 2014
FIRST_GLOVE_SEASON = 2016
FIRST_PRIOR_GLOVE_SEASON = 2000
SHORT_SEASON = {2020: 162 / 60}
FIELD_POSITIONS = ("1B", "2B", "3B", "SS", "LF", "CF", "RF")
MIN_FIELD_INNINGS = 400.0
MIN_PITCHER_INNINGS = 100.0
MIN_MVP_WAR = 1.0
MIN_CY_INNINGS = 30.0
MIN_RELIEVER_GAMES = 30.0
MAX_RELIEVER_START_SHARE = 0.2
MIN_ROOKIE_WAR = 0.0
PITCHER_PA_LIMIT = 100.0

FEATURES = {
    "MVP": ["war", "wrc_plus", "hr", "rbi", "win_pct", "playoff", "is_pitcher"],
    "CY": ["war", "ra9_war", "era_diff", "k", "ip", "wins", "saves"],
    "ROY": ["war", "hr", "ops", "is_pitcher", "era_diff", "ip"],
    "RP": ["saves", "holds", "era_diff", "k", "ip", "war"],
    "GG_FIELD": ["oaa", "frv", "innings", "error_rate", "ops", "prior_wins"],
    "GG_C": ["framing", "blocking", "throwing", "innings", "error_rate", "ops", "prior_wins"],
    "GG_P": ["war", "innings", "error_rate", "prior_wins"],
}


def _numbers(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce") if column in frame else pd.Series(np.nan, index=frame.index)


def season_lines(season: int, league: str, pool: str = "ALL") -> pd.DataFrame:
    """Hitting and pitching joined per player; fWAR adds across the two for a two-way player."""
    hit = data.league_lines(season, league, "hitting", pool)
    pit = data.league_lines(season, league, "pitching", pool)
    scale = SHORT_SEASON.get(season, 1.0)
    lg_era = 27 * data.league_lines(season, league, "pitching").earnedRuns.sum() / data.league_lines(season, league, "pitching").outs.sum()
    h = pd.DataFrame({"player_id": hit.player_id, "player_name": hit.player_name, "team_id": hit.team_id, "team": hit.team,
                      "pa": hit.plateAppearances, "hr": hit.homeRuns * scale, "rbi": hit.rbi * scale,
                      "ops": _numbers(hit, "ops"), "avg": _numbers(hit, "avg"), "real_hr": hit.homeRuns, "real_rbi": hit.rbi})
    outs = pit.outs.clip(lower=1)
    p = pd.DataFrame({"player_id": pit.player_id, "p_name": pit.player_name, "p_team_id": pit.team_id, "p_team": pit.team,
                      "ip": pit.outs / 3 * scale, "real_ip": pit.outs / 3, "k": pit.strikeOuts * scale, "real_k": pit.strikeOuts,
                      "wins": pit.wins * scale, "real_wins": pit.wins, "saves": pit.saves * scale, "real_saves": pit.saves,
                      "holds": pit.holds * scale, "real_holds": pit.holds, "games_pitched": pit.gamesPitched,
                      "starts": pit.gamesStarted, "era": 27 * pit.earnedRuns / outs,
                      "era_diff": lg_era - 27 * pit.earnedRuns / outs})
    f = h.merge(p, on="player_id", how="outer")
    f["player_name"] = f.player_name.fillna(f.p_name)
    f["team_id"] = f.team_id.fillna(f.p_team_id)
    f["team"] = f.team.fillna(f.p_team)
    bat = data.league_war(season, league, "hitting")
    arm = data.league_war(season, league, "pitching")
    f = f.merge(bat[["player_id", "war", "wrc_plus"]].rename(columns={"war": "war_bat"}), on="player_id", how="left")
    f = f.merge(arm[["player_id", "war", "ra9_war"]].rename(columns={"war": "war_pit"}), on="player_id", how="left")
    for column in ("war_bat", "war_pit", "ra9_war", "wrc_plus"):
        f[column] = pd.to_numeric(f[column], errors="coerce")
    f["real_war"] = f.war_bat.fillna(0) + f.war_pit.fillna(0)
    f["war"] = f.real_war * scale
    f["ra9_war"] = f.ra9_war * scale
    f["is_pitcher"] = (f.pa.fillna(0) < PITCHER_PA_LIMIT).astype(float)
    f.loc[f.is_pitcher.eq(1.0), "wrc_plus"] = 100.0
    f = f.merge(data.team_context(season, league), on="team_id", how="left")
    if f.loc[f.war.gt(1), "win_pct"].isna().any():
        raise ValueError(f"Team context missing for a candidate in {league} {season}")
    defaults = {"pa": 0.0, "hr": 0.0, "rbi": 0.0, "real_hr": 0.0, "real_rbi": 0.0, "ops": 0.0, "avg": 0.0, "ip": 0.0,
                "real_ip": 0.0, "k": 0.0, "real_k": 0.0, "wins": 0.0, "real_wins": 0.0, "saves": 0.0, "real_saves": 0.0,
                "holds": 0.0, "real_holds": 0.0, "games_pitched": 0.0, "starts": 0.0, "era_diff": 0.0, "ra9_war": 0.0,
                "wrc_plus": 100.0, "win_pct": 0.5, "playoff": 0.0}
    return f.fillna(defaults).assign(season=season, league=league)


def ballot_pool(award: str, season: int, league: str) -> pd.DataFrame:
    if award == "MVP":
        f = season_lines(season, league)
        f = f[f.real_war * SHORT_SEASON.get(season, 1.0) >= MIN_MVP_WAR]
    elif award == "CY":
        f = season_lines(season, league)
        f = f[f.ip >= MIN_CY_INNINGS]
    elif award == "ROY":
        f = season_lines(season, league, "ROOKIES")
        f = f[f.war >= MIN_ROOKIE_WAR]
    elif award == "RP":
        f = season_lines(season, league)
        f = f[(f.games_pitched >= MIN_RELIEVER_GAMES / SHORT_SEASON.get(season, 1.0))
              & (f.starts <= MAX_RELIEVER_START_SHARE * f.games_pitched)]
    else:
        raise ValueError(f"Unknown ballot award {award}")
    if f.empty:
        raise ValueError(f"Empty {award} pool for {league} {season}")
    return f.reset_index(drop=True)


def ballot_targets(award: str, pool: pd.DataFrame, season: int, league: str) -> np.ndarray:
    """Vote share per candidate, or the winner alone where no votes are published."""
    if award == "RP":
        winner = data.recipients(f"{league}REL", season).player_id
        y = pool.player_id.isin(winner).to_numpy(float)
    else:
        votes = data.ballots(season)
        votes = votes[votes.league.eq(league) & votes.award.eq(award)].dropna(subset=["player_id"])
        y = pool.player_id.map(votes.set_index("player_id").share).fillna(0.0).to_numpy(float)
    if y.sum() <= 0:
        raise ValueError(f"No {award} votes landed in the {league} {season} pool")
    return y / y.sum()


def ballot_winner(award: str, season: int, league: str) -> pd.Series:
    code = {"MVP": f"{league}MVP", "CY": f"{league}CY", "ROY": f"{league}ROY", "RP": f"{league}REL"}[award]
    return data.recipients(code, season).player_id


def prior_glove_wins(season: int, league: str) -> pd.Series:
    """Career Gold Gloves before this season, in either league."""
    wins = pd.concat([data.recipients(f"{code}GG", year).player_id
                      for year in range(FIRST_PRIOR_GLOVE_SEASON, season) for code in data.LEAGUES])
    return wins.value_counts()


def glove_positions(season: int, league: str) -> dict[int, str]:
    """Which position each winner won at. A recipient row names the player's primary position
    today, so past winners are placed by innings that season, one winner per position."""
    winners = data.recipients(f"{league}GG", season)
    utility = set(winners.player_id[winners.position.eq("UT")]) if season >= 2022 else set()
    field = winners[~winners.player_id.isin(utility)].reset_index(drop=True)
    positions = list(data.POSITION_CODES)
    cost = np.zeros((len(field), len(positions)))
    for j, position in enumerate(positions):
        innings = data.fielding_lines(season, league, position).set_index("player_id").innings
        cost[:, j] = -field.player_id.map(innings).fillna(0.0).to_numpy()
    rows, cols = linear_sum_assignment(cost)
    placed = {int(field.player_id[i]): positions[j] for i, j in zip(rows, cols) if cost[i, j] < 0}
    return placed | {int(p): "UT" for p in utility}


def _zscore(values: pd.Series) -> pd.Series:
    deviation = values.std(ddof=0)
    return (values - values.mean()) / deviation if deviation > 0 else values * 0.0


def glove_pool(season: int, league: str, position: str) -> pd.DataFrame:
    """Qualified fielders at one position, each input standardized within the pool so positions share a scale."""
    field = data.fielding_lines(season, league, position)
    lines = season_lines(season, league)[["player_id", "ops", "war", "team", "real_war"]]
    scale = SHORT_SEASON.get(season, 1.0)
    field = field[["player_id", "player_name", "team", "innings", "errors", "chances"]].merge(
        lines.drop(columns="team"), on="player_id", how="left", validate="one_to_one")
    minimum = MIN_PITCHER_INNINGS if position == "P" else MIN_FIELD_INNINGS
    pool = field[field.innings * scale >= minimum].copy().reset_index(drop=True)
    if pool.empty:
        raise ValueError(f"No qualified {position} in {league} {season}")
    pool["error_rate"] = pool.errors / pool.innings
    pool["prior_wins"] = np.log1p(pool.player_id.map(prior_glove_wins(season, league)).fillna(0.0))
    pool["ops"] = pool.ops.fillna(0.0)
    pool["war"] = pool.war.fillna(0.0)
    if position == "C":
        pool = pool.merge(data.catcher_runs(season), on="player_id", how="left").fillna({"framing": 0.0, "blocking": 0.0, "throwing": 0.0})
    elif position != "P":
        pool = pool.merge(data.outs_above_average(season, position), on="player_id", how="left").fillna({"oaa": 0.0, "frv": 0.0})
    pool = pool.assign(season=season, league=league, position=position)
    names = FEATURES["GG_C" if position == "C" else "GG_P" if position == "P" else "GG_FIELD"]
    for name in names:
        pool[f"raw_{name}"] = pool[name]
        pool[name] = _zscore(pool[name])
    return pool


def utility_pool(season: int, league: str, finalists: list[int]) -> pd.DataFrame:
    """Finalists' fielding summed across positions, on the scale of every qualified fielder that season."""
    pools = [glove_pool(season, league, position) for position in FIELD_POSITIONS]
    everyone = pd.concat(pools, ignore_index=True)
    rows = []
    for player_id in finalists:
        parts = []
        for position in FIELD_POSITIONS:
            lines = data.fielding_lines(season, league, position)
            line = lines[lines.player_id.eq(player_id)]
            if line.empty:
                continue
            value = data.outs_above_average(season, position)
            value = value[value.player_id.eq(player_id)]
            parts.append({"name": line.player_name.iloc[0], "team": line.team.iloc[0], "innings": float(line.innings.iloc[0]),
                          "errors": float(line.errors.iloc[0]), "oaa": float(value.oaa.sum()), "frv": float(value.frv.sum())})
        if not parts:
            raise ValueError(f"Utility finalist {player_id} has no fielding line in {league} {season}")
        total = pd.DataFrame(parts)
        rows.append({"player_id": player_id, "player_name": total.name.iloc[0], "team": total.team.iloc[-1],
                     "raw_oaa": total.oaa.sum(), "raw_frv": total.frv.sum(), "raw_innings": total.innings.sum(),
                     "raw_error_rate": total.errors.sum() / total.innings.sum()})
    pool = pd.DataFrame(rows).assign(season=season, league=league, position="UT")
    lines = season_lines(season, league)[["player_id", "ops", "real_war"]]
    pool = pool.merge(lines, on="player_id", how="left")
    pool["raw_ops"] = pool.ops.fillna(0.0)
    pool["raw_prior_wins"] = np.log1p(pool.player_id.map(prior_glove_wins(season, league)).fillna(0.0))
    for name in FEATURES["GG_FIELD"]:
        deviation = everyone[f"raw_{name}"].std(ddof=0)
        pool[name] = pool[f"raw_{name}"] / deviation
    return pool
