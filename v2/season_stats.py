"""Season stat lines beneath the player and prospect grades: one packed row per published player.

The population is whoever the site shows: ``player_values`` (pool ``mlb``) and ``prospect_values`` (pool
``prospect``), each player as a hitter or a pitcher by his card's position group. The season is the latest one
with regular-season games at that pool's levels.

Production comes from the MLB Stats API season lines (official counts; every rate is recomputed from them, so a
prospect's line across several levels pools exactly). MLB players add fWAR and FIP from the same API's
sabermetrics block; a prospect's FIP uses each league's own constant. Underlying skills come from Statcast for
MLB and from the prospect feed caches for the minors (tracked games only for exit velocity and fastball velocity;
pitch calls only from fully logged games). The Arizona Fall League is never part of a line.

``stats`` is a smallint array in a fixed order per role, scaled to integers (see BAT and PIT); a null is a stat
that does not exist for this player (no tracked sample, no fWAR in the minors). Rates the site can derive exactly
(K% and BB% from K, BB and PA or BF) are not stored.

Usage:
    python -m v2.season_stats [--season 2026] [--no-write]
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import requests
from sqlalchemy import text

from backend.data.game_types import regular_season_pitches
from backend.db import engine
from backend.log import setup_logging
from v2.players.inputs import HARD_HIT, IN_ZONE_MAX, STATCAST_DIR, SWING, WHIFF
from v2.prospects import events, feeds

log = logging.getLogger(__name__)

STATS_URL = "https://statsapi.mlb.com/api/v1/stats"
MINOR_SPORT_IDS = (11, 12, 13, 14, 16)  # AAA through rookie; 15 (short-season A) ended in 2020, 17 is the AFL
FASTBALLS = {"FF", "SI", "FA", "FT"}
PITCHER_GROUPS = {"SP", "RP"}
EV90_MIN_BALLS = events.EV90_MIN_BALLS

# (name, scale): stored value = round(value * scale). Rates in [0, 1] are stored in tenths of a percent.
BAT = (("pa", 1), ("war", 10), ("avg", 1000), ("obp", 1000), ("slg", 1000), ("ops", 1000), ("hr", 1), ("rbi", 1),
       ("sb", 1), ("k", 1), ("bb", 1), ("xwoba", 1000), ("barrel", 1000), ("hard_hit", 1000), ("ev", 10), ("la", 10),
       ("chase", 1000), ("ev90", 10), ("bbe", 1))
PIT = (("bf", 1), ("outs", 1), ("war", 10), ("era", 100), ("fip", 100), ("whip", 100), ("k", 1), ("bb", 1), ("hr", 1),
       ("whiff", 1000), ("fb_velo", 10), ("gb", 1000), ("xwoba", 1000), ("hard_hit", 1000), ("bbe", 1), ("pitches", 1))
SMALLINT_MAX = 32767
MIN_ROWS = 2000
BATCH_ROWS = 2000


def _get(params: dict) -> dict:
    response = requests.get(STATS_URL, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def official_lines(season: int, group: str, sport_ids: tuple[int, ...]) -> pd.DataFrame:
    """One row per player and sport (the API pools a player's teams within a sport), with the league for constants."""
    rows = []
    for sport_id in sport_ids:
        block, = _get({"stats": "season", "group": group, "season": season, "sportIds": sport_id,
                       "playerPool": "ALL", "limit": 10000, "gameType": "R"})["stats"]
        if len(block["splits"]) != block["totalSplits"]:
            raise ValueError(f"Truncated {group} lines for sport {sport_id} in {season}")
        rows += [{**s["stat"], "player_id": s["player"]["id"], "sport_id": sport_id,
                  "league_id": (s.get("league") or {}).get("id")} for s in block["splits"]]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    if frame.duplicated(["player_id", "sport_id"]).any():
        raise ValueError(f"Duplicate {group} lines in {season}")
    return frame


def sabermetrics(season: int) -> pd.DataFrame:
    """MLB fWAR for hitters and pitchers and FIP for pitchers, as MLB distributes FanGraphs' values."""
    parts = []
    for block in _get({"stats": "sabermetrics", "group": "hitting,pitching", "season": season, "sportIds": 1,
                       "playerPool": "ALL", "limit": 10000, "gameType": "R"})["stats"]:
        if len(block["splits"]) != block["totalSplits"]:
            raise ValueError(f"Truncated sabermetrics in {season}")
        role = "bat" if block["group"]["displayName"] == "hitting" else "pit"
        parts.append(pd.DataFrame([{"player_id": s["player"]["id"], "role": role, "war": s["stat"].get("war"),
                                    "fip": s["stat"].get("fip")} for s in block["splits"]]))
    return pd.concat(parts, ignore_index=True)


def _level(lines: pd.DataFrame) -> pd.Series:
    level = lines["sport_id"].map(events.SPORT_LEVEL)
    rookie = lines["league_id"].map(events.ROOKIE_LEAGUE_LEVEL).fillna("rk")
    return level.where(lines["sport_id"].ne(16), rookie)


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.where(den > 0)


def hitting_production(lines: pd.DataFrame) -> pd.DataFrame:
    c = {k: lines[v].astype(float) for k, v in (("pa", "plateAppearances"), ("ab", "atBats"), ("h", "hits"),
         ("bb", "baseOnBalls"), ("hbp", "hitByPitch"), ("sf", "sacFlies"), ("tb", "totalBases"), ("hr", "homeRuns"),
         ("rbi", "rbi"), ("sb", "stolenBases"), ("k", "strikeOuts"))}
    g = pd.DataFrame(c).assign(player_id=lines["player_id"]).groupby("player_id").sum()
    g["avg"] = _ratio(g["h"], g["ab"])
    g["obp"] = _ratio(g["h"] + g["bb"] + g["hbp"], g["ab"] + g["bb"] + g["hbp"] + g["sf"])
    g["slg"] = _ratio(g["tb"], g["ab"])
    g["ops"] = g["obp"] + g["slg"]
    return g


def _pitching_counts(lines: pd.DataFrame) -> pd.DataFrame:
    c = {k: lines[v].astype(float) for k, v in (("bf", "battersFaced"), ("outs", "outs"), ("er", "earnedRuns"),
         ("h", "hits"), ("bb", "baseOnBalls"), ("hbp", "hitBatsmen"), ("hr", "homeRuns"), ("k", "strikeOuts"))}
    f = pd.DataFrame(c).assign(player_id=lines["player_id"], league=lines["league_id"].fillna(lines["sport_id"]))
    f["core"] = 13 * f["hr"] + 3 * (f["bb"] + f["hbp"]) - 2 * f["k"]
    return f


def fip_constants(lines: pd.DataFrame) -> pd.Series:
    """Per league: the constant that puts FIP on that league's ERA scale, from every pitcher in it."""
    totals = _pitching_counts(lines).groupby("league")[["er", "outs", "core"]].sum()
    return (27 * totals["er"] - 3 * totals["core"]) / totals["outs"]


def pitching_production(lines: pd.DataFrame, constants: pd.Series | None) -> pd.DataFrame:
    """With league constants, FIP across several leagues weights each league's constant by innings."""
    f = _pitching_counts(lines)
    if constants is not None:
        f["fip_runs"] = 3 * f["core"] + f["league"].map(constants) * f["outs"]
    g = f.drop(columns="league").groupby("player_id").sum()
    g["era"] = _ratio(27 * g["er"], g["outs"])
    g["whip"] = _ratio(3 * (g["h"] + g["bb"]), g["outs"])
    if constants is not None:
        g["fip"] = _ratio(g["fip_runs"], g["outs"])
    return g


def statcast_skills(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hitter and pitcher underlying skills from the season's MLB Statcast cache."""
    cols = ["game_type", "game_date", "batter", "pitcher", "description", "zone", "events", "bb_type", "pitch_type",
            "release_speed", "launch_speed", "launch_angle", "launch_speed_angle", "estimated_woba_using_speedangle",
            "woba_value", "woba_denom"]
    p = regular_season_pitches(pd.read_parquet(STATCAST_DIR / f"statcast_{season}.parquet", columns=cols), season)
    for col in cols[2:]:
        if col not in ("description", "events", "bb_type", "pitch_type", "game_type", "game_date"):
            p[col] = pd.to_numeric(p[col], errors="coerce")
    p["swing"] = p["description"].isin(SWING)
    p["whiff"] = p["description"].isin(WHIFF)
    p["chase_chance"] = p["zone"] > IN_ZONE_MAX
    bbe = p[p["description"].eq("hit_into_play") & p["launch_speed"].notna()].assign(
        barrel=lambda b: b["launch_speed_angle"].eq(6), hard=lambda b: b["launch_speed"] >= HARD_HIT,
        gb=lambda b: b["bb_type"].eq("ground_ball"))
    # xwOBA: the expected value of each tracked batted ball, the actual wOBA value of every other plate appearance.
    pa = p[p["woba_denom"].gt(0)].copy()
    tracked = pa["description"].eq("hit_into_play") & pa["estimated_woba_using_speedangle"].notna()
    pa["xw"] = pa["estimated_woba_using_speedangle"].where(tracked, pa["woba_value"])
    out = []
    for actor in ("batter", "pitcher"):
        b = bbe.groupby(actor)
        s = pd.DataFrame({"bbe": b.size(), "barrel": b["barrel"].mean(), "hard_hit": b["hard"].mean(),
                          "ev": b["launch_speed"].mean(), "la": b["launch_angle"].mean(), "gb": b["gb"].mean()})
        x = pa.groupby(actor)
        s = s.join(_ratio(x["xw"].sum(), x["woba_denom"].sum()).rename("xwoba"), how="outer")
        g = p.groupby(actor)
        s = s.join(g.size().rename("pitches"), how="outer")
        chase = p[p["chase_chance"]].groupby(actor)["swing"].mean().rename("chase")
        whiff = _ratio(g["whiff"].sum(), g["swing"].sum()).rename("whiff")
        velo = p[p["pitch_type"].isin(FASTBALLS)].groupby(actor)["release_speed"].mean().rename("fb_velo")
        s = s.join(chase if actor == "batter" else whiff, how="left")
        if actor == "pitcher":
            s = s.join(velo, how="left")
        s.index = s.index.astype("int64").rename("player_id")
        out.append(s)
    return out[0], out[1]


def feed_skills(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hitter and pitcher underlying skills across a season's affiliated levels from the prospect feed caches."""
    games = events.load_games([season])
    games = games[games["level"].ne("afl")]
    pa = events.plate_appearances([season], games)
    for col in ("launch_speed", "launch_angle"):
        pa[col] = pd.to_numeric(pa[col], errors="coerce")
    bip = pa[pa["in_play_ball"]]
    tracked = bip[bip["launch_speed"].notna()].assign(hard=lambda b: b["launch_speed"] >= HARD_HIT)
    t = tracked.groupby("batter")["launch_speed"]
    hit = pd.DataFrame({"bbe": t.size(), "ev": t.mean(), "hard_hit": tracked.groupby("batter")["hard"].mean(),
                        "la": tracked.groupby("batter")["launch_angle"].mean(),
                        "ev90": t.quantile(0.9).where(t.size() >= EV90_MIN_BALLS)})
    # Pitch calls exist only in fully logged games (``plate_appearances`` blanks the rest), so sums skip them.
    g = pa.groupby("pitcher")
    pit = pd.DataFrame({"whiff": _ratio(g["swinging_strikes"].sum(min_count=1), g["swings"].sum(min_count=1)),
                        "gb": bip.groupby("pitcher")["bb_type"].agg(lambda s: s.eq("gb").mean())})
    tp = tracked.groupby("pitcher")
    pit = pit.join(pd.DataFrame({"bbe": tp.size(), "hard_hit": tp["hard"].mean()}), how="outer")
    pitches = pd.concat([feeds.read_kind(sid, season, "pitches") for sid in MINOR_SPORT_IDS], ignore_index=True)
    pitches = pitches[pitches["game_pk"].isin(games["game_pk"]) & pd.to_numeric(pitches["start_speed"], errors="coerce").notna()]
    pitches = pitches.assign(start_speed=pd.to_numeric(pitches["start_speed"]))
    pit = pit.join(pitches.groupby("pitcher").size().rename("pitches"), how="left")
    pit = pit.join(pitches[pitches["pitch_type"].isin(FASTBALLS)].groupby("pitcher")["start_speed"].mean().rename("fb_velo"), how="left")
    for f in (hit, pit):
        f.index = f.index.astype("int64").rename("player_id")
    return hit, pit


def _latest_season(group: str, sport_ids: tuple[int, ...], season: int) -> tuple[int, pd.DataFrame]:
    """The latest season from ``season`` back with games played: before opening day the last one is shown."""
    lines = official_lines(season, group, sport_ids)
    if lines.empty:
        season -= 1
        lines = official_lines(season, group, sport_ids)
    if lines.empty:
        raise ValueError(f"No {group} lines for sports {sport_ids} in {season} or {season + 1}")
    return season, lines


def _pack(frame: pd.DataFrame, spec: tuple[tuple[str, int], ...]) -> pd.Series:
    cols = [name for name, _ in spec]
    values = frame.reindex(columns=cols).astype(float).to_numpy() * np.array([scale for _, scale in spec], float)
    values = np.round(values)
    # Only a cameo's ERA or WHIP can leave the smallint range (a run allowed per out or worse); it shows as missing.
    values[~np.isfinite(values) | (np.abs(values) > SMALLINT_MAX)] = np.nan
    return pd.Series([[None if np.isnan(v) else int(v) for v in row] for row in values], index=frame.index)


def build(population: pd.DataFrame, season: int) -> pd.DataFrame:
    """Rows (player_id, pool, season, levels, stats) for every published player with a line in the pool's season."""
    parts = []
    for pool, sport_ids in (("mlb", (1,)), ("prospect", MINOR_SPORT_IDS)):
        pop = population[population["pool"].eq(pool)]
        year, hitting = _latest_season("hitting", sport_ids, season)
        pitching = official_lines(year, "pitching", sport_ids)
        skills = statcast_skills(year) if pool == "mlb" else feed_skills(year)
        saber = sabermetrics(year).set_index(["role", "player_id"]) if pool == "mlb" else None
        constants = fip_constants(pitching) if pool == "prospect" else None
        for role, lines, spec, skill in (("bat", hitting, BAT, skills[0]), ("pit", pitching, PIT, skills[1])):
            ids = pop.loc[pop["role"].eq(role), "player_id"]
            lines = lines[lines["player_id"].isin(ids)]
            f = hitting_production(lines) if role == "bat" else pitching_production(lines, constants)
            if saber is not None:
                f["war"] = saber.loc[role, "war"].reindex(f.index)
                if role == "pit":
                    f["fip"] = saber.loc[role, "fip"].reindex(f.index)
            f = f.join(skill.drop(columns=[c for c in skill if c in f]), how="left")
            row = pd.DataFrame({"player_id": f.index.astype("int64"), "pool": pool, "season": year, "levels": None})
            if pool == "prospect":
                lv = lines.assign(level=_level(lines))
                lv = lv.assign(rank=lv["level"].map(events.LEVEL_RANK)).sort_values("rank", ascending=False)
                row["levels"] = row["player_id"].map(lv.groupby("player_id")["level"].agg(",".join))
            row["stats"] = _pack(f, spec).to_numpy()
            parts.append(row)
            log.info(f"{pool} {role} {year}: {len(row)} of {len(ids)} published players have a line")
    return pd.concat(parts, ignore_index=True)


def population() -> pd.DataFrame:
    with engine.connect() as conn:
        frame = pd.read_sql(text("SELECT player_id, 'mlb' AS pool, position_group FROM player_values "
                                 "UNION ALL SELECT player_id, 'prospect', position_group FROM prospect_values"), conn)
    if frame.duplicated(["player_id", "pool"]).any():
        raise ValueError("Duplicate players in a published pool")
    frame["role"] = np.where(frame["position_group"].isin(PITCHER_GROUPS), "pit", "bat")
    return frame


def _unchanged(rows: pd.DataFrame) -> bool:
    with engine.connect() as conn:
        live = pd.read_sql(text("SELECT player_id, pool, season, levels, stats FROM player_season_stats"), conn)
    def key(f: pd.DataFrame) -> set:
        return {(int(r.player_id), r.pool, int(r.season), r.levels if isinstance(r.levels, str) else None, tuple(r.stats))
                for r in f.itertuples()}
    return key(live) == key(rows)


def publish(rows: pd.DataFrame) -> None:
    if len(rows) < MIN_ROWS:
        raise ValueError(f"refusing to publish {len(rows)} player_season_stats rows (under {MIN_ROWS})")
    now = datetime.now(UTC).isoformat()
    records = [{**r, "updated_at": now} for r in rows.to_dict("records")]
    with engine.begin() as conn:
        conn.execute(text("CREATE TEMP TABLE player_season_stats_in (LIKE player_season_stats) ON COMMIT DROP"))
        for start in range(0, len(records), BATCH_ROWS):
            conn.execute(text("INSERT INTO player_season_stats_in SELECT * FROM "
                              "jsonb_populate_recordset(NULL::player_season_stats_in, CAST(:rows AS jsonb))"),
                         {"rows": json.dumps(records[start:start + BATCH_ROWS])})
        conn.execute(text("TRUNCATE player_season_stats"))
        conn.execute(text("INSERT INTO player_season_stats SELECT * FROM player_season_stats_in ORDER BY pool, player_id"))
    log.info(f"published {len(records)} player_season_stats rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, default=date.today().year)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    setup_logging()
    rows = build(population(), args.season)
    if args.no_write:
        log.info(f"built {len(rows)} rows; not writing")
    elif _unchanged(rows):
        log.info("player_season_stats unchanged; skipping the publish")
    else:
        publish(rows)


if __name__ == "__main__":
    main()
