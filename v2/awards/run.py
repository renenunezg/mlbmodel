"""Freeze one season's award boards, then grade them once the winners are public.

publish refuses to run twice for a season and reads no recipients for that season.
grade reads them live and only after the board is in the database, so a forecast
can never have seen its own result.

Usage:
    python -m v2.awards.run publish --season 2026 [--no-write]
    python -m v2.awards.run grade --season 2026
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sqlalchemy import text

from backend.db import engine, writes_allowed
from backend.log import setup_logging
from v2.awards import data, features as F
from v2.awards.model import Fit, Race, backtest, fit

log = logging.getLogger(__name__)

MODEL_VERSION = "awards-v1"
BOARD_SIZE = 15
FIRST_BACKTEST = {"MVP": 2016, "CY": 2016, "ROY": 2016, "RP": 2020, "GG_FIELD": 2020, "GG_C": 2020, "GG_P": 2020}
# Backtest comparison: the single stat a reader would pick the winner by.
BASELINE = {"MVP": "war", "CY": "war", "ROY": "war", "RP": "war", "GG_FIELD": "oaa", "GG_C": "framing", "GG_P": "war"}
FINALISTS = Path(__file__).parent / "inputs" / "gold_glove_finalists_{season}.json"
BOARD_COLUMNS = ["season", "league", "award", "position", "player_id", "player_name", "team", "rank",
                 "win_probability", "vote_share", "stat_line"]


def _ballot_races(award: str, last_season: int) -> list[Race]:
    first = F.FIRST_RELIEVER_SEASON if award == "RP" else F.FIRST_BALLOT_SEASON
    races = []
    for season in range(first, last_season + 1):
        for league in data.LEAGUES:
            pool = F.ballot_pool(award, season, league)
            winner = np.flatnonzero(pool.player_id.isin(F.ballot_winner(award, season, league)).to_numpy())
            races.append(Race(season, pool[F.FEATURES[award]].to_numpy(float), F.ballot_targets(award, pool, season, league),
                              int(winner[0]) if len(winner) else -1))
    return races


def _glove_races(kind: str, positions: tuple[str, ...], last_season: int) -> list[Race]:
    races = []
    for season in range(F.FIRST_GLOVE_SEASON, last_season + 1):
        for league in data.LEAGUES:
            winners = {position: player for player, position in F.glove_positions(season, league).items()}
            for position in positions:
                pool = F.glove_pool(season, league, position)
                winner = np.flatnonzero(pool.player_id.eq(winners.get(position)).to_numpy())
                if not len(winner):
                    raise ValueError(f"{position} winner outside the {league} {season} pool")
                y = np.zeros(len(pool))
                y[winner[0]] = 1.0
                races.append(Race(season, pool[F.FEATURES[kind]].to_numpy(float), y, int(winner[0])))
    return races


def _stat_line(award: str, row: pd.Series) -> str:
    war = f"{row.real_war:.1f} fWAR"
    if award == "RP":
        return f"{row.real_saves:.0f} SV · {row.real_holds:.0f} HLD · {row.era:.2f} ERA · {row.real_k:.0f} K · {war}"
    if award == "CY" or (row.is_pitcher and award in ("MVP", "ROY")):
        return f"{war} · {row.era:.2f} ERA · {row.real_ip:.0f} IP · {row.real_k:.0f} K · {row.real_wins:.0f} W"
    return f"{war} · {row.real_hr:.0f} HR · {row.real_rbi:.0f} RBI · {f'{row.ops:.3f}'.removeprefix('0')} OPS"


def _ballot_board(award: str, season: int, model: Fit) -> pd.DataFrame:
    parts = []
    for league in data.LEAGUES:
        pool = F.ballot_pool(award, season, league)
        x = pool[F.FEATURES[award]].to_numpy(float)
        top = pool.assign(win_probability=model.win(x), vote_share=model.share(x) if award != "RP" else np.nan)
        top = top.sort_values("win_probability", ascending=False).head(BOARD_SIZE).reset_index(drop=True)
        top["stat_line"] = top.apply(lambda row: _stat_line(award, row), axis=1)
        parts.append(top.assign(league=league, award=award, position="", rank=np.arange(1, len(top) + 1)))
    return pd.concat(parts, ignore_index=True)


def _name_key(name: str) -> str:
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\b(jr|sr|ii|iii)\b", "", re.sub(r"[^a-z ]", "", plain)).strip()


def _finalist_ids(names: list[str], lines: pd.DataFrame, what: str) -> list[int]:
    keys = lines.player_name.map(_name_key)
    ids = []
    for name in names:
        match = lines.player_id[keys.eq(_name_key(name))]
        if match.nunique() != 1:
            raise ValueError(f"{what} finalist {name!r} matched {match.nunique()} players")
        ids.append(int(match.iloc[0]))
    return ids


def _glove_stat_line(position: str, row: pd.Series) -> str:
    prior = int(round(np.expm1(row.raw_prior_wins)))
    won_before = f" · {prior} prior" if prior else ""
    if position == "P":
        return f"{row.raw_war:.1f} fWAR · {row.raw_innings:.0f} IP · {row.errors:.0f} E{won_before}"
    if position == "C":
        return (f"{row.raw_framing:+.0f} framing · {row.raw_blocking:+.0f} blocking · {row.raw_throwing:+.0f} throwing runs · "
                f"{row.raw_innings:.0f} inn{won_before}")
    return f"{row.raw_oaa:+.0f} OAA · {row.raw_frv:+.0f} runs · {row.raw_innings:.0f} inn{won_before}"


def _glove_board(season: int, models: dict[str, Fit]) -> pd.DataFrame:
    finalists = json.loads(Path(str(FINALISTS).format(season=season)).read_text())
    parts = []
    for league in data.LEAGUES:
        for position, names in finalists[league].items():
            if position == "UT":
                lines = F.season_lines(season, league)
                ids = _finalist_ids(names, lines, f"{league} UT")
                pool, kind = F.utility_pool(season, league, ids), "GG_FIELD"
                pool["errors"] = 0.0
            else:
                full = F.glove_pool(season, league, position)
                ids = _finalist_ids(names, full, f"{league} {position}")
                pool = full[full.player_id.isin(ids)].reset_index(drop=True)
                kind = "GG_C" if position == "C" else "GG_P" if position == "P" else "GG_FIELD"
            if len(pool) != len(names):
                raise ValueError(f"{league} {position} finalists fell outside the pool")
            scores = models[kind].win(pool[F.FEATURES[kind]].to_numpy(float))
            # A pitcher or utility forecast has not earned a probability: the first called 2 of 12, no better than fWAR, and the second has no backtest.
            shown = np.full(len(pool), np.nan) if position in ("P", "UT") else scores
            pool = pool.assign(score=scores, win_probability=shown, vote_share=np.nan).sort_values("score", ascending=False)
            pool["stat_line"] = pool.apply(lambda row: _glove_stat_line(position, row), axis=1)
            parts.append(pool.assign(league=league, award="GG", position=position, rank=np.arange(1, len(pool) + 1)))
    return pd.concat(parts, ignore_index=True)


def build(season: int) -> tuple[pd.DataFrame, list[dict]]:
    races = {award: _ballot_races(award, season - 1) for award in ("MVP", "CY", "ROY", "RP")}
    races |= {"GG_FIELD": _glove_races("GG_FIELD", F.FIELD_POSITIONS, season - 1),
              "GG_C": _glove_races("GG_C", ("C",), season - 1), "GG_P": _glove_races("GG_P", ("P",), season - 1)}
    models = {kind: fit(rows) for kind, rows in races.items()}
    as_of = datetime.now(UTC).isoformat()
    meta = []
    for kind, rows in races.items():
        summary = backtest(rows, FIRST_BACKTEST[kind], F.FEATURES[kind].index(BASELINE[kind]))
        meta.append({"season": season, "model": kind, "as_of": as_of, "model_version": MODEL_VERSION,
                     "backtest": {**summary, "baseline": BASELINE[kind], "penalty": models[kind].penalty,
                                  "sharpen": models[kind].sharpen, "trained_through": season - 1}})
    board = pd.concat([_ballot_board(award, season, models[award]) for award in ("MVP", "CY", "ROY", "RP")]
                      + [_glove_board(season, models)], ignore_index=True)
    board["season"] = season
    return board[BOARD_COLUMNS], meta


def _json_rows(frame: pd.DataFrame) -> str:
    clean = frame.astype(object).where(frame.notna(), None)
    return json.dumps(clean.to_dict("records"), default=lambda v: v.item())


def publish(board: pd.DataFrame, meta: list[dict]) -> None:
    season = int(board.season.iloc[0])
    with engine.begin() as conn:
        if conn.execute(text("SELECT 1 FROM mlb.award_models WHERE season = :season LIMIT 1"), {"season": season}).first():
            raise RuntimeError(f"{season} award boards are already frozen")
        # One statement per table, so one revalidation per table.
        conn.execute(text("INSERT INTO mlb.award_boards (season, league, award, position, player_id, player_name, team, rank, "
                          "win_probability, vote_share, stat_line) SELECT season, league, award, position, player_id, player_name, "
                          "team, rank, win_probability, vote_share, stat_line FROM jsonb_populate_recordset(NULL::mlb.award_boards, "
                          "CAST(:rows AS jsonb))"), {"rows": _json_rows(board)})
        conn.execute(text("INSERT INTO mlb.award_models SELECT * FROM jsonb_populate_recordset(NULL::mlb.award_models, "
                          "CAST(:rows AS jsonb))"), {"rows": json.dumps(meta)})
    log.info(f"froze {len(board)} award rows and {len(meta)} models for {season}")


def _live_recipients(award_id: str, season: int) -> pd.DataFrame:
    """Uncached on purpose: an early read of an unannounced award is empty or 404, and must not stick."""
    response = requests.get(f"{data.STATSAPI}/awards/{award_id}/recipients", params={"season": season}, timeout=60)
    if response.status_code == 404:
        return pd.DataFrame(columns=["player_id", "player_name"])
    response.raise_for_status()
    return pd.DataFrame([{"player_id": a["player"]["id"], "player_name": a["player"]["nameFirstLast"]}
                         for a in response.json()["awards"]], columns=["player_id", "player_name"])


def grade(season: int) -> None:
    """Mark every announced race. Winners off the board are added unranked with no forecast."""
    with engine.begin() as conn:
        board = pd.read_sql(text("SELECT league, award, position, player_id, won FROM mlb.award_boards WHERE season = :season"),
                            conn, params={"season": season})
        if board.empty:
            raise RuntimeError(f"No frozen {season} board to grade")
        codes = {"MVP": "{}MVP", "CY": "{}CY", "ROY": "{}ROY"}
        for (league, award, position), race in board[board.won.isna()].groupby(["league", "award", "position"]):
            if award == "GG":
                recipients = _live_recipients(f"{league}GG", season)
                winner = recipients[recipients.player_id.isin(race.player_id)]
            elif award == "RP":
                # The BBWAA has no Stats API award yet; a reliever winner is entered by hand once its code is known.
                continue
            else:
                winner = _live_recipients(codes[award].format(league), season)
            if winner.empty:
                continue
            ids = [int(p) for p in winner.player_id]
            where = "season = :s AND league = :l AND award = :a AND position = :p"
            keys = {"s": season, "l": league, "a": award, "p": position}
            conn.execute(text(f"UPDATE mlb.award_boards SET won = player_id = ANY(:ids) WHERE {where} AND won IS NULL"), {**keys, "ids": ids})
            for row in winner[~winner.player_id.isin(race.player_id)].itertuples():
                conn.execute(text("INSERT INTO mlb.award_boards (season, league, award, position, player_id, player_name, won) "
                                  "VALUES (:s, :l, :a, :p, :id, :name, true)"), {**keys, "id": int(row.player_id), "name": row.player_name})
            log.info(f"graded {league} {award} {position}: {', '.join(winner.player_name)}")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["publish", "grade"])
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    if args.command == "grade":
        grade(args.season)
        return
    board, meta = build(args.season)
    for row in meta:
        b = row["backtest"]
        log.info(f"{row['model']}: {b['hits']} of {b['races']} winners, {b['baseline']} leader {b['baseline_hits']}, "
                 f"log loss {b['log_loss']:.2f} vs uniform {b['uniform_log_loss']:.2f}")
    if args.no_write or not writes_allowed():
        pd.set_option("display.width", 200, "display.max_colwidth", 70, "display.max_rows", 400)
        print(board[board["rank"].le(3)].drop(columns=["season", "player_id", "team"]).to_string(index=False))
        return
    publish(board, meta)


if __name__ == "__main__":
    main()
