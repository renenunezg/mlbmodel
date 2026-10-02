"""Build a read-only playoff forecast; --publish explicitly writes its Supabase snapshot.

python -m v2.postseason.publish --output /tmp/playoffs.json
Only fresh, full-resolution forecasts may be published.
One snapshot is kept per round, taken before that round's first final, so the
site can show what the bracket prediction was going into each round.
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from functools import cache
from pathlib import Path

import numpy as np
import requests
from sqlalchemy import text

from backend.db import engine
from backend.strategy import HOME_FIELD_LOGIT
from backend.team_mappings import normalize_team
from v2.bayesian._common import POSTERIORS_DIR
from v2.pipeline.score_games import STATCAST_VENUE_CODES
from v2.postseason.bracket import HOME_GAMES, bracket_nodes, forecast
from v2.simulator import BullpenQueue, GameInputs, load_advancement_table, load_out_subtype_table, simulate_game
from v2.simulator.posteriors import load_posteriors, posterior_provenance

log = logging.getLogger(__name__)
API = "https://statsapi.mlb.com/api/v1"
MAX_MODEL_AGE = 14
MIN_PUBLISH_SIMS = 10000


def api(path: str, **params) -> dict:
    response = requests.get(f"{API}/{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def field(year: int) -> list[dict]:
    standings = api("standings", leagueId="103,104", season=year, standingsTypes="regularSeason", hydrate="team")
    teams = []
    for league_id, league in ((103, "AL"), (104, "NL")):
        rows = [r for div in standings["records"] if div["league"]["id"] == league_id for r in div["teamRecords"]]
        champs = sorted(
            [r for r in rows if r.get("divisionChamp") and r["divisionRank"] == "1"], key=lambda r: int(r["leagueRank"])
        )
        wild = sorted([r for r in rows if r.get("clinchIndicator") == "w"], key=lambda r: int(r["wildCardRank"]))
        if len(champs) != 3 or len(wild) != 3:
            raise ValueError("The official 12-team postseason field is not settled yet")
        for seed, row in enumerate(champs + wild, 1):
            t = row["team"]
            teams.append(
                dict(
                    code=normalize_team(t["abbreviation"]),
                    name=t["name"],
                    id=t["id"],
                    seed=seed,
                    league=league,
                    wins=row["wins"],
                    losses=row["losses"],
                    ws_rank=int(row["sportRank"]),
                )
            )
    if len({t["ws_rank"] for t in teams}) != 12:
        raise ValueError("World Series home-field ranking is unresolved")
    return teams


def regular_season_end(year: int) -> date:
    """Date of the last regular-season game, the latest a model can be trained through."""
    dates = [d["date"] for d in api("schedule", sportId=1, season=year, gameType="R")["dates"]]
    if not dates:
        raise ValueError("The regular-season schedule is unavailable")
    return date.fromisoformat(max(dates))


def model_is_stale(cutoff: date, today: date, season_end: date) -> bool:
    """The model trains on regular-season games only, so once it covers the
    whole regular season it stays current for the entire postseason."""
    return cutoff < season_end and (today - cutoff).days > MAX_MODEL_AGE


def roster_plan(team: dict, year: int) -> dict:
    roster = api(
        f"teams/{team['id']}/roster",
        rosterType="active",
        hydrate=f"person(stats(group=[hitting,pitching],type=[season],season={year}))",
    )["roster"]
    players = {}
    for row in roster:
        p = row["person"]
        stats = {}
        for group in p.get("stats", []):
            splits = [s for s in group.get("splits", []) if s.get("gameType", "R") == "R"]
            # Traded players can have several team splits; counts, unlike rates, are additive.
            stats[group["group"]["displayName"]] = {
                k: sum(int(s["stat"].get(k, 0)) for s in splits)
                for k in ("plateAppearances", "gamesStarted", "gamesPitched", "outs", "saves", "holds")
            }
        players[p["id"]] = dict(
            id=p["id"],
            name=p["fullName"],
            hand=p.get("pitchHand", {}).get("code", "R"),
            hitting=stats.get("hitting", {}),
            pitching=stats.get("pitching", {}),
        )
    hitters = sorted(
        [p for p in players.values() if p["hitting"].get("plateAppearances", 0) > 0],
        key=lambda p: -p["hitting"]["plateAppearances"],
    )
    starters = sorted(
        [
            p
            for p in players.values()
            if p["pitching"].get("gamesStarted", 0) >= 2
            and p["pitching"]["gamesStarted"] >= p["pitching"].get("gamesPitched", 0) / 2
        ],
        key=lambda p: (-p["pitching"]["gamesStarted"], -p["pitching"].get("outs", 0)),
    )[:4]
    relievers = sorted(
        [
            p
            for p in players.values()
            if p["pitching"].get("gamesPitched", 0) >= 3
            and p["pitching"].get("gamesStarted", 0) < p["pitching"]["gamesPitched"] / 2
        ],
        key=lambda p: (-p["pitching"].get("saves", 0) - p["pitching"].get("holds", 0), -p["pitching"]["gamesPitched"]),
    )[:8]
    if len(hitters) < 9 or len(starters) < 3 or len(relievers) < 3:
        raise ValueError(f"Insufficient current roster evidence for {team['code']}")
    return {
        "players": players,
        "lineup": [p["id"] for p in hitters[:9]],
        "rotation": [p["id"] for p in starters],
        "bullpen": [p["id"] for p in relievers],
    }


def schedule_context(teams: list[dict], year: int) -> tuple[dict, dict]:
    games = [
        g
        for d in api("schedule", sportId=1, season=year, gameTypes="F,D,L,W", hydrate="team,probablePitcher")["dates"]
        for g in d["games"]
    ]
    by_id = {t["id"]: t for t in teams}
    nodes = {n["id"]: n for n in bracket_nodes(teams)}
    completed, scheduled = {}, {}
    for game in games:
        sides = [game["teams"][s] for s in ("home", "away")]
        if any(s["team"]["id"] not in by_id for s in sides):
            continue  # Unassigned future opponents have placeholder team ids.
        h, a = [by_id[s["team"]["id"]] for s in sides]
        league = h["league"]
        kind = game["gameType"]
        if kind == "F":
            suffix = "45" if {h["seed"], a["seed"]} == {4, 5} else "36"
            node_id = f"{league}-WC{suffix}"
        elif kind == "D":
            node_id = f"{league}-DS{min(h['seed'], a['seed'])}"
        else:
            node_id = "WS" if kind == "W" else f"{league}-CS"
        if node_id not in nodes:
            raise ValueError("Official schedule does not match the postseason bracket")
        number = int(game["seriesGameNumber"])
        scheduled[node_id, h["code"], a["code"], number] = game
        if game["status"]["abstractGameState"] == "Final":
            if sides[0].get("score") == sides[1].get("score"):
                raise ValueError("Tied final postseason game")
            score = completed.setdefault(node_id, {h["code"]: 0, a["code"]: 0})
            score[h["code"] if sides[0]["score"] > sides[1]["score"] else a["code"]] += 1
    return completed, scheduled


def stage(nodes: list[dict], completed: dict) -> tuple[str, bool]:
    """The first round still undecided, and whether one of its games is final."""
    for name in ("WC", "DS", "CS", "WS"):
        series = [n for n in nodes if n["round"] == name]
        wins = [max(completed.get(n["id"], {}).values(), default=0) for n in series]
        if any(w < n["best_of"] // 2 + 1 for w, n in zip(wins, series)):
            return name, any(completed.get(n["id"]) for n in series)
    raise ValueError("The postseason is complete; there is no round left to forecast")


def queue(plan: dict, starter: int) -> BullpenQueue:
    p = plan["players"].get(starter, {}).get("pitching", {})
    regular = p.get("gamesStarted", 0) >= 2 and p["gamesStarted"] >= p.get("gamesPitched", 0) / 2
    # A probable opener or an unknown pitcher gets one inning and no guessed bulk arm.
    outs = min(18, max(3, round(p.get("outs", 0) / max(1, p.get("gamesPitched", 0))))) if regular else 3
    relief = [pid for pid in plan["bullpen"] if pid != starter]
    return BullpenQueue(
        starter,
        relief,
        starter_role=0 if regular else 1,
        workloads={starter: (outs,), **{pid: (3,) for pid in relief}},
        roster_source="active_roster",
        availability_assumption="full_rest",
    )


def build_snapshot(
    n_sims: int = 10000, seed: int = 2026, posteriors_dir: Path = POSTERIORS_DIR, for_publication: bool = False
) -> dict:
    today = date.today()
    year = today.year
    provenance = posterior_provenance(posteriors_dir)
    cutoff = date.fromisoformat(provenance["training_max_date"])
    if cutoff >= today:
        raise ValueError("Model training cutoff must precede the forecast date")
    stale = model_is_stale(cutoff, today, regular_season_end(year))
    if for_publication and (stale or n_sims < MIN_PUBLISH_SIMS):
        raise ValueError("Refresh model artifacts and use at least 10,000 simulations before publication")
    teams = field(year)
    with ThreadPoolExecutor(max_workers=4) as pool:
        plans = dict(zip([t["code"] for t in teams], pool.map(lambda t: roster_plan(t, year), teams)))
    completed, scheduled = schedule_context(teams, year)
    pm = load_posteriors(posteriors_dir)
    adv, sub = load_advancement_table(), load_out_subtype_table()
    if (
        not adv.training_max_date
        or adv.training_max_date >= today.isoformat()
        or sub.training_max_date != adv.training_max_date
    ):
        raise ValueError("Simulator tables need consistent pre-forecast training cutoffs")
    warnings = []
    if stale:
        warnings.append(f"Model training ends {cutoff}; refresh model artifacts before publication.")
    if n_sims < MIN_PUBLISH_SIMS:
        warnings.append(f"Preview uses {n_sims:,} simulations per game; publication requires {MIN_PUBLISH_SIMS:,}.")
    rng = np.random.default_rng(seed)
    by_code = {t["code"]: t for t in teams}
    for code, plan in plans.items():
        team = by_code[code]
        known_batters = set(pm.batter_ids)
        known_pitchers = set(pm.pitcher_ids)
        missing = [pid for pid in plan["lineup"] if pid not in known_batters]
        missing += [pid for pid in plan["rotation"] + plan["bullpen"] if pid not in known_pitchers]
        team["lineup"] = [plan["players"][p]["name"] for p in plan["lineup"]]
        team["rotation"] = [plan["players"][p]["name"] for p in plan["rotation"]]
        team["unmodeled_players"] = [plan["players"][p]["name"] for p in missing]
        if missing:
            warnings.append(f"{code}: {len(missing)} projected players use league-average skill fallbacks.")

    @cache
    def game_probability(home: str, away: str, hp: int, ap: int) -> float:
        ph, pa = plans[home], plans[away]
        inputs = GameInputs(
            np.array(ph["lineup"]),
            np.array(pa["lineup"]),
            queue(ph, hp),
            queue(pa, ap),
            STATCAST_VENUE_CODES.get(home, home),
            {pid: p["hand"] for pid, p in ph["players"].items()},
            {pid: p["hand"] for pid, p in pa["players"].items()},
            automatic_runner=False,
        )
        hr, ar = simulate_game(rng, pm, adv, sub, inputs, n_sims=n_sims)
        if np.any(hr == ar):
            raise ValueError("Unresolved postseason simulation")
        # Jeffreys smoothing avoids certainties from finite Monte Carlo samples.
        p = (float((hr > ar).sum()) + 0.5) / (n_sims + 1)
        return float(1 / (1 + np.exp(-(np.log(p / (1 - p)) + HOME_FIELD_LOGIT))))

    node_lookup = {(n["league"], n["best_of"], n["left"]): n["id"] for n in bracket_nodes(teams)}

    def probabilities(a: str, b: str, advantage: str, best_of: int) -> list[float]:
        other = b if advantage == a else a
        league = by_code[a]["league"]
        if best_of == 3:
            node = f"{league}-WC{'45' if by_code[a]['seed'] in (4, 5) else '36'}"
        elif best_of == 5:
            node = node_lookup[league, 5, a]
        else:
            node = "WS" if by_code[a]["league"] != by_code[b]["league"] else f"{league}-CS"
        ps = []
        for i, at_home in enumerate(HOME_GAMES[best_of]):
            home, away = (advantage, other) if at_home else (other, advantage)
            hplan, aplan = plans[home], plans[away]
            hp, ap = hplan["rotation"][i % len(hplan["rotation"])], aplan["rotation"][i % len(aplan["rotation"])]
            official = scheduled.get((node, home, away, i + 1))
            if official:
                hp = official["teams"]["home"].get("probablePitcher", {}).get("id", hp)
                ap = official["teams"]["away"].get("probablePitcher", {}).get("id", ap)
            p = game_probability(home, away, hp, ap)
            ps.append(p if home == a else 1 - p)
        log.info("%s %s vs %s", node, a, b)
        return ps

    result = forecast(teams, probabilities, completed)
    current_stage, stage_started = stage(result["nodes"], completed)
    return dict(
        schema_version=1,
        season=year,
        stage=current_stage,
        stage_started=stage_started,
        generated_at=datetime.now(UTC).isoformat(),
        model_version=provenance["model_version"],
        training_max_date=str(cutoff),
        n_sims=n_sims,
        seed=seed,
        probability_source="pure_model_hfa",
        teams=teams,
        warnings=warnings,
        publishable=not stale and n_sims >= MIN_PUBLISH_SIMS,
        assumptions=[
            "Model-only plate-appearance simulation, with the model home-field adjustment. No betting-market blend.",
            "Current active-roster hitters ranked by season plate appearances form projected batting orders.",
            "Probable pitchers are used where announced. Other games cycle projected regular starters ranked by season starts.",
            "Rotations reset each round; full-rest bullpens and neutral weather. Future injuries, substitutions and fatigue are not simulated.",
            "Posterior mean player skills are used. Odds include game randomness, not full parameter uncertainty.",
            "Completed games are locked at the snapshot time. In-progress games are treated as unplayed.",
            "Series odds are conditional on the displayed matchup. Championship odds sum every possible bracket path.",
        ],
        **result,
    )


def publish(snapshot: dict) -> None:
    if not snapshot["publishable"]:
        raise ValueError("Refusing to publish a stale or low-resolution forecast")
    # A round's snapshot is the prediction going into it: refresh it freely
    # until the round's first final, then keep it. A round with no snapshot
    # yet still gets one, with the finals so far locked in.
    with engine.begin() as conn:
        stored = conn.execute(
            text("""INSERT INTO playoff_forecasts (season, stage, generated_at, payload)
            VALUES (:season, :stage, :generated_at, CAST(:payload AS jsonb))
            ON CONFLICT (season, stage) DO UPDATE
            SET generated_at = EXCLUDED.generated_at, payload = EXCLUDED.payload
            WHERE playoff_forecasts.generated_at < EXCLUDED.generated_at AND NOT :stage_started"""),
            dict(
                season=snapshot["season"],
                stage=snapshot["stage"],
                generated_at=snapshot["generated_at"],
                payload=json.dumps(snapshot),
                stage_started=snapshot["stage_started"],
            ),
        ).rowcount
    if not stored:
        log.info("%s is under way; its pre-round snapshot stays frozen", snapshot["stage"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sims", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--posteriors-dir", type=Path, default=POSTERIORS_DIR)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()
    if args.n_sims < 100:
        ap.error("--n-sims must be at least 100")
    snapshot = build_snapshot(args.n_sims, args.seed, args.posteriors_dir, for_publication=args.publish)
    if args.output:
        args.output.write_text(json.dumps(snapshot, indent=2) + "\n")
    if args.publish:
        publish(snapshot)
    log.info("Completed forecast: %s", ", ".join(f"{o['team']} {o['champion']:.1%}" for o in snapshot["odds"][:3]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
