"""Exact bracket propagation conditional on simulated per-game probabilities."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable

HOME_GAMES = {
    3: (True, True, True),
    5: (True, True, False, False, True),
    7: (True, True, False, False, False, True, True),
}


def series_outcomes(probabilities: list[float], wins: tuple[int, int] = (0, 0)) -> list[dict]:
    """Absorbing-state recursion, including completed games exactly once."""
    best_of = len(probabilities)
    if best_of not in HOME_GAMES or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Invalid series game probabilities")
    target = best_of // 2 + 1
    a, b = wins
    if any(type(w) is not int or w < 0 or w > target for w in wins) or (a == b == target):
        raise ValueError("Invalid current series score")
    active = {wins: 1.0}
    finished: dict[tuple[int, int], float] = defaultdict(float)
    while active:
        next_states: dict[tuple[int, int], float] = defaultdict(float)
        for (a, b), mass in active.items():
            if max(a, b) == target:
                finished[a, b] += mass
                continue
            p = probabilities[a + b]
            next_states[a + 1, b] += mass * p
            next_states[a, b + 1] += mass * (1 - p)
        active = next_states
    return [{"wins": list(score), "probability": p} for score, p in sorted(finished.items()) if p > 0]


def bracket_nodes(teams: list[dict]) -> list[dict]:
    if len(teams) != 12 or len({t["code"] for t in teams}) != 12:
        raise ValueError("The bracket requires 12 distinct teams")
    nodes = []
    for league in ("AL", "NL"):
        seeds = {t["seed"]: t["code"] for t in teams if t["league"] == league}
        if set(seeds) != set(range(1, 7)):
            raise ValueError(f"Invalid {league} seeds")
        for suffix, left, right in (("45", 4, 5), ("36", 3, 6)):
            nodes.append(
                dict(
                    id=f"{league}-WC{suffix}",
                    round="WC",
                    league=league,
                    best_of=3,
                    left=seeds[left],
                    right=seeds[right],
                )
            )
        for seed, suffix in ((1, "45"), (2, "36")):
            nodes.append(
                dict(
                    id=f"{league}-DS{seed}",
                    round="DS",
                    league=league,
                    best_of=5,
                    left=seeds[seed],
                    right=f"{league}-WC{suffix}",
                )
            )
        nodes.append(
            dict(id=f"{league}-CS", round="CS", league=league, best_of=7, left=f"{league}-DS1", right=f"{league}-DS2")
        )
    nodes.append(dict(id="WS", round="WS", league="MLB", best_of=7, left="AL-CS", right="NL-CS"))
    return nodes


def forecast(teams: list[dict], game_probabilities: Callable, completed: dict | None = None) -> dict:
    """Sum every possible path; client sampling consumes these same terminal masses."""
    nodes = bracket_nodes(teams)
    by_code = {t["code"]: t for t in teams}
    distributions = {t["code"]: {t["code"]: 1.0} for t in teams}
    odds = {
        t["code"]: {"team": t["code"], "DS": float(t["seed"] <= 2), "CS": 0.0, "WS": 0.0, "champion": 0.0}
        for t in teams
    }
    matchups = {}
    completed = completed or {}
    if set(completed) - {node["id"] for node in nodes}:
        raise ValueError("Unknown completed series")
    for node in nodes:
        result: dict[str, float] = defaultdict(float)
        for a, mass_a in distributions[node["left"]].items():
            for b, mass_b in distributions[node["right"]].items():
                if not mass_a * mass_b:
                    continue
                ta, tb = by_code[a], by_code[b]
                # Official ranking includes regular-season tiebreakers; seeds only apply within a league.
                home = a if (ta["ws_rank"] < tb["ws_rank"] if node["round"] == "WS" else ta["seed"] < tb["seed"]) else b
                game_ps = game_probabilities(a, b, home, node["best_of"])
                if len(game_ps) != node["best_of"]:
                    raise ValueError("Game probabilities do not match the series length")
                known = completed.get(node["id"], {})
                if known and set(known) != {a, b}:
                    raise ValueError(f"Completed results contradict bracket at {node['id']}")
                current = (known.get(a, 0), known.get(b, 0))
                outcomes = series_outcomes(game_ps, current)
                key = f"{node['id']}:{a}:{b}"
                matchups[key] = {
                    "teams": [a, b],
                    "home_field": home,
                    "current_wins": list(current),
                    "game_probabilities": game_ps,
                    "outcomes": outcomes,
                }
                p_a = sum(o["probability"] for o in outcomes if o["wins"][0] > o["wins"][1])
                result[a] += mass_a * mass_b * p_a
                result[b] += mass_a * mass_b * (1 - p_a)
        distributions[node["id"]] = {t: p for t, p in result.items() if p > 1e-15}
        if not math.isclose(sum(result.values()), 1.0, abs_tol=1e-9):
            raise ValueError(f"Probability mass lost at {node['id']}")
        stage = {"WC": "DS", "DS": "CS", "CS": "WS", "WS": "champion"}[node["round"]]
        for team, p in result.items():
            odds[team][stage] = p
    return {"nodes": nodes, "matchups": matchups, "odds": sorted(odds.values(), key=lambda x: -x["champion"])}
