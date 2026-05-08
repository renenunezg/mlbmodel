"""Phase 4 acceptance gate: simulated 2025 runs/team-game mean and variance within 5%
of actual MLB 2025."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest

from v2.data.pa_dataset import EVENT_TO_OUTCOME, NON_PA_EVENTS

POSTERIORS = [Path("v2/bayesian/posteriors") / f for f in ("batter_skill.nc", "pitcher_skill.nc", "park_effects.nc")]
TABLES = [Path("v2/simulator/tables") / f for f in ("advancement.parquet", "out_subtype.parquet")]
CACHE_2025 = Path("cache/statcast_2025.parquet")

skip_if_missing = pytest.mark.skipif(
    not (all(p.exists() for p in POSTERIORS) and all(p.exists() for p in TABLES) and CACHE_2025.exists()),
    reason="missing posteriors / tables / 2025 cache",
)


def _classify_roles_2025(pa_df: pd.DataFrame) -> dict[int, int]:
    """Same heuristic as v2/tests/test_pa_sim.py."""
    inning1 = pa_df[pa_df["inning"] == 1].copy()
    starter_side = np.where(inning1["inning_topbot"] == "Top", "away", "home")
    inning1 = inning1.assign(side=starter_side)
    starters = (
        inning1.groupby(["game_pk", "side"])["pitcher"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
        .dropna().astype("int64")
    )
    starts = starters.value_counts()
    games = pa_df.groupby("pitcher")["game_pk"].nunique()
    share = (starts / games).fillna(0.0)
    sp = set(share[share >= 0.5].index)
    return {int(p): (0 if int(p) in sp else 1) for p in games.index}


def _build_pa_frame(year: int) -> pd.DataFrame:
    df = pd.read_parquet(
        Path(f"cache/statcast_{year}.parquet"),
        columns=[
            "game_pk", "game_date", "inning", "inning_topbot", "at_bat_number", "pitch_number",
            "events", "batter", "pitcher", "stand", "p_throws", "home_team", "away_team",
        ],
    )
    df = df[df["events"].notna() & ~df["events"].isin(NON_PA_EVENTS)]
    df["outcome"] = df["events"].map(EVENT_TO_OUTCOME)
    df = df[df["outcome"].notna()].copy()
    df = df.sort_values(["game_pk", "inning", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    return df


def _build_game_inputs(pa_df: pd.DataFrame, queues: dict, role_lookup: dict[int, int]):
    """For each game, build a GameInputs object using actual lineups + bullpens.

    Lineup = first 9 distinct batters per side in at_bat_number order.
    Bullpen = pitchers used in that game from `queues`.
    """
    from v2.simulator.game_sim import GameInputs

    # global throws lookup
    throws_lookup = (
        pa_df.drop_duplicates("pitcher").set_index("pitcher")["p_throws"].to_dict()
    )
    throws_lookup = {int(k): v for k, v in throws_lookup.items()}

    games = []
    for gp, grp in pa_df.groupby("game_pk", sort=False):
        # batting side per PA: top → away batting, bot → home batting
        away_pa = grp[grp["inning_topbot"] == "Top"]
        home_pa = grp[grp["inning_topbot"] == "Bot"]

        away_lineup = []
        for b in away_pa["batter"]:
            b = int(b)
            if b not in away_lineup:
                away_lineup.append(b)
            if len(away_lineup) == 9:
                break
        home_lineup = []
        for b in home_pa["batter"]:
            b = int(b)
            if b not in home_lineup:
                home_lineup.append(b)
            if len(home_lineup) == 9:
                break
        if len(away_lineup) < 9 or len(home_lineup) < 9:
            continue
        venue = grp["home_team"].iloc[0]
        if (gp, "home") not in queues or (gp, "away") not in queues:
            continue
        games.append((
            int(gp),
            GameInputs(
                home_lineup=np.array(home_lineup, dtype=np.int64),
                away_lineup=np.array(away_lineup, dtype=np.int64),
                home_queue=queues[(gp, "home")],
                away_queue=queues[(gp, "away")],
                venue=str(venue),
                home_p_throws_lookup=throws_lookup,
                away_p_throws_lookup=throws_lookup,
            ),
            int((grp["inning_topbot"] == "Top").sum()),  # away PAs (proxy for sim length expectation)
        ))
    return games


@skip_if_missing
def test_single_game_smoke():
    """Smoke: run one 2025 game with n_sims=200, ensure runs are bounded and finite."""
    from v2.simulator import load_posteriors
    from v2.simulator.baserunner import load_advancement_table, load_out_subtype_table
    from v2.simulator.bullpen import build_queues_from_cache
    from v2.simulator.game_sim import simulate_game

    pa = _build_pa_frame(2025)
    role_lookup = _classify_roles_2025(pa)
    queues = build_queues_from_cache(2025)
    games = _build_game_inputs(pa, queues, role_lookup)
    assert len(games) > 0, "no games built"

    pm = load_posteriors()
    adv = load_advancement_table()
    sub_table = load_out_subtype_table()

    rng = np.random.default_rng(0)
    _, gi, _ = games[0]
    h, a = simulate_game(rng, pm, adv, sub_table, gi, n_sims=200, role_lookup=role_lookup)
    assert h.shape == (200,) and a.shape == (200,)
    assert (h >= 0).all() and (a >= 0).all()
    assert h.max() < 30 and a.max() < 30, f"runaway scores: max h={h.max()} a={a.max()}"
    print(f"\n  smoke: home {h.mean():.2f} ± {h.std():.2f}, away {a.mean():.2f} ± {a.std():.2f}")


@skip_if_missing
def test_runs_per_game_within_5pct():
    """The Phase 4 acceptance gate.

    Stratified-sample 200 games across 2025; n_sims=1000 per game.
    Compare simulated mean & variance of runs/team-game to actual 2025.
    """
    from v2.simulator import load_posteriors
    from v2.simulator.baserunner import load_advancement_table, load_out_subtype_table
    from v2.simulator.bullpen import build_queues_from_cache
    from v2.simulator.game_sim import simulate_game

    pa = _build_pa_frame(2025)
    role_lookup = _classify_roles_2025(pa)
    queues = build_queues_from_cache(2025)
    games = _build_game_inputs(pa, queues, role_lookup)

    pm = load_posteriors()
    adv = load_advancement_table()
    sub_table = load_out_subtype_table()

    # actual 2025 runs/team-game from cache: aggregate by (game_pk, side)
    actuals = []
    for gp, grp in pa.groupby("game_pk"):
        # we need actual runs scored. Reload with bat_score.
        pass  # computed below in one shot

    df_runs = pd.read_parquet(CACHE_2025, columns=["game_pk", "inning_topbot", "bat_score", "post_bat_score", "events"])
    df_runs = df_runs[df_runs["events"].notna() & ~df_runs["events"].isin(NON_PA_EVENTS)]
    df_runs["runs"] = (df_runs["post_bat_score"].fillna(0) - df_runs["bat_score"].fillna(0)).clip(0, 4)
    actual_per_team = (
        df_runs.groupby(["game_pk", "inning_topbot"])["runs"].sum().reset_index()
    )
    actual_runs = actual_per_team["runs"].to_numpy(dtype=np.float64)
    actual_mean = float(actual_runs.mean())
    actual_var = float(actual_runs.var())

    # stratified 200 games
    n_take = min(200, len(games))
    if len(games) > n_take:
        step = len(games) // n_take
        games = games[::step][:n_take]

    rng = np.random.default_rng(20260508)
    sim_runs = []
    t0 = time.time()
    for i, (gp, gi, _) in enumerate(games):
        h, a = simulate_game(rng, pm, adv, sub_table, gi, n_sims=1000, role_lookup=role_lookup)
        sim_runs.append(h)
        sim_runs.append(a)
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(games)} games  ({elapsed:.0f}s elapsed, {elapsed/(i+1):.2f}s/game)")
    sim_runs = np.concatenate(sim_runs)
    sim_mean = float(sim_runs.mean())
    sim_var = float(sim_runs.var())

    mean_rel = abs(sim_mean - actual_mean) / actual_mean
    var_rel = abs(sim_var - actual_var) / actual_var

    print(f"\n  games simulated: {len(games)}  samples: {len(sim_runs):,}")
    print(f"  actual:  mean={actual_mean:.4f}  var={actual_var:.4f}")
    print(f"  sim:     mean={sim_mean:.4f}  var={sim_var:.4f}")
    print(f"  rel diff: mean={mean_rel:.4f}  var={var_rel:.4f}  (5% gate)")

    assert mean_rel < 0.05, f"mean diff {mean_rel:.4f} exceeds 5% gate"
    assert var_rel < 0.05, f"variance diff {var_rel:.4f} exceeds 5% gate"
