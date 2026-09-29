"""Synthetic recovery gates for the three Bayesian skill models."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests._synthetic_pa import synth_batter_pa, synth_pitcher_pa
from v2.bayesian import batter_skill, park_effects, pitcher_skill


def test_training_readers_exclude_postseason_and_wrong_year_cache_rows(monkeypatch, tmp_path):
    from datetime import date

    from backend.data import fangraphs
    from v2.data import build_cache, pa_dataset
    from v2.pipeline import score_games, write_posterior_summaries
    from v2.simulator import build_advancement_table, gb_quartiles

    rows = []
    for player in range(4):
        for pa in range(4):
            rows.append({
                "game_pk": 1, "game_date": "2026-09-27", "game_type": "R",
                "batter": 101 + player, "pitcher": 201 + player,
                "stand": "R", "p_throws": "R", "home_team": "LAD", "away_team": "SD",
                "balls": 0, "strikes": 2, "events": "field_out", "inning": 1,
                "inning_topbot": "Top", "launch_speed": 80., "launch_angle": 10.,
                "at_bat_number": 4 * player + pa + 1, "pitch_number": 1, "outs_when_up": 2,
                "on_1b": None, "on_2b": None, "on_3b": None, "bat_score": 0, "post_bat_score": 0,
                "bb_type": "ground_ball" if pa < player else "fly_ball",
            })
    regular = pd.DataFrame(rows)
    playoff = pd.concat([regular.assign(game_type=kind, game_pk=i + 10, batter=999, pitcher=998)
                         for i, kind in enumerate(("F", "D", "L", "W", "S"))], ignore_index=True)
    following_year = regular.assign(game_pk=2, game_date="2027-04-01", batter=301, pitcher=401)
    mixed = pd.concat([regular, playoff, following_year], ignore_index=True)
    for year in (2026, 2027):
        mixed.to_parquet(tmp_path / f"statcast_{year}.parquet", index=False)
    for module in (pa_dataset, build_cache, score_games, write_posterior_summaries,
                   build_advancement_table, gb_quartiles, fangraphs):
        monkeypatch.setattr(module, "CACHE_DIR", tmp_path)

    trained = pa_dataset.load_pa_dataset(2026, 2027)
    assert len(trained) == len(regular) + len(following_year)
    assert set(trained.batter) == {101, 102, 103, 104, 301}
    assert set(build_advancement_table._load_pa_rows([2026]).batter) == {101, 102, 103, 104}
    monkeypatch.setattr(gb_quartiles, "MIN_BIP", 1)
    assert set(gb_quartiles.build_gb_quartiles([2026]).player_id) == {101, 102, 103, 104, 201, 202, 203, 204}
    assert set(score_games.load_cache_for_year(2027).batter) == {301}
    assert set(write_posterior_summaries._load_window_pa(date(2026, 9, 1), date(2026, 10, 1)).batter) == {101, 102, 103, 104}

    fetched = []

    def fetch(_fetcher, start, end):
        fetched.append((start, end))
        return mixed

    monkeypatch.setattr(build_cache, "_fetch_statcast", fetch)
    cached = build_cache.fetch_year(2026)
    assert fetched[0][0] == "2026-09-27"  # A 2027 contaminant cannot advance this cursor.
    pd.testing.assert_frame_equal(cached.reset_index(drop=True), regular.reset_index(drop=True))
    assert pd.read_parquet(tmp_path / "statcast_2026.parquet").game_type.eq("R").all()
    with pytest.raises(ValueError, match="lacks game_type"):
        pa_dataset.transform_pitch_frame(mixed.drop(columns="game_type"))

    class WinterDate(date):
        @classmethod
        def today(cls):
            return cls(2027, 1, 15)

    # The year-round workflow includes 2027 in January, before pitches exist.
    # It must create a typed empty file, not request an inverted date range or
    # copy 2026's observations into the 2027 training input.
    (tmp_path / "statcast_2027.parquet").unlink()
    monkeypatch.setattr(build_cache, "date", WinterDate)
    assert build_cache.fetch_year(2027).empty
    assert len(fetched) == 1
    assert len(pa_dataset.load_pa_dataset(2026, 2027)) == len(regular)
    assert write_posterior_summaries._load_window_pa(date(2027, 1, 1), date(2027, 1, 15)).empty


def test_batter_model_recovers_platoon_direction():
    pa, truth = synth_batter_pa(n_batters=40, pa_per_cell_mean=150, seed=42)
    idata, _, _ = batter_skill.fit(
        pa, draws=300, tune=300, chains=2, target_accept=0.9, random_seed=0
    )
    posterior = idata.posterior
    beta_platoon = (
        posterior["sigma_platoon"].mean(("chain", "draw")).values
        * posterior["z_platoon"].mean(("chain", "draw")).values
    )

    assert np.corrcoef(truth["platoon"][:, 0], beta_platoon[:, 0])[0, 1] > 0.3


def test_pitcher_model_recovers_role_widths():
    # Preserve legitimate two-way pitching while removing position-player innings.
    frame = pd.DataFrame({"batter": [660271] * 60 + [123] * 60 + [999] * 2,
                          "pitcher": [999] * 120 + [660271, 123]})
    filtered, dropped = pitcher_skill.filter_position_player_pitching(frame)
    assert 660271 not in dropped and 123 in dropped
    assert 660271 in set(filtered.pitcher)
    pa, _ = synth_pitcher_pa(
        n_sp=20,
        n_rp=20,
        pa_per_sp=300,
        pa_per_rp=80,
        seed=7,
    )
    idata, _, _ = pitcher_skill.fit(
        pa, draws=300, tune=300, chains=2, target_accept=0.9, random_seed=0
    )
    sigma_pitcher = idata.posterior["sigma_pitcher"].mean(("chain", "draw")).values

    assert sigma_pitcher[0].mean() > sigma_pitcher[1].mean()


def test_park_model_recovers_synthetic_signal():
    rng = np.random.default_rng(0)
    true_log_pf = np.array([0.10, -0.05, 0.0, -0.07, 0.04, -0.06])
    from scipy.special import softmax

    from v2.bayesian.park_effects import PARK_GRID, WOBA_VEC
    base = np.array([.22, .085, .011, .14, .045, .005, .034, .46])
    base /= base.sum()
    curves = np.array([softmax(np.log(base) + x * WOBA_VEC) @ WOBA_VEC for x in PARK_GRID])
    actual = np.array([softmax(np.log(base) + x * WOBA_VEC) @ WOBA_VEC for x in true_log_pf])
    venue_df = pd.DataFrame({
        "home_team": ["COL", "LAD", "NYY", "SDP", "BOS", "MIA"],
        "observed_woba": actual + rng.normal(0, 0.001, len(true_log_pf)),
        "response_curve": [curves] * len(true_log_pf),
        "resid_var": np.full(len(true_log_pf), 0.27),
        "n": np.full(len(true_log_pf), 20000),
    })
    idata, _, _ = park_effects.fit(
        venue_df, draws=400, tune=400, chains=2, target_accept=0.9, random_seed=0
    )
    estimated = idata.posterior["park_log"].mean(("chain", "draw")).values

    assert (np.abs(estimated - true_log_pf) < 0.06).all()
