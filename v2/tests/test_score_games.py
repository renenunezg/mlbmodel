"""Smoke test: end-to-end Phase 5 scoring on a real date.

Uses the live Supabase (read-only against games + probable_starters + odds) and
the 2026 statcast cache. Skips if posteriors aren't built. Always runs with
write=False so it doesn't touch model_outputs.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from v2.bayesian._common import POSTERIORS_DIR

POSTERIORS_PRESENT = (POSTERIORS_DIR / "batter_skill.nc").exists() and (
    POSTERIORS_DIR / "pitcher_skill.nc"
).exists() and (POSTERIORS_DIR / "park_effects.nc").exists()

CACHE_2026 = Path(__file__).resolve().parents[2] / "cache" / "statcast_2026.parquet"


SMOKE_DATE = "2026-04-15"
N_SIMS = 1000


@pytest.mark.skipif(not POSTERIORS_PRESENT, reason="posteriors not built")
@pytest.mark.skipif(not CACHE_2026.exists(), reason="2026 statcast cache missing")
def test_score_games_end_to_end():
    from v2.pipeline.score_games import score

    df = score(SMOKE_DATE, n_sims=N_SIMS, write=False, seed=0)
    if df.empty:
        pytest.skip("no games for SMOKE_DATE; pick a different date")

    # Two rows per game.
    games = df["game_pk"].unique()
    assert len(df) == 2 * len(games), f"expected 2 rows per game, got {len(df)} rows for {len(games)} games"

    # Per-game invariants.
    for gp in games:
        g = df[df.game_pk == gp]
        assert len(g) == 2
        # win_prob sums to 1
        assert abs(float(g["win_prob"].sum()) - 1.0) < 1e-6, f"win_prob sum != 1 for game {gp}"
        # our_total = sum of expected_runs
        et_sum = float(g["expected_runs"].sum())
        ot = float(g["our_total"].iloc[0])
        assert abs(et_sum - ot) < 0.01, f"our_total {ot} != sum(expected_runs) {et_sum}"
        # percentile ordering
        for col_root in ("expected_runs", "total"):
            for _, row in g.iterrows():
                p10 = row[f"{col_root}_p10"]
                p50 = row[f"{col_root}_p50"]
                p90 = row[f"{col_root}_p90"]
                assert p10 <= p50 <= p90, f"{col_root} percentile ordering violated"

    # Kelly bounds + numeric finiteness on key cols.
    for col in ("kelly_full_ml", "kelly_quarter_ml", "kelly_full_rl", "kelly_quarter_rl",
                "kelly_full_total", "kelly_quarter_total"):
        vals = df[col].dropna()
        assert (vals >= 0).all() and (vals <= 1).all(), f"{col} out of [0,1]"

    # win_prob in [0,1]
    assert (df["win_prob"] >= 0).all() and (df["win_prob"] <= 1).all()

    # win_prob_p10/p90 from per-posterior-draw sampling. Should be populated,
    # finite, in [0,1], ordered, and anti-correlated across the home/away pair.
    assert df["win_prob_p10"].notna().all() and df["win_prob_p90"].notna().all()
    assert (df["win_prob_p10"] >= 0).all() and (df["win_prob_p90"] <= 1).all()
    assert (df["win_prob_p10"] <= df["win_prob_p90"]).all(), "win_prob_p10 must be <= p90"
    for gp in games:
        rows = list(df[df.game_pk == gp].itertuples(index=False))
        assert abs((rows[0].win_prob_p10 + rows[1].win_prob_p90) - 1.0) < 1e-3
        assert abs((rows[0].win_prob_p90 + rows[1].win_prob_p10) - 1.0) < 1e-3

    # ev_flag / total_play / run_line_ev_flag are strings, never null
    for col in ("ev_flag", "total_play", "run_line_ev_flag", "high_variance_flag"):
        assert df[col].notna().all(), f"{col} has nulls"


