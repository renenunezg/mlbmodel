"""build_game_rows suppresses bet flags when the lineup is not fully live.

A game scored on the top9-by-season-PA fallback (lineup_tag != "live") can swing
10+ points and flip the favorite, so its flags are unreliable. The gate mirrors
the TBD-starter gate: estimate still writes, bets are suppressed.
"""
from __future__ import annotations

import numpy as np

from v2.markets.writer import build_game_rows

_ODDS = {"moneyline": -150, "spread": -1.5, "spread_odds": 120,
         "total": 8.0, "total_over_odds": -110, "total_under_odds": -110}


def _rows(lineups_live: bool):
    # Home wins 90%: enough edge to flag at -150 (~0.60 implied), both win
    # probs nonzero so our_odds stays finite.
    home_runs = np.concatenate([np.full(900, 6), np.full(100, 2)])
    away_runs = np.concatenate([np.full(900, 3), np.full(100, 7)])
    return build_game_rows(
        game_pk=1, game_date=np.datetime64("2026-06-13"), start_time=None,
        home_team="LAD", away_team="CHW", home_starter="x", away_starter="y",
        home_runs=home_runs, away_runs=away_runs,
        home_odds=_ODDS, away_odds={**_ODDS, "moneyline": 130, "spread": 1.5},
        lineup_source="lineup_top9+queue_cache", lineups_locked=False,
        posterior_age_days=0, lineups_live=lineups_live,
    )


def test_live_lineup_keeps_pick():
    home, _ = _rows(lineups_live=True)
    assert home["ev_flag"] == "LAD"


def test_fallback_lineup_suppresses_pick():
    home, away = _rows(lineups_live=False)
    for row in (home, away):
        assert row["ev_flag"] == "No Play"
        assert row["run_line_ev_flag"] == "No Play"
        assert row["expected_runs"] > 0  # estimate still writes
