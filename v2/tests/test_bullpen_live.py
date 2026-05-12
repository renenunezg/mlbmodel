"""Unit tests for build_queues_live."""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

from v2.simulator.bullpen import (
    ELIG_OUTS_1D,
    ELIG_OUTS_2D,
    BullpenQueue,
    LiveQueueContext,
    build_queues_live,
)


class _FakeConn:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _FakeEngine:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def begin(self):
        return _FakeConn(self._df)


@pytest.fixture(autouse=True)
def _patch_read_sql(monkeypatch):
    """Route _load_workload's pd.read_sql through the fake engine's df."""
    def _fake_read_sql(_sql, conn, params=None):
        df = conn._df.copy()
        if params and "teams" in params:
            df = df[df["team"].isin(params["teams"])]
        if params and "lo" in params:
            df = df[(df["game_date"] >= params["lo"]) & (df["game_date"] < params["hi"])]
        return df.reset_index(drop=True)
    monkeypatch.setattr("v2.simulator.bullpen.pd.read_sql", _fake_read_sql)


def _wl(rows):
    return pd.DataFrame(rows, columns=["game_date", "pitcher_id", "team", "outs", "role"])


def test_eligibility_drops_overworked():
    today = date(2026, 5, 11)
    workload = _wl([
        (today - timedelta(days=1), 100, "NYY", ELIG_OUTS_1D, "RP"),    # dropped (1d cap)
        (today - timedelta(days=1), 101, "NYY", ELIG_OUTS_2D, "RP"),    # dropped (2d cap)
        (today - timedelta(days=1), 102, "NYY", 3, "RP"),               # eligible
    ])
    engine = _FakeEngine(workload)
    ctx = LiveQueueContext(game_pk=1, side="home", team="NYY", starter_id=999)
    with patch("backend.data.mlb_api.fetch_active_pitchers", return_value=[100, 101, 102, 999]):
        out = build_queues_live(today, [ctx], engine=engine)
    q = out[(1, "home")]
    assert q.starter == 999
    assert q.relievers == [102]


def test_orders_freshest_first():
    today = date(2026, 5, 11)
    yesterday = today - timedelta(days=1)
    two_ago = today - timedelta(days=2)
    workload = _wl([
        (yesterday, 200, "BOS", 3, "RP"),     # 2d total = 3
        (two_ago,   201, "BOS", 5, "RP"),     # 2d total = 5
        (yesterday, 202, "BOS", 1, "RP"),     # 2d total = 1 (freshest)
    ])
    engine = _FakeEngine(workload)
    ctx = LiveQueueContext(game_pk=5, side="away", team="BOS", starter_id=999)
    with patch("backend.data.mlb_api.fetch_active_pitchers", return_value=[200, 201, 202, 999]):
        out = build_queues_live(today, [ctx], engine=engine)
    assert out[(5, "away")].relievers == [202, 200, 201]


def test_excludes_starter_from_relievers():
    today = date(2026, 5, 11)
    engine = _FakeEngine(_wl([]))
    ctx = LiveQueueContext(game_pk=7, side="home", team="LAD", starter_id=555)
    with patch("backend.data.mlb_api.fetch_active_pitchers", return_value=[555, 600, 601]):
        out = build_queues_live(today, [ctx], engine=engine)
    q = out[(7, "home")]
    assert q.starter == 555
    assert 555 not in q.relievers
    assert set(q.relievers) == {600, 601}


def test_empty_roster_omits_side():
    today = date(2026, 5, 11)
    engine = _FakeEngine(_wl([]))
    ctx = LiveQueueContext(game_pk=9, side="home", team="LAD", starter_id=555)
    with patch("backend.data.mlb_api.fetch_active_pitchers", return_value=[]):
        out = build_queues_live(today, [ctx], engine=engine)
    assert (9, "home") not in out
