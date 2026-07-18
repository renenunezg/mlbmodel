"""Tests for temporary Statcast fetch failures."""
from __future__ import annotations

import pandas as pd
import pytest

from v2.data.build_cache import _fetch_statcast


def test_fetch_statcast_retries_then_returns_frame():
    calls = 0
    sleeps: list[float] = []
    expected = pd.DataFrame({"game_pk": [1]})

    def fetcher(*, start_dt: str, end_dt: str) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        assert (start_dt, end_dt) == ("2026-07-17", "2026-07-18")
        if calls == 1:
            raise ConnectionError("Statcast unavailable")
        return expected

    result = _fetch_statcast(fetcher, "2026-07-17", "2026-07-18", sleeps.append)

    assert result is expected
    assert calls == 2
    assert sleeps == [5]


def test_fetch_statcast_raises_after_bounded_retries():
    calls = 0
    sleeps: list[float] = []

    def fetcher(*, start_dt: str, end_dt: str) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        raise ConnectionError("Statcast unavailable")

    with pytest.raises(ConnectionError, match="Statcast unavailable"):
        _fetch_statcast(fetcher, "2026-07-17", "2026-07-18", sleeps.append)

    assert calls == 3
    assert sleeps == [5, 10]
