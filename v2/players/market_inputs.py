"""Dated relief usage inputs for contract pricing, separate from fWAR."""
from __future__ import annotations

import numpy as np
import pandas as pd

from v2.players import bio, workload

PROFILE_COLUMNS = ["expected_saves", "bat_pa", "pit_pa"]


def add_profiles(playing_time: pd.DataFrame, first: int) -> pd.DataFrame:
    """Scale prior relief usage by each year's expected MLB workload.

    Saves identify the relief role that the salary market pays for, without
    altering fWAR. Use only the previous three completed seasons. Missing MLB
    history means no observed closer role, not an invented WAR forecast.
    """
    stats = bio.league_season_lines(list(range(first-3, first)), "pitching")
    stats = stats.groupby("player_id")[["saves", "battersFaced"]].sum()
    save_rate = stats.saves / stats.battersFaced.clip(lower=1)
    result = playing_time.copy()
    result["expected_saves"] = result.pa * result.player_id.map(save_rate).fillna(0) * result.role.eq("pit")
    result["bat_pa"] = result.pa * result.role.eq("bat")
    result["pit_pa"] = result.pa * result.role.eq("pit")
    if not np.isfinite(result[PROFILE_COLUMNS].to_numpy()).all():
        raise ValueError("Non-finite market usage profile")
    return result


def historical_profiles(fitted: dict, first: int, last: int) -> pd.DataFrame:
    """Use an October roster and a workload fit completed before the winter."""
    if fitted.get("last_target_season", first) >= first:
        raise ValueError("Market workload forecast overlaps signing outcomes")
    if fitted.get("roster_date") != "October 1 before forecast origin":
        raise ValueError("Market workload training requires pre-winter roster snapshots")
    h = workload.history(workload.load_lines(list(range(first-5, first))), first,
                         set(bio.october_roster(first-1).player_id))
    ages = h.drop_duplicates("player_id").set_index("player_id").age
    return add_profiles(workload.predict(h, ages, fitted, first, last), first)
