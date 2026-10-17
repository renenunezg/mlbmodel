"""Verified cash schedules supplement payroll accounting, never production estimates.

Match the entire signing identity before replacing a payroll schedule. Unknown
payment timing stays explicit. Historical scoring cannot see a later signing.
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from v2.players.value import ContractYear

DATA = Path(__file__).parent / "data"


def sources() -> list[Path]:
    return [*sorted(DATA.glob("free_agent_signings_mixed_*.json")), DATA / "contract_cash_exceptions.json"]


@lru_cache(maxsize=1)
def schedules() -> dict[tuple[int, int], dict]:
    result = {}
    for path in sources():
        for c in json.loads(path.read_text())["contracts"]:
            if c.get("has_exercisable_option", False):
                continue
            # A deferred aggregate requires explicit allocation to earned years.
            # Do not charge a new club for liabilities earned before acquisition.
            if "earned_cash" not in c and len(c["cash_musd"]) > c["years"]:
                continue
            payments = c.get("earned_cash", {str(c["first_season"]+i): [[c["first_season"]+i, v]]
                                             for i, v in enumerate(c.get("cash_musd", []))})
            total = sum(amount for flows in payments.values() for _, amount in flows)
            if not np.isclose(total, c["guarantee_musd"]) or any(
                    year < int(earned) or amount < 0 or not np.isfinite(amount)
                    for earned, flows in payments.items() for year, amount in flows):
                raise ValueError("Verified contract cash does not reconcile")
            key = (c["player_id"], c["first_season"])
            if key in result:
                raise ValueError("Duplicate verified signing identity")
            result[key] = {**c, "earned_cash": payments}
    return result


def apply(rows: pd.DataFrame, contract: list[ContractYear], as_of: date | None) -> tuple[list[ContractYear], dict]:
    metadata = {"basis": "Payroll schedule; actual cash timing not verified", "verified": False}
    if as_of is None or not {"player_id", "contract_start", "contract_years", "contract_total_musd"} <= set(rows):
        return contract, metadata
    row = rows.iloc[0]
    if pd.isna(row.contract_start):
        return contract, metadata
    c = schedules().get((int(row.player_id), int(row.contract_start)))
    if c is None or c["date"] > as_of.isoformat():
        return contract, metadata
    expected_total = c.get("payroll_summary_musd", c["guarantee_musd"])
    if row.contract_years != c["years"] or not np.isclose(row.contract_total_musd, expected_total, atol=.01):
        return contract, metadata
    result = []
    for year in contract:
        flows = c["earned_cash"].get(str(year.season))
        if flows is None:
            result.append(year)
            continue
        if year.salary_kind != "guaranteed":
            raise ValueError("Verified cash schedule conflicts with current contract rights")
        amounts = tuple((int(y), float(v)) for y, v in flows)
        updated = replace(year, salary_musd=sum(v for _, v in amounts), cashflows=amounts)
        if year.season == c.get("opt_out_before"):
            updated = replace(updated, salary_kind="buyout", salary_musd=0.0,
                              option_musd=sum(v for _, v in amounts), option_kind="player_block",
                              retention_raise_musd=c.get("retention_raise_musd", 0.0), cashflows=None)
        result.append(updated)
    return result, {"basis": "Verified payments allocated to the season earning the obligation", "verified": True,
                    "source": c.get("cash_source", c["source"]),
                    "liability_scope": c.get("liability_scope", "Past paid bonuses are not charged to future control years")}
