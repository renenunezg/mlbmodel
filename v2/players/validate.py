"""Offline workload holdout and source-backed historical market diagnostics.

Run with a model trained strictly before the market season:
    python -m v2.players.validate --model PATH --cases tests/fixtures/player_market_2025.json --output /tmp/player-validation.json

Add the offseason trade-set comparison to release evidence (models frozen before each winter):
    python -m v2.players.validate --model PATH --release-evidence EVIDENCE --trade-set \
        --trade-model 2025=PATH --trade-model 2026=PATH --trade-baseline 2025=PATH --trade-baseline 2026=PATH --output EVIDENCE_OUT

No database or publication path. Market diagnostics report implied prices and
coverage, never treat incomplete packages or a selected case series as a pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from v2.players import aging, cash, inputs, market, pricing, production, value, workload
from v2.players.market import historical_projection
from v2.players.model import ValueModel


def _error_comparison(actual: np.ndarray, prediction: np.ndarray, baseline: np.ndarray, weights: np.ndarray | None = None) -> dict:
    e, b = prediction - actual, baseline - actual
    rng = np.random.default_rng(20261017)
    ix = rng.integers(0, len(e), (5000, len(e)))
    w = np.ones(len(e)) if weights is None else weights
    return {"n": len(e), "zero_targets": int((actual == 0).sum()),
            "candidate_rmse": float(np.sqrt(np.average(e**2, weights=w))), "baseline_rmse": float(np.sqrt(np.average(b**2, weights=w))),
            "candidate_bias": float(np.average(e, weights=w)), "baseline_bias": float(np.average(b, weights=w)),
            "mse_change_95pct": np.quantile((w*(e**2-b**2))[ix].sum(axis=1)/w[ix].sum(axis=1), [.025, .975]).tolist()}


def workload_check() -> dict:
    from scipy.optimize import lsq_linear

    from v2.players import bio

    lines = workload.load_lines(list(range(workload.FIRST_HISTORY_SEASON, 2027)))
    rosters = {year: set(bio.october_roster(year-1)["player_id"]) for year in range(workload.FIRST_ORIGIN, 2027)}
    panel = workload.training_frame(lines, rosters, 2026)
    report = {"population": "100 PA/BF in any prior three seasons; absent target players remain zero",
              "baseline": "Previous recursive lag-3/age method, refitted on the same historical population",
              "injury_proxy": "Retained January 1 roster member with last-year volume below half the prior four-year peak; not a medical diagnosis",
              "roster_coverage": {str(y): len(ids) for y, ids in rosters.items()},
              "splits": {}, "production_rates": {},
              "production_scope": "Conditional volume-weighted rate errors; joint runs include zero-volume players and use the same workload on both sides. This isolates aging using the historical quality proxy, not the final trait-based grade or WAR model.",
              "limitations": ["2026 was previously inspected; this is a regression check, not a new untouched holdout",
                                             "Multi-year aging and workload errors share players across horizons; intervals below are per origin/horizon",
                                             "Workload beyond the longest completed replay remains unvalidated"]}
    # Freeze EVERY parameter before the origin, even when checking year eight.
    # Fitting through target-1 would leak seven future seasons into that replay.
    for origin in (2018, 2019, 2025, 2026):
        fitted = workload.fit(panel, origin - 1)
        history = workload.history(lines, origin, rosters[origin])
        history = history[history[list(workload.LAGS[:3])].max(axis=1) >= 100]
        age = history.drop_duplicates("player_id").set_index("player_id")["age"]
        last = min(2026, origin + 7)
        prediction = workload.predict(history, age, fitted, origin, last)
        rate_model = aging.fit_projection(panel, origin-1)
        production = aging.add_projection_shifts(prediction, history, age, rate_model, origin)
        curves = aging.fit_curves_from_cache(origin-1)
        for (year, role), rows in production.groupby(["season", "role"]):
            horizon = int(year-origin+1)
            if year == 2020 or horizon not in (2, 5, 8):
                continue
            r = rows.merge(lines[["player_id", "role", "season", "volume", "quality"]],
                           on=["player_id", "role", "season"], how="left").fillna({"volume": 0.0, "quality": 0.0})
            component = "bat_runs_600" if role == "bat" else "pitch_runs_180"
            delta = np.array([aging.apply(0, component, int(age[pid]), int(age[pid])+horizon-1, curves) for pid in r["player_id"]])
            r["baseline_rate"] = r["run_rate"] - r["run_shift"] + delta
            active = r[r["volume"] > 0]
            unit = 600 if role == "bat" else 180*aging.PA_PER_IP
            report["production_rates"][f"{origin}:{year}:{role}"] = {
                "training_through": origin-1,
                "conditional_rate": _error_comparison(active["quality"].to_numpy(), active["run_rate"].to_numpy(), active["baseline_rate"].to_numpy(), active["volume"].to_numpy()),
                "joint_runs": _error_comparison((r["quality"]*r["volume"]/unit).to_numpy(),
                                               (r["run_rate"]*r["pa"]/unit).to_numpy(), (r["baseline_rate"]*r["pa"]/unit).to_numpy())}
        baseline_rows = []
        for role in ("bat", "pit"):
            train = panel[(panel["horizon"] == 1) & (panel["target_season"] < origin) & panel["role"].eq(role)]
            def old_design(f, ages):
                lag = f[["lag1", "lag2", "lag3"]].to_numpy(float)
                return np.column_stack([lag, np.maximum(ages-30, 0)*lag.mean(axis=1)/10])
            beta = lsq_linear(old_design(train, train["age"].to_numpy()), train["actual"].to_numpy(),
                              bounds=([0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, 0])).x
            cur = history[history["role"].eq(role)].copy()
            for year in range(origin, last+1):
                pa = np.maximum(0, old_design(cur, cur["age"].to_numpy()+year-origin) @ beta)
                baseline_rows.append(cur[["player_id", "role"]].assign(season=year, baseline=pa))
                cur["lag3"], cur["lag2"], cur["lag1"] = cur["lag2"], cur["lag1"], pa
        joined = prediction.merge(pd.concat(baseline_rows), on=["player_id", "role", "season"])
        joined = joined.merge(lines[["player_id", "role", "season", "volume"]], on=["player_id", "role", "season"], how="left").fillna({"volume": 0})
        h = history.assign(drop=history["rostered"].eq(1) & (history["lag1"] < .5*history[list(workload.LAGS[1:])].max(axis=1)))
        joined = joined.merge(h[["player_id", "role", "rostered", "drop"]], on=["player_id", "role"])
        for (year, role), rows in joined.groupby(["season", "role"]):
            if year == 2020 or year-origin+1 not in (1, 2, 5, 8):
                continue
            strata = {"all": rows, "retained": rows[rows["rostered"].eq(1)], "retained_after_drop": rows[rows["drop"]]}
            report["splits"][f"{origin}:{year}:{role}"] = {
                "training_through": origin-1, "horizon": int(year-origin+1),
                **{name: _error_comparison(r["volume"].to_numpy(), r["pa"].to_numpy(), r["baseline"].to_numpy())
                   for name, r in strata.items() if len(r)}}
    return report


def production_check() -> dict:
    """Forward WAR errors on a fixed, source-defined historical population."""
    lines = production.load_lines(2026)
    panel = production.panel(lines, 2026)
    report = {"target": "FanGraphs total WAR, including negative outcomes",
              "population": "100 PA or 30 IP in any prior five seasons; absent target players are zero",
              "baseline": "5/4/3 weighted prior WAR with 0.5 WAR annual decline after year one",
              "scope": "Historical forward replays; 2025/2026 were previously inspected and are regression checks, not fresh holdouts",
              "splits": {}, "limitations": ["Current historical WAR revisions are not archived as-of data",
                   "Workload-drop cohorts are not confirmed medical diagnoses",
                   "Beyond observed validation horizons, coefficients have training support but no independent replay here"]}
    for origin in (2015, 2018, 2021, 2024, 2025, 2026):
        fitted = production.fit(panel, origin-1)
        h = production.history(lines, origin, include_roster=True, include_surgery=True, include_aaa=True)
        last = min(2026, origin+7)
        pred = production.predict(h, fitted, origin, last)
        for (year, role), members in pred.groupby(["season", "role"]):
            horizon = int(year-origin+1)
            if year in production.SHORT_SEASONS or horizon not in (1, 3, 5, 8):
                continue
            actual = lines[lines.season.eq(year)][production.KEYS+["war"]]
            r = members.merge(actual, on=production.KEYS, how="left", validate="one_to_one").fillna({"war": 0})
            r = r.merge(h, on=production.KEYS, validate="one_to_one")
            r["baseline"] = np.maximum(0, (5*r.war1+4*r.war2+3*r.war3)/12-.5*(horizon-1))
            peak = r[[f"volume{i}" for i in range(2, 6)]].max(axis=1)
            strata = {"all": r, "after_workload_drop": r[r.volume1 < .5*peak],
                      "prior_three_war": r[(5*r.war1+4*r.war2+3*r.war3)/12 >= 3]}
            report["splits"][f"{origin}:{year}:{role}"] = {
                "training_through": origin-1, "horizon": horizon,
                **{name: _error_comparison(g.war.to_numpy(), np.maximum(0, g.expected_war.to_numpy()), g.baseline.to_numpy())
                   for name, g in strata.items() if len(g)},
                "raw_bias": float((r.expected_war-r.war).mean()),
                "predicted_total": float(np.maximum(0, r.expected_war).sum()), "actual_total": float(r.war.sum())}
    # Do not turn a better simple benchmark comparison into a market release.
    report["status"] = "evaluated_not_market_validated"
    return report


def compare_serving_production(model: ValueModel, baseline: ValueModel, first: int,
                               baseline_projection: pd.DataFrame | None = None) -> dict:
    old = baseline_projection if baseline_projection is not None else historical_projection(baseline, first, first)
    new = historical_projection(model, first, first)
    actual = production.actual_war(first)
    new["baseline"] = old.war_by_season.map(lambda row: row[first])
    new["candidate"] = new.war_by_season.map(lambda row: row[first])
    new["actual"] = actual.reindex(new.index).fillna(0)
    valid = new[new.candidate.notna() & new.baseline.notna() & old.grade.notna()]
    return {"scope": "Same players graded by the prior model; both paths use only pre-season inputs",
            "eligible_previous": int(old.grade.notna().sum()), "paired": len(valid),
            "missing_candidate": int((old.grade.notna() & new.candidate.isna()).sum()),
            "missing_candidate_ids": [int(pid) for pid in new.index[old.grade.notna() & new.candidate.isna()]],
            "roles": {"pit" if pit else "bat": _error_comparison(
                rows.actual.to_numpy(), rows.candidate.to_numpy(), rows.baseline.to_numpy())
                for pit, rows in valid.groupby(valid.position_group.isin(("SP", "RP")))}}


def market_check(model: ValueModel, cases: dict, projections: pd.DataFrame | None = None) -> dict:
    first = cases["first_season"]
    last = max([first + c["years"] - 1 for c in cases["contracts"]]
               + [cy["season"] for t in cases["trades"] for side in ("left", "right")
                  for a in t[side] if a["kind"] == "mlb" for cy in a["contract"]])
    if projections is None:
        projections = historical_projection(model, first, last)
    contracts = []
    for c in cases["contracts"]:
        row = projections.loc[c["player_id"]]
        wars = np.array([row.war_by_season[y] for y in range(first, first+c["years"])], dtype=float)
        # Dollars/win is an output here, avoiding the production price's future market leakage.
        weighted_wins = float(wars @ ((1 - value.DISCOUNT_PER_YEAR) * (1 + model.price["inflation"])) ** np.arange(len(wars)))
        cash = c.get("cash_musd")
        if cash is not None:
            if not np.isclose(sum(cash), c["guarantee_musd"]):
                raise ValueError(f"Cash does not reconcile for {c['name']}")
            cost = float(np.array(cash) @ (1 - value.DISCOUNT_PER_YEAR) ** np.arange(len(cash)))
            bounds = [cost, cost]
            timing = "Reported annual cash including bonuses; discounted by calendar year"
        else:
            bounds = [c["guarantee_musd"] * (1 - value.DISCOUNT_PER_YEAR) ** (c["years"] - 1), c["guarantee_musd"]]
            timing = "Timing sensitivity only: all cash in final vs first season; deferrals not verified"
        price = model.price_model()
        replacement = pricing.replacement_present_value(first, c["years"], value.DISCOUNT_PER_YEAR) if price.above_minimum else 0.0
        contracts.append({**c, "projected_war": wars.tolist(), "playing_time_first": row.playing_time,
                          "production_pv_in_first_year_wins": weighted_wins, "salary_pv_bounds_musd": bounds,
                          "replacement_pv_musd": replacement,
                          "implied_price_bounds_musd": [(x-replacement) / weighted_wins for x in bounds] if weighted_wins > 0 else None,
                          "cash_timing": timing, "predicted_contract_pv_musd": sum(
                              price.value_for_roles(pricing.contract_roles(c, row.war_by_role[y]), y, first,
                                                    row.market_profiles[y] if row.get("market_profiles") is not None else None) * (1-value.DISCOUNT_PER_YEAR)**(y-first)
                              for y in range(first, first+c["years"]))})
    trades = []
    for t in cases["trades"]:
        left = market.package_value(t["left"], projections, model, first)
        right = market.package_value(t["right"], projections, model, first)
        complete = left["complete_accounting"] and right["complete_accounting"]
        gap = ([left["scenario_bounds_musd"][0]-right["scenario_bounds_musd"][1],
                left["scenario_bounds_musd"][1]-right["scenario_bounds_musd"][0]] if complete else None)
        trades.append({"name": t["name"], "date": t["date"], "source": t["source"], "left": left, "right": right,
                       "gap_bounds_musd": gap, "complete_accounting": complete,
                       "dollar_validation": "unresolved" if not complete else "gap" if not gap[0] <= 0 <= gap[1] else "overlapping_scenarios",
                       "scope": cases["scope"] + " Scenario bounds are not predictive intervals."})
    warnings = ["Package equality is a diagnostic, not an observed dollar label or a predictive interval",
                "Individual contract errors do not establish or disprove aggregate calibration"]
    if any(a["kind"] in ("prospect", "draft_pick") for t in cases["trades"] for side in ("left", "right") for a in t[side]):
        warnings.append("Prospect and draft values are external priors; exact historical grade timestamps are not all archived")
    return {"status": "diagnostic_complete" if all(t["complete_accounting"] for t in trades) else "incomplete",
            "training_through": max(model.dev_seasons), "inputs_through": first-1,
            "scope": cases["scope"], "price": model.price, "contracts": contracts, "trades": trades,
            "external_prior_sources": [market.PROSPECT_SOURCE, market.LOWER_PROSPECT_SOURCE, market.DRAFT_SOURCE],
            "warnings": warnings}


TRADE_SET_FIXTURES = sorted(Path("tests/fixtures").glob("player_market_trades_*.json"))
# A trade after the deadline means the coming arbitration season was already tendered.
NON_TENDER_DEADLINES = {2025: "2024-11-22", 2026: "2025-11-21"}
SUPER_TWO_DAYS = 130  # Approximate recent Super Two cutoff within the 2.xxx service class.
TRADE_SET_MIN_TRADES = 20


def _service_years(text: str) -> tuple[float, int, int]:
    """MLB 'Y.DDD' service time as decimal years (172 days a year), whole years and days."""
    years, _, days = str(text).strip().partition(".")
    if not years.isdigit() or not (days or "0").isdigit() or len(days) > 3:
        raise ValueError(f"Unreadable service time {text!r}")
    d = int(days.ljust(3, "0")) if days else 0
    return int(years) + d / 172, int(years), d


def _trade_schedule(asset: dict, first: int, trade_date: str) -> tuple[list[dict], float]:
    """Contract seasons from ``first`` until free agency, from service time and signed terms."""
    service, years, days = _service_years(asset["service_time"])
    terms = asset.get("contract") or {}
    guaranteed = {int(g["season"]): g["salary_musd"] for g in terms.get("guaranteed") or [] if int(g["season"]) >= first}
    options = {int(o["season"]): o for o in terms.get("options") or [] if int(o["season"]) >= first}
    last_signed = max([*guaranteed, *options], default=first - 1)
    super_two = years == 2 and days >= SUPER_TWO_DAYS
    rows = []
    for season in range(first, first + 16):
        before = service + season - first
        if season in guaranteed:
            if guaranteed[season] is None:
                raise ValueError(f"{asset['name']}: unknown guaranteed {season} salary")
            rows.append({"season": season, "salary_kind": "guaranteed", "salary_musd": float(guaranteed[season]), "option_musd": None})
        elif season in options:
            o = options[season]
            kind = "team" if o.get("kind", "team") == "vesting" else o.get("kind", "team")
            declined = "free_agent" if before + 1 >= 6 or kind in ("player", "mutual") else "arbitration"
            rows.append({"season": season, "salary_kind": "buyout", "salary_musd": float(o.get("buyout_musd") or 0.0),
                         "option_musd": float(o["amount_musd"]), "option_kind": kind, "declined_to": declined})
        elif season <= last_signed:
            raise ValueError(f"{asset['name']}: gap in signed seasons at {season}")
        elif before >= 6 - 1e-9:
            return rows, service
        elif before >= 3 - 1e-9 or (super_two and before >= 2):
            agreed = terms.get("arb_salary_agreed_musd") if season == first else None
            if first not in NON_TENDER_DEADLINES:
                raise ValueError(f"No non-tender deadline recorded before {first}")
            rows.append({"season": season, "salary_kind": "arbitration", "option_musd": None,
                         "salary_musd": None if agreed is None else float(agreed),
                         "tendered": season == first and (agreed is not None or trade_date > NON_TENDER_DEADLINES[first])})
        else:
            rows.append({"season": season, "salary_kind": "pre_arb", "salary_musd": None, "option_musd": None})
    raise ValueError(f"{asset['name']}: control does not end within sixteen seasons")


def trade_cases(fixture: dict) -> tuple[dict, list[dict]]:
    """Researched trades in the market_check case layout; unconvertible trades are reported, never dropped."""
    first = fixture["first_season"]
    trades, rejected = [], []
    for t in fixture["trades"]:
        sides = {}
        try:
            for side in ("left", "right"):
                sides[side] = []
                for a in t[side]:
                    if a["kind"] == "mlb":
                        if a.get("player_id") is None:
                            raise ValueError(f"MLB player without an identifier: {a['name']}")
                        contract, service = _trade_schedule(a, first, t["date"])
                        sides[side].append({"kind": "mlb", "name": a["name"], "player_id": int(a["player_id"]),
                                            "service_time": service, "service_season": first, "contract": contract})
                    elif a["kind"] == "prospect":
                        if a["role"] not in market.PROSPECT_MUSD or (a["fv"] != "35+" and a["fv"] not in market.PROSPECT_MUSD[a["role"]]):
                            raise ValueError(f"Prospect grade outside the published table: {a['name']} {a['fv']}")
                        sides[side].append({"kind": "prospect", "name": a["name"], "player_id": a.get("player_id"),
                                            "fv": a["fv"], "role": a["role"]})
                    elif a["kind"] == "cash":
                        sides[side].append({"kind": "cash", "name": a["name"], "cash_musd": [float(x) for x in a["cash_musd"]]})
                    elif a["kind"] == "draft_pick":
                        sides[side].append({"kind": "draft_pick", "name": a["name"], "pick_range": [int(x) for x in a["pick_range"]]})
                    else:
                        raise ValueError(f"Unknown trade asset kind {a['kind']}")
        except ValueError as exc:
            rejected.append({"name": t["name"], "first_season": first, "reason": str(exc)})
            continue
        trades.append({"name": t["name"], "date": t["date"], "source": t["source"], **sides})
    if len({t["name"] for t in trades}) != len(trades):
        raise ValueError(f"Duplicate trade names in the {first} trade set")
    return {"first_season": first, "scope": fixture["selection"], "contracts": [], "trades": trades}, rejected


def trade_set_check(models: dict[int, ValueModel], baselines: dict[int, ValueModel], fixtures: list[dict]) -> dict:
    """Paired package-balance errors on every qualifying offseason trade.

    Each winter is priced by models frozen before it. A fair trade balances, so
    the relative gap |left - right| / (|left| + |right|) cannot be lowered by
    shrinking every value toward zero. Trades without an MLB player only test
    the external prospect table and are reported, not scored.
    """
    rows, rejected = [], []
    for fixture in fixtures:
        first = fixture["first_season"]
        cases, skipped = trade_cases(fixture)
        rejected += skipped
        reports = {label: {t["name"]: t for t in market_check(m[first], cases)["trades"]}
                   for label, m in (("candidate", models), ("baseline", baselines))}
        for case in cases["trades"]:
            priced = {label: reports[label][case["name"]] for label in reports}
            if not all(p["complete_accounting"] for p in priced.values()):
                missing = sorted({n for p in priced.values() for s in ("left", "right") for n in p[s]["unpriced_assets"]})
                rejected.append({"name": case["name"], "first_season": first, "reason": "Unpriced: " + ", ".join(missing)})
                continue
            if not any(a["kind"] == "mlb" for s in ("left", "right") for a in case[s]):
                rejected.append({"name": case["name"], "first_season": first, "reason": "No MLB player; tests only the external prospect table"})
                continue
            row = {"name": case["name"], "first_season": first}
            for label, p in priced.items():
                left, right = (float(np.mean(p[s]["scenario_bounds_musd"])) for s in ("left", "right"))
                size = abs(left) + abs(right)
                row |= {f"{label}_left_musd": left, f"{label}_right_musd": right, f"{label}_gap_musd": left - right,
                        f"{label}_relative_gap": abs(left - right) / size if size > 0 else 0.0}
            rows.append(row)
    if not rows:
        raise ValueError("No trade in the set could be priced")
    frame = pd.DataFrame(rows)
    change = (frame.candidate_relative_gap - frame.baseline_relative_gap).to_numpy()
    rng = np.random.default_rng(20261104)
    draws = change[rng.integers(0, len(change), (10000, len(change)))].mean(axis=1)
    return {"n": len(frame), "rejected": rejected, "trades": rows,
            "candidate_relative_gap": float(frame.candidate_relative_gap.mean()),
            "baseline_relative_gap": float(frame.baseline_relative_gap.mean()),
            "relative_gap_change_95pct": np.quantile(draws, [.025, .975]).tolist(),
            "candidate_gap_rmse_musd": float(np.sqrt(np.mean(frame.candidate_gap_musd**2))),
            "baseline_gap_rmse_musd": float(np.sqrt(np.mean(frame.baseline_gap_musd**2))),
            "by_season": {str(s): {"n": len(g), "candidate_relative_gap": float(g.candidate_relative_gap.mean()),
                                   "baseline_relative_gap": float(g.baseline_relative_gap.mean())}
                          for s, g in frame.groupby("first_season")},
            "selection": [f["selection"] for f in fixtures]}


def assess_release(model: ValueModel, evidence: dict) -> dict:
    """Require joint production, cash, salary and trade regression improvements.

    Absolute trade-package equality can reward a model that prices both sides
    near zero. Trades therefore supplement actual fWAR and salary targets; they
    never replace them. Their paired gap comparison is descriptive with a small
    sample, and individual discrepancies remain visible in the release evidence.
    The previously inspected contract subset is a regression check, not a new
    independent significance test on a second, overlapping population.
    """
    production = [row for year in evidence["production"].values() for row in year.values()]
    contracts, original = evidence["contracts"], evidence["original_contracts"]
    trades = evidence["trades"]
    repairs = evidence["required_trade_repairs"]
    arbitration = evidence["arbitration"]
    scoring = evidence["current_scoring"]
    trade_set = evidence["trade_set"]
    checks = {
        "fwar_improves_on_each_paired_role_and_year": bool(production) and all(
            r["mse_change_95pct"][1] < 0 for r in production),
        "broad_contract_error_and_bias_pass": contracts["status"] == "passed",
        "original_contract_regression_removed": original["paired_count"] > 0 and not original["lost_coverage"]
            and original["candidate_rmse_musd"] < original["baseline_rmse_musd"],
        "later_arbitration_salary_accuracy_improves": arbitration["mse_change_95pct"][1] < 0,
        "all_trade_assets_accounted": bool(trades["cases"]) and all(
            t[s]["complete_accounting"] for t in trades["cases"] for s in ("left", "right")),
        "paired_trade_gap_error_improves": trades["comparison"]["n"] > 0
            and trades["comparison"]["candidate_rmse"] < trades["comparison"]["baseline_rmse"],
        "identified_trade_regressions_improve": bool(repairs) and all(
            r["candidate_gap_musd"] < r["previous_candidate_gap_musd"] for r in repairs),
        "complete_scoring_population": scoring["players"] >= 500 and scoring["graded_without_war"] == 0
            and scoring["grade_changes"] == 0,
        "offseason_trade_balance_no_worse": trade_set["n"] >= TRADE_SET_MIN_TRADES
            and trade_set["candidate_relative_gap"] <= trade_set["baseline_relative_gap"],
    }
    return {"status": "validated" if all(checks.values()) else "not_validated", "checks": checks,
            "model_fingerprint": model.validation_fingerprint(),
            "scope": "Joint numerical improvement against the serving pipeline; individual trade gaps remain diagnostic",
            "limitations": ["Previously inspected 2025/2026 production and FA contracts are regression replays, not fresh untouched holdouts",
                            "Trade comparisons use scenario bounds and only a small set of deals; aggregate improvement is not statistically established",
                            "Future arbitration costs are estimates; quoted salaries and guarantees take precedence"],
            "metrics": {"contract_rmse_musd": contracts["candidate_rmse_musd"],
                        "original_contract_rmse_musd": original["candidate_rmse_musd"],
                        "arbitration_rmse_musd": arbitration["candidate_rmse"],
                        "trade_gap_rmse_musd": trades["comparison"]["candidate_rmse"],
                        "trade_set_relative_gap": trade_set["candidate_relative_gap"]}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--cases", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--release-evidence", type=Path, help="Complete paired evaluation evidence for assess_release")
    ap.add_argument("--validated-model-output", type=Path, help="Save a model bound to passing release evidence")
    ap.add_argument("--calibration-cohort", type=Path, default=Path(__file__).parent / "data/free_agent_signings_2025.json")
    ap.add_argument("--holdout", type=Path, default=Path("tests/fixtures/player_market_holdout_2026.json"))
    ap.add_argument("--holdout-model", type=Path, help="Model frozen before the holdout; defaults to --model")
    ap.add_argument("--baseline-model", type=Path, help="Prior production model frozen before the calibration season")
    ap.add_argument("--baseline-holdout-model", type=Path, help="Prior production model frozen before the later season")
    ap.add_argument("--additional-trades", type=Path, default=Path("tests/fixtures/player_market_additional_trades_2025.json"))
    ap.add_argument("--baseline-report", type=Path, help="Previous complete validation for a paired end-to-end dollar comparison")
    ap.add_argument("--calibration-history", type=Path,
                    help="JSON list of {cohort, model} paths, each model frozen before that signing winter")
    ap.add_argument("--baseline-holdout-projections", type=Path,
                    help="Trusted local pickle of the serving pipeline replay; requires --baseline-holdout-model for its price")
    ap.add_argument("--baseline-projections", type=Path, help="Trusted local serving replay for the earlier season")
    ap.add_argument("--trade-set", action="store_true",
                    help="Add the offseason trade-set comparison to --release-evidence and write the complete evidence to --output")
    ap.add_argument("--trade-model", action="append", default=[], metavar="SEASON=PATH",
                    help="Candidate model frozen before that offseason; one per trade-set season")
    ap.add_argument("--trade-baseline", action="append", default=[], metavar="SEASON=PATH",
                    help="Serving model frozen before that offseason; one per trade-set season")
    args = ap.parse_args()
    if args.trade_set:
        if not args.release_evidence:
            ap.error("--trade-set extends --release-evidence")
        fixtures = [json.loads(p.read_text()) for p in TRADE_SET_FIXTURES]
        seasons = {f["first_season"] for f in fixtures}

        def frozen(specs: list[str]) -> dict[int, ValueModel]:
            loaded = {int(s): ValueModel.load(Path(p)) for s, _, p in (x.partition("=") for x in specs)}
            if set(loaded) != seasons:
                ap.error(f"Trade-set models must cover exactly the seasons {sorted(seasons)}")
            return loaded

        evidence = json.loads(args.release_evidence.read_text())
        evidence["trade_set"] = trade_set_check(frozen(args.trade_model), frozen(args.trade_baseline), fixtures)
        evidence["sha256"] = {**evidence.get("sha256", {}),
                              **{str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [*TRADE_SET_FIXTURES, *(Path(x.partition("=")[2]) for x in args.trade_model + args.trade_baseline)]}}
        args.output.write_text(json.dumps(evidence, indent=2, allow_nan=False))
        print(json.dumps({k: v for k, v in evidence["trade_set"].items() if k not in ("trades", "rejected", "selection")}, indent=2))
        return
    model = ValueModel.load(args.model)
    if args.release_evidence:
        evidence = json.loads(args.release_evidence.read_text())
        review = assess_release(model, evidence)
        review["evidence_sha256"] = hashlib.sha256(args.release_evidence.read_bytes()).hexdigest()
        args.output.write_text(json.dumps({"release": review, "evidence": evidence}, indent=2, allow_nan=False))
        print(json.dumps(review, indent=2))
        if args.validated_model_output:
            model.evaluation["market"] = review
            model.require_market_validation()
            model.save(args.validated_model_output)
        return
    if not args.cases or args.validated_model_output:
        ap.error("Diagnostic evaluation requires --cases; model approval requires --release-evidence")
    cases = json.loads(args.cases.read_text())
    cohort = json.loads(args.calibration_cohort.read_text())
    last = cases["first_season"] + max(c["years"] for c in cases["contracts"]) - 1
    projections = historical_projection(model, cases["first_season"], last)
    held_model = ValueModel.load(args.holdout_model) if args.holdout_model else ValueModel.from_payload(model.to_payload())
    held_cohort = json.loads(args.holdout.read_text())
    history_paths = []
    if args.calibration_history:
        manifests = json.loads(args.calibration_history.read_text())
        history_paths = [Path(item[key]) for item in manifests for key in ("cohort", "model")]
        cohorts = [json.loads(Path(item["cohort"]).read_text()) for item in manifests]
        fitted = {c["first_season"]: ValueModel.load(Path(item["model"])) for c, item in zip(cohorts, manifests)}
        if any(c["first_season"] >= held_cohort["first_season"] for c in cohorts):
            raise ValueError("Calibration history includes market evaluation season")
        case_price, case_calibration = market.calibrate_history([c for c in cohorts if c["first_season"] < cases["first_season"]], fitted)
        price, calibration = market.calibrate_history(cohorts, fitted)
        from v2.players.arbitration import calibrate as calibrate_arbitration

        model.price, held_model.price = asdict(case_price), asdict(price)
        model.arbitration, case_calibration["arbitration"] = calibrate_arbitration(
            {y: m for y, m in fitted.items() if y < cases["first_season"]})
        held_model.arbitration, calibration["arbitration"] = calibrate_arbitration(fitted)
        calibration_cutoff = calibration["cutoff"]
    else:
        price, calibration = market.calibrate(model, cohort, projections)
        model.price = held_model.price = asdict(price)
        case_calibration = calibration
        calibration_cutoff = cohort.get("information_cutoff", cohort["cutoff"])
    baseline_projections = pd.read_pickle(args.baseline_holdout_projections) if args.baseline_holdout_projections else None
    if baseline_projections is not None and not args.baseline_holdout_model:
        raise ValueError("Serving projections require their baseline model")
    baseline_price = ValueModel.load(args.baseline_holdout_model).price_model() if baseline_projections is not None else None
    report = {"production": production_check() if model.production else workload_check(), "calibration": calibration,
              "case_calibration": case_calibration, "market": market_check(model, cases, projections),
              "contract_holdout": market.contract_holdout(held_model, held_cohort, calibration_cutoff,
                  baseline_projections=baseline_projections, baseline_price=baseline_price)}
    if args.baseline_model and args.baseline_holdout_model:
        report["previous_production_comparison"] = {
            str(cases["first_season"]): compare_serving_production(model, ValueModel.load(args.baseline_model), cases["first_season"],
                pd.read_pickle(args.baseline_projections) if args.baseline_projections else None),
            str(held_cohort["first_season"]): compare_serving_production(held_model, ValueModel.load(args.baseline_holdout_model),
                held_cohort["first_season"], baseline_projections)}
    report["additional_trades"] = market_check(model, json.loads(args.additional_trades.read_text()), projections)
    calibration_ids = {r["player_id"] for r in case_calibration["contracts"]}
    for case in report["market"]["contracts"]:
        case["used_for_price_calibration"] = case["player_id"] in calibration_ids
    # This diagnostic run does not include arbitration or complete population
    # checks. Only assess_release can authorize a model for publication.
    report["publishable"] = False
    if args.baseline_report:
        previous = json.loads(args.baseline_report.read_text())["contract_holdout"]["contracts"]
        before = {(r["player_id"], r["first_season"]): r for r in previous}
        after = {(r["player_id"], r["first_season"]): r for r in report["contract_holdout"]["contracts"]}
        keys = sorted(before.keys() & after.keys())
        report["contract_comparison_coverage"] = {"same_contracts": len(keys),
            "newly_covered": [after[k]["name"] for k in after.keys()-before.keys()],
            "lost_coverage": [before[k]["name"] for k in before.keys()-after.keys()]}
        if not keys:
            raise ValueError("Contract comparison has no common population")
        actual = np.array([after[k]["salary_pv_musd"] for k in keys])
        if not np.allclose(actual, [before[k]["salary_pv_musd"] for k in keys]):
            raise ValueError("Contract comparison cash targets differ")
        report["previous_candidate_comparison"] = _error_comparison(
            actual, np.array([after[k]["predicted_pv_musd"] for k in keys]),
            np.array([before[k]["predicted_pv_musd"] for k in keys]))
        comparison = report["previous_candidate_comparison"]
        comparison["status"] = "passed" if comparison["mse_change_95pct"][1] < 0 and not report["contract_comparison_coverage"]["lost_coverage"] else "not_improved_reliably"
        # Beating an old price on candidate WAR is not the same as beating the
        # prior complete pipeline. Keep that stronger regression gate explicit.
        report["publishable"] &= comparison["status"] == "passed"
    source_paths = [args.model, args.cases, args.calibration_cohort, args.holdout, args.additional_trades,
                    *history_paths, *([args.calibration_history] if args.calibration_history else []),
                    *([args.baseline_holdout_projections] if args.baseline_holdout_projections else []),
                    *([args.baseline_projections] if args.baseline_projections else []),
                    *production.source_paths(2026),
                    *cash.sources(),
                    *[p for p in sorted(Path(__file__).parent.glob("*.py")) if " " not in p.name],
                    *([args.holdout_model] if args.holdout_model else []),
                    *([args.baseline_model] if args.baseline_model else []),
                    *([args.baseline_holdout_model] if args.baseline_holdout_model else []),
                    *([args.baseline_report] if args.baseline_report else []),
                    *[inputs.CACHE_DIR / f"outcomes_{s}.parquet" for s in range(2021, 2027)],
                    *[inputs.CACHE_DIR / f"inputs_{s}.parquet" for s in range(2021, 2026)],
                    *[inputs.CACHE_DIR / f"season_lines_{g}.parquet" for g in ("hitting", "pitching")],
                    *sorted(inputs.CACHE_DIR.glob("league_*.parquet")), *sorted(inputs.CACHE_DIR.glob("roster_jan1_*.parquet")),
                    *sorted(inputs.CACHE_DIR.glob("roster_oct1_*.parquet")), *sorted(inputs.CACHE_DIR.glob("aaa_season_*.json")),
                    inputs.CACHE_DIR / "tommy_john_surgeries.csv"]
    report["sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"price": asdict(price), "contract_holdout": {k:v for k,v in report["contract_holdout"].items() if k not in ("contracts", "excluded")},
                      "market": report["market"]["status"], "publishable": report["publishable"]}, indent=2))


if __name__ == "__main__":
    main()
