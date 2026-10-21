"""Source-backed contract calibration and market replay using the serving projections."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from v2.players import aging, bio, inputs, market_inputs, pricing, production, score, value, workload
from v2.players.model import ValueModel


def cached_lines() -> pd.DataFrame:
    return pd.concat([pd.read_parquet(bio.CACHE_DIR / f"season_lines_{name}.parquet").assign(role=role)
                      for name, role in (("hitting", "bat"), ("pitching", "pit"))], ignore_index=True)


def historical_projection(model: ValueModel, first: int, last: int) -> pd.DataFrame:
    from v2.players import fit

    model.validate(first)
    if max(model.dev_seasons) >= first or model.workload["last_target_season"] >= first:
        raise ValueError("Historical market replay overlaps training")
    # Do not load target-season inputs or filter on today's roster.
    seasons = list(range(first - 3, first))
    data = fit.SeasonData(seasons)
    win, counts, groups, prior = data.window(first - 1, model)
    rates = inputs.shrunk_rates(counts, groups, model.shrink_strength, prior)
    lines = cached_lines()
    prior = lines[lines["season"] < first].dropna(subset=["age"]).sort_values("season").drop_duplicates("player_id", keep="last")
    ages = (prior["age"] + first - prior["season"]).set_axis(prior["player_id"])
    ids = groups.index.intersection(ages.index)
    pop = pd.DataFrame({"player_id": ids, "player_name": ids.astype(str), "age": ages.reindex(ids).values,
                        "team": "", "team_id": None, "status": "forty_man", "bats": "", "throws": "",
                        "listed_position": groups.reindex(ids).values})
    group_frame = groups.reindex(ids).rename("position_group").reset_index()
    group_frame["secondary_positions"] = [[] for _ in range(len(group_frame))]
    h = workload.history(workload.load_lines(list(range(first-5, first))), first,
                         set(bio.october_roster(first-1)["player_id"]))
    h = h[h["player_id"].isin(ids)]
    pt = workload.predict(h, ages, model.workload, first, last)
    if model.production:
        pt = production.add_forecasts(pt, model.production, first, last)
        if model.price.get("profile_prices"):
            pt = market_inputs.add_profiles(pt, first)
    elif model.production_aging:
        pt = aging.add_projection_shifts(pt, h, ages, model.production_aging, first)
    # Zero-cost schedules expose projected production through the SAME serving assembler.
    contracts = pd.DataFrame([{"player_id": pid, "season": year, "salary_kind": "guaranteed", "salary_musd": 0,
                               "option_musd": None, "option_kind": None, "service_time": 7,
                               "fa_year": last + 1, "how_acquired": "Free Agency"}
                              for pid in ids for year in range(first, last + 1)])
    details = []
    result, _ = score.assemble_values(pop, group_frame, rates, win, pd.DataFrame(), pt, contracts, model, first, projection_details=details)
    return result.set_index("player_id").join(pd.DataFrame(details).set_index("player_id"))



def calibrate(model: ValueModel, cohort: dict, projections: pd.DataFrame | None = None) -> tuple[pricing.PriceModel, dict]:
    """Fit the predetermined signing cohort and retain every exclusion and error.

    This is calibration, not a holdout pass. The skill and workload models must be
    frozen before the signing season. Contracts cannot be silently lost in a join.
    """
    first = cohort["first_season"]
    if max(model.dev_seasons) >= first or model.workload["last_target_season"] >= first:
        raise ValueError("Calibration forecasts overlap model training")
    if projections is None:
        last = first + max(c["years"] for c in cohort["contracts"]) - 1
        projections = historical_projection(model, first, last)
    rows, excluded = [], list(cohort.get("excluded", []))
    for c in cohort["contracts"]:
        if c["date"] > cohort["cutoff"] or c["first_season"] != first:
            raise ValueError("Signing lies outside the declared calibration cohort")
        if c["player_id"] not in projections.index:
            excluded.append({"name": c["name"], "reason": "No MLB projection"})
            continue
        row = projections.loc[c["player_id"]]
        if not row.get("production_supported", True):
            excluded.append({"name": c["name"], "reason": "Insufficient WAR production history"})
            continue
        if not model.production and pd.isna(row.grade):
            excluded.append({"name": c["name"], "reason": "Insufficient MLB sample"})
            continue
        wars = [row.war_by_season[y] for y in range(first, first + c["years"])]
        cost, production = pricing.contract_present_values(c, wars, value.DISCOUNT_PER_YEAR, pricing.FALLBACK_INFLATION)
        rows.append({**c, "position_group": row.position_group, "projected_war": wars,
                     "salary_pv_musd": cost, "production_pv_wins": production,
                     "replacement_pv_musd": pricing.replacement_present_value(first, c["years"], value.DISCOUNT_PER_YEAR)})
    frame = pd.DataFrame(rows)
    price = pricing.fit_price_per_win(frame, first)
    x, y = frame.production_pv_wins.to_numpy(), frame.salary_pv_musd.to_numpy()
    replacement = frame.replacement_pv_musd.to_numpy()
    excess = y - replacement
    denom = x @ x - x*x
    if (denom <= 0).any():
        raise ValueError("Dollar calibration is identified by only one contract")
    loo_price = (x @ excess - x*excess) / denom
    loo = replacement + x * loo_price
    rng = np.random.default_rng(20261017)
    ix = rng.integers(0, len(x), (10000, len(x)))
    bs_denom = np.sum(x[ix]**2, axis=1)
    identified = bs_denom > 0
    if identified.mean() < .975:
        raise ValueError("Too many unidentified bootstrap samples for dollar calibration")
    bs = np.sum(x[ix[identified]]*excess[ix[identified]], axis=1) / bs_denom[identified]
    for i, r in enumerate(rows):
        r.update(predicted_pv_musd=float(replacement[i]+price.price_musd*x[i]), loo_predicted_pv_musd=float(loo[i]),
                 residual_musd=float(replacement[i]+price.price_musd*x[i]-y[i]))
    return price, {"status": "calibrated_not_validated", "price": asdict(price), "price_bootstrap_95pct": np.quantile(bs, [.025,.975]).tolist(),
                   "selection": cohort["selection"], "cohort_source": cohort["cohort_source"], "cutoff": cohort["cutoff"],
                   "information_cutoff": cohort.get("information_cutoff", cohort["cutoff"]),
                   "cash_convention": cohort["cash_convention"], "unidentified_bootstrap_samples": int((~identified).sum()),
                   "candidate_count": len(cohort["contracts"])+len(cohort.get("excluded", [])), "included_count": len(rows),
                   "excluded": excluded, "contracts": rows, "inputs_through": first-1, "training_through": max(model.dev_seasons),
                   "loo_rmse_musd": float(np.sqrt(np.mean((loo-y)**2))),
                   "old_price_rmse_same_population_musd": float(np.sqrt(np.mean((7.353442753*x-y)**2))),
                   "fallback_rmse_same_population_musd": float(np.sqrt(np.mean((11*x-y)**2))),
                   "limitations": ["Calibration fit does not validate unseen contracts or option prices",
                                   "Excluded deals remain visible; this is not full-market coverage",
                                   "Bootstrap interval is calibration uncertainty, not an individual contract prediction interval"]}


def calibration_rows(fitted: dict, cohort: dict, workload_fitted: dict | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Project a fixed signing cohort with only the pre-signing fWAR model.

    Earlier cohorts need no descriptive grade model. The production history and
    prediction functions are the same ones used by the serving assembler.
    """
    first = int(cohort["first_season"])
    if fitted["last_target_season"] >= first:
        raise ValueError("Contract projections overlap training outcomes")
    h = production.history(production.load_lines(first-1), first,
                           include_roster="hitter_roster_features" in fitted,
                           include_surgery="surgery_features" in fitted, include_aaa="aaa" in fitted)
    groups = h[production.KEYS].assign(pricing_role=production.pricing_roles(h))
    last = first + max(c["years"] for c in cohort["contracts"]) - 1
    forecast = production.predict(h, fitted, first, last).merge(groups, on=production.KEYS, validate="many_to_one")
    profiles = (market_inputs.historical_profiles(workload_fitted, first, last)
                .groupby(["player_id", "season"])[market_inputs.PROFILE_COLUMNS].sum()
                if workload_fitted is not None else None)
    rows, excluded = [], list(cohort.get("excluded", []))
    for c in cohort["contracts"]:
        if c["first_season"] != first or not f"{first-1}-11-01" <= c["date"] <= cohort["cutoff"]:
            raise ValueError("Contract outside declared signing cohort")
        f = forecast[forecast.player_id.eq(c["player_id"])]
        if f.empty:
            excluded.append({"name": c["name"], "source": c["source"], "reason": "No eligible MLB production history"})
            continue
        if f.role.nunique() != 1:
            raise ValueError("Two-way contract needs separate role regressors; cannot fit a single-role slope")
        f = f.set_index("season")
        wars = [max(0.0, float(f.loc[y, "expected_war"])) for y in range(first, first+c["years"])]
        cost, wins = pricing.contract_present_values(c, wars, value.DISCOUNT_PER_YEAR, pricing.FALLBACK_INFLATION)
        signing_roles = pricing.contract_roles(c, {f.pricing_role.iloc[0]: wars[0]})
        profile = ([profiles.loc[(c["player_id"], y)].to_dict() for y in range(first, first+c["years"])]
                   if profiles is not None else None)
        rows.append({**c, "role": next(iter(signing_roles)), "projected_war": wars,
                     "projected_profiles": profile,
                     "prediction_training_through": fitted["last_target_season"],
                     "salary_pv_musd": cost, "production_pv_wins": wins,
                     "replacement_pv_musd": pricing.replacement_present_value(first, c["years"], value.DISCOUNT_PER_YEAR),
                     "annuity": sum((1-value.DISCOUNT_PER_YEAR)**i for i in range(c["years"]))})
    return pd.DataFrame(rows), excluded


def calibrate_history(cohorts: list[dict], fitted_by_season: dict[int, ValueModel]) -> tuple[pricing.PriceModel, dict]:
    """Fit multiple completed signing winters with chronologically frozen WAR."""
    rows, coverage = [], []
    for cohort in cohorts:
        first = int(cohort["first_season"])
        model = fitted_by_season[first]
        frame, excluded = calibration_rows(model.production, cohort, model.workload)
        rows.append(frame)
        coverage.append({"first_season": first, "cutoff": cohort["cutoff"], "included": len(frame),
                         "excluded": excluded, "source": cohort["cohort_source"], "selection": cohort["selection"]})
    frame = pd.concat(rows, ignore_index=True)
    price, diagnostics = pricing.fit_market_prices(frame, int(frame.first_season.max()), value.DISCOUNT_PER_YEAR)
    frame["predicted_pv_musd"] = [sum(price.value_for_roles({r.role: w}, r.first_season+i, r.first_season, p)
                                     * (1-value.DISCOUNT_PER_YEAR)**i
                                     for i, (w, p) in enumerate(zip(r.projected_war, r.projected_profiles)))
                                  for r in frame.itertuples()]
    return price, {"status": "calibrated_not_validated", "price": asdict(price), **diagnostics,
                   "cohorts": coverage, "contracts": frame.astype(object).where(frame.notna(), None).to_dict("records"),
                   "cutoff": max(c["cutoff"] for c in cohorts),
                   "limitations": ["Option and unsupported cash schedules are excluded visibly",
                                   "Calibration errors do not establish future market accuracy",
                                   "Historical fWAR and medical source revisions are not archived real-time inputs"]}


# Historical external surplus priors, not outputs of our MLB WAR model.
# Edwards (2018): https://blogs.fangraphs.com/an-update-to-prospect-valuation/
# Lower tiers: https://blogs.fangraphs.com/putting-a-dollar-value-on-prospects-outside-the-top-100/
PROSPECT_MUSD = {
    "bat": {"70": 112, "65": 62, "60": 55, "55": 46, "50": 28, "45+": 8, "45": 6, "40+": 4, "40": 2},
    "pit": {"70": 85, "65": 64, "60": 60, "55": 34, "50": 21, "45+": 6, "45": 4, "40+": 3, "40": 1},
}
PROSPECT_SOURCE = "https://blogs.fangraphs.com/an-update-to-prospect-valuation/"
LOWER_PROSPECT_SOURCE = "https://blogs.fangraphs.com/putting-a-dollar-value-on-prospects-outside-the-top-100/"
DRAFT_SOURCE = "https://blogs.fangraphs.com/an-update-on-how-to-value-draft-picks/"
# Published pick 51..70 surplus, already net of the signing bonus.
DRAFT_TAIL = np.array([5.8, 5.7, 5.5, 5.4, 5.3, 5.2, 5.0, 4.9, 4.8, 4.7,
                       4.6, 4.5, 4.4, 4.3, 4.3, 4.2, 4.1, 4.0, 3.9, 3.8])


def package_value(assets: list[dict], projections: pd.DataFrame, model: ValueModel, first: int) -> dict:
    """Account for every asset once; never turn an unknown asset into zero.

    External priors show a 0%-3% annual inflation sensitivity, not a confidence
    interval. They are not scaled by our model-specific price per projected win.
    Thin MLB samples retain their shrunk projection with an explicit warning;
    this never makes them eligible for a public grade.
    """
    components, unpriced = [], []
    for a in assets:
        kind = a["kind"]
        notes = []
        if kind == "cash":
            amount = np.asarray(a["cash_musd"], dtype=float)
            if not np.isfinite(amount).all():
                raise ValueError("Invalid trade cash")
            point = float(amount @ (1-value.DISCOUNT_PER_YEAR)**np.arange(len(amount)))
            bounds = [point, point]
        elif kind == "mlb":
            if a["player_id"] not in projections.index:
                unpriced.append(a["name"])
                continue
            row = projections.loc[a["player_id"]]
            schedule = [value.ContractYear(**c) for c in a["contract"]]
            wars = row.war_by_season
            needed = [c.season for c in schedule if c.season >= first]
            if row["sample"] <= 0 or any(y not in wars or not np.isfinite(wars[y]) for y in needed):
                unpriced.append(a["name"])
                continue
            outlook = value.project_outlook(wars, {}, schedule, model.price_model(), model.arb_fractions,
                                            a["service_time"], first, service_season=a["service_season"],
                                            war_by_role=row.get("war_by_role"), market_profiles=row.get("market_profiles"), arbitration_model=model.arbitration)
            point = value.trade_value(outlook)
            bounds = [point, point]
            if pd.isna(row.grade):
                notes.append("Thin MLB sample: shrunk estimate, not a publishable player grade")
            notes.append("Unsigned future salaries use arbitration assumptions, not known guarantees")
        elif kind == "prospect":
            table = PROSPECT_MUSD[a["role"]]
            if a["fv"] == "35+":
                # The old published study stops at 40. Report a monotone bound
                # below 40, rather than invent a point estimate or use 2026 results.
                point = None
                bounds = [0.0, table["40"] * 1.03**(first-2019)]
                notes.append("35+ below published table: bounded from zero to the 40-FV prior; no supported point value")
            else:
                base = table[a["fv"]]
                point = float(base * 1.03**(first-2019))
                bounds = [float(base), point]
            notes.append("External FV surplus prior, including development risk and control salaries; inflation sensitivity only")
        elif kind == "draft_pick":
            lo, hi = a["pick_range"]
            if not 51 <= lo <= hi <= 80:
                raise ValueError("Draft prior only supports the observed tail and a ten-pick extrapolation")
            beta = np.polyfit(np.arange(51, 71), np.log(DRAFT_TAIL), 1)
            def pick_value(pick: float) -> float:
                return float(np.interp(pick, np.arange(51, 71), DRAFT_TAIL)) if pick <= 70 else float(np.exp(np.polyval(beta, pick)))
            point = pick_value((lo+hi)/2) * 1.03**(first-2019)
            bounds = [pick_value(hi), pick_value(lo) * 1.03**(first-2019)]
            notes.append("Draft location and inflation sensitivity; picks after 70 extrapolate the published log-linear tail; bonus already subtracted")
        else:
            raise ValueError(f"Unknown trade asset kind: {kind}")
        components.append({**a, "value_musd": point, "scenario_bounds_musd": bounds, "valuation_notes": notes})
    complete = not unpriced and len(components) == len(assets)
    total = [float(sum(c["scenario_bounds_musd"][i] for c in components)) for i in (0, 1)] if complete else None
    return {"asset_count": len(assets), "valued_count": len(components), "unpriced_assets": unpriced,
            "complete_accounting": complete, "scenario_bounds_musd": total, "components": components}


def contract_holdout(model: ValueModel, cohort: dict, calibration_cutoff: str, *,
                     baseline_projections: pd.DataFrame | None = None,
                     baseline_price: pricing.PriceModel | None = None) -> dict:
    """Paired dollar errors on the next season, with no fitting or exclusions by result."""
    first = cohort["first_season"]
    if any(c["date"] <= calibration_cutoff for c in cohort["contracts"]):
        raise ValueError("Holdout signing overlaps the price calibration dates")
    if (baseline_projections is None) != (baseline_price is None):
        raise ValueError("Full baseline comparison requires both forecasts and their fitted price")
    projections = historical_projection(model, first, first + max(c["years"] for c in cohort["contracts"]) - 1)
    rows, excluded = [], list(cohort.get("excluded", []))
    for c in cohort["contracts"]:
        if (c["player_id"] not in projections.index
                or not projections.loc[c["player_id"]].get("production_supported", True)
                or (not model.production and pd.isna(projections.loc[c["player_id"], "grade"]))):
            excluded.append({"name": c["name"], "reason": "Insufficient MLB projection history"})
            continue
        p = projections.loc[c["player_id"]]
        wars = [p.war_by_season[y] for y in range(first, first+c["years"])]
        cost, _ = pricing.contract_present_values(c, wars, value.DISCOUNT_PER_YEAR, model.price["inflation"])
        price = model.price_model()
        prediction = sum(price.value_for_roles(pricing.contract_roles(c, p.war_by_role[y]), y, first,
                                               p.market_profiles[y] if p.get("market_profiles") is not None else None) * (1-value.DISCOUNT_PER_YEAR)**(y-first)
                         for y in range(first, first+c["years"]))
        baseline = None
        if baseline_projections is not None and c["player_id"] in baseline_projections.index:
            old = baseline_projections.loc[c["player_id"]]
            years = range(first, first+c["years"])
            if pd.notna(old.grade) and all(pd.notna(old.war_by_season.get(y)) for y in years):
                baseline = sum(baseline_price.value_for(old.war_by_season[y], y, first) * (1-value.DISCOUNT_PER_YEAR)**(y-first)
                               for y in years)
        rows.append({**c, "war": wars, "salary_pv_musd": cost, "predicted_pv_musd": prediction,
                     "baseline_pv_musd": baseline, "error_musd": prediction-cost,
                     "annuity": sum((1-value.DISCOUNT_PER_YEAR)**i for i in range(c["years"]))})
    if not rows:
        raise ValueError("Empty market holdout")
    report = {"status": "baseline_not_evaluated", "n": len(rows), "selection": cohort["selection"],
              "contracts": rows, "excluded": excluded,
              "baseline_scope": "Full supplied pipeline, not the old price applied to candidate WAR"}
    paired = [r for r in rows if r["baseline_pv_musd"] is not None]
    report["paired_count"] = len(paired)
    report["newly_covered"] = [r["name"] for r in rows if r["baseline_pv_musd"] is None]
    covered = {r["player_id"] for r in rows}
    report["lost_coverage"] = ([c["name"] for c in cohort["contracts"]
                               if c["player_id"] not in covered and c["player_id"] in baseline_projections.index
                               and pd.notna(baseline_projections.loc[c["player_id"], "grade"])]
                              if baseline_projections is not None else [])
    if not paired:
        return report
    # Same annualized cash objective as calibration; total PV errors remain
    # visible so a long guarantee cannot disappear in a one-year average.
    c = np.array([r["error_musd"]/r["annuity"] for r in paired])
    b = np.array([(r["baseline_pv_musd"]-r["salary_pv_musd"])/r["annuity"] for r in paired])
    rng = np.random.default_rng(20261017)
    ix = rng.integers(0, len(c), (10000, len(c)))
    interval = np.quantile((c*c-b*b)[ix].mean(axis=1), [.025,.975]).tolist()
    bias = np.quantile(c[ix].mean(axis=1), [.025,.975]).tolist()
    return {**report, "status": "passed" if interval[1] < 0 and bias[0] <= 0 <= bias[1] and not report["lost_coverage"] else "failed",
            "criterion": "Paired annualized PV MSE improvement 95% interval below zero and dollar bias interval includes zero",
            "candidate_rmse_musd": float(np.sqrt(np.mean(c*c))),
            "baseline_rmse_musd": float(np.sqrt(np.mean(b*b))), "mse_change_95pct": interval, "bias_95pct_musd": bias,
            "candidate_total_pv_rmse_musd": float(np.sqrt(np.mean([r["error_musd"]**2 for r in paired]))),
            "baseline_total_pv_rmse_musd": float(np.sqrt(np.mean([(r["baseline_pv_musd"]-r["salary_pv_musd"])**2 for r in paired])))}
