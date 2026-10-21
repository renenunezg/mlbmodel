"""Nightly scoring: Grade, Trade value, Impact, and explained traits for the population.

Reads the stored model, computes the window of inputs, and assembles the two
published frames. Skips every write when the inputs are unchanged.

Usage:
    python -m v2.players.score --as-of 2026-10-16 [--no-write] [--force] [--model PATH] [--player ID ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from backend.db import writes_allowed
from backend.log import setup_logging
from v2.data.pa_dataset import OUTCOMES
from v2.players import aging, bio, features, inputs, minors, production, value, workload
from v2.players.contracts import estimated_schedule, load_contracts
from v2.players.model import ValueModel, load_latest_from_db
from v2.players.pricing import league_minimum
from v2.players.runs import FULL_SEASON_PA, RELIEVER_INNINGS, RUNS_PER_OAA, STARTER_INNINGS

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "players"
LAST_SCORED = CACHE_DIR / "last_scored.json"
WINDOW_SEASONS = 3
DRAWS = 30
# Fallback display volume for players without an MLB workload history.
PLAYING_TIME_NORM = {"hitter": 450.0, "SP": 150.0 * 4.37, "RP": 60.0 * 4.37}  # PA or batters faced
HEADSHOT = "https://img.mlbstatic.com/mlb-photos/image/upload/w_213,q_auto:best/v1/people/{player_id}/headshot/67/current"
FADED_FREE_AGENT_SEASONS = 2
CATCHER_RUN_INPUTS = ("reliability.framing_runs", "reliability.blocking_runs", "arm.cs_above_average")


def season_for(as_of: date, starts: dict[int, date]) -> int:
    """The scored season: the latest season whose regular season has started by ``as_of``."""
    started = [s for s, start in starts.items() if start <= as_of]
    return max(started) if started else min(starts)


def current_season(as_of: date) -> int:
    starts = {}
    for season in (as_of.year, as_of.year - 1):
        try:
            starts[season], _ = bio.season_dates(season)
        except Exception:  # the next season's dates are not published yet
            continue
    return season_for(as_of, starts)


def next_season(as_of: date, season: int) -> int:
    _, end = bio.season_dates(season)
    return season if as_of <= end else season + 1


def window_weights_by_season(model: ValueModel, last_season: int) -> dict[int, float]:
    return {last_season - back: w for back, w in model.window_weights.items()}


def fingerprint(as_of: date, population: pd.DataFrame, contracts: pd.DataFrame, model: ValueModel, seasons: list[int]) -> str:
    parts = [model.model_version, model.fitted_at, json.dumps(bio.FIELD_POSITIONS, sort_keys=True),
             json.dumps(model.price, sort_keys=True), json.dumps(model.workload, sort_keys=True)]
    if model.production:
        from v2.players import cash

        parts.extend(hashlib.sha256(path.read_bytes()).hexdigest() for path in cash.sources())
        parts.append(json.dumps(model.production, sort_keys=True))
        parts.extend(hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in production.source_paths(max(seasons)))
        for path in [bio.CACHE_DIR / "tommy_john_surgeries.csv",
                     *[bio.CACHE_DIR / f"roster_oct1_{year}.parquet" for year in (max(seasons)-1, max(seasons))],
                     *[bio.CACHE_DIR / f"aaa_season_{year}.json" for year in (max(seasons)-1, max(seasons))]]:
            if path.exists():
                parts.append(hashlib.sha256(path.read_bytes()).hexdigest())
    for season in seasons:
        path = inputs.STATCAST_DIR / f"statcast_{season}.parquet"
        parts.append(str(pd.read_parquet(path, columns=["game_date"])["game_date"].max()) if path.exists() else "none")
    parts.append(str(contracts["fetched_at"].max()) if "fetched_at" in contracts and len(contracts) else "no-contracts")
    parts.extend(f"{r.player_id}:{r.status}:{r.team}" for r in population.sort_values("player_id").itertuples())
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _contract_years(rows: pd.DataFrame) -> list[value.ContractYear]:
    return [
        value.ContractYear(int(r.season), str(r.salary_kind), None if pd.isna(r.salary_musd) else float(r.salary_musd),
                           None if pd.isna(r.option_musd) else float(r.option_musd), (r.option_kind if isinstance(r.option_kind, str) else "team"),
                           r.declined_to if isinstance(getattr(r, "declined_to", None), str) else None)
        for r in rows.sort_values("season").itertuples()
    ]


def assemble_values(
    population: pd.DataFrame, groups: pd.DataFrame, rates: pd.DataFrame, inputs: pd.DataFrame, il_days: pd.DataFrame,
    playing_time: pd.DataFrame, contracts: pd.DataFrame, model: ValueModel, next_season: int,
    actual: pd.DataFrame | None = None, draws: np.ndarray | None = None, as_of: date | None = None,
    projection_details: list[dict] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """From already-computed frames to the player_values and player_traits frames."""
    if model.production and model.production.get("method") != production.METHOD:
        raise ValueError("Economic production requires a fitted fWAR model")
    pop = population.merge(groups, on="player_id", how="left")
    pop["position_group"] = pop["position_group"].fillna(pop["listed_position"].map(lambda p: "RP" if p in ("P", "SP", "RP", "TWP") else ("LF" if p == "OF" else p))).fillna("DH")
    pop = pop.set_index("player_id")
    group = pop["position_group"]
    pool = pd.Series(features.pool_of(group), index=pop.index)

    # Inputs, z-scores, composites.
    values_wide, samples_wide = features.wide_inputs(inputs) if not inputs.empty else (pd.DataFrame(index=pop.index), pd.DataFrame(index=pop.index))
    values_wide = values_wide.reindex(pop.index)
    samples_wide = samples_wide.reindex(pop.index)
    z = features.oriented_z(values_wide, pool) if not values_wide.empty else values_wide
    pct = features.input_percentiles(values_wide, group) if not values_wide.empty else values_wide
    composites, coverage = features.trait_composites(z, pool, model)

    # Shrunk runs and the projection.
    role_series = pd.Series(np.where(group.isin(features.PITCHER_GROUPS), "pit", "bat"), index=pop.index)
    leverage = values_wide["workload.leverage"] if "workload.leverage" in values_wide.columns else None
    rates_pop = features.role_rows(rates[rates["role"].isin(["bat", "pit"])], group)
    shrunk = features.shrunk_runs(rates_pop, model, group, leverage) if not rates_pop.empty else pd.Series(dtype="float64")
    shrunk = shrunk.reindex(pop.index)
    # Two-way players: the pitching side (a starter's workload) is added to the hitter's projection
    # through the shrunk-runs path, since one row per player carries one set of traits.
    pit_rows = rates[(rates["role"] == "pit") & (rates["n"] >= features.MIN_GRADE_SAMPLE)]
    two_way = pit_rows[pit_rows["player_id"].isin(pop.index[~group.isin(features.PITCHER_GROUPS)])]
    two_way_runs = pd.Series(0.0, index=pop.index)
    if not two_way.empty:
        sp_role = pd.Series("SP", index=two_way["player_id"].values)
        gw = model.grade_weight_frame()
        sp = gw[gw["pool"] == "SP"].set_index("feature")["coefficient"] if not gw.empty else pd.Series(dtype="float64")
        pit_runs = features.pitching_runs_for(two_way, model, sp_role, leverage)
        projected_pit = float(sp.get("intercept", 0.0)) + float(sp.get("shrunk_runs", 1.0)) * pit_runs - model.replacement.get("SP", 0.0)
        two_way_runs.loc[projected_pit.index] = projected_pit.values
    sample = rates_pop.set_index("player_id")["n"] if not rates_pop.empty else pd.Series(dtype="float64")
    sample = sample.reindex(pop.index).fillna(0.0)
    design = features.grade_features(composites, shrunk, pool, pop["age"])
    two_way_ids = set(two_way["player_id"]) if not two_way.empty else set()
    projected = features.project_runs(design, group, model)
    has_fit = bool(model.grade_weights)
    if not has_fit:
        projected = shrunk  # no fitted weights: the shrunk runs stand in
    gradeable = sample >= features.MIN_GRADE_SAMPLE
    # A thin MLB record backed by enough Triple-A plate appearances gets a provisional grade on the
    # graded players' scale; it never enters that scale, the percentiles, or the dollar projections.
    prior_n = (rates_pop.set_index("player_id")["prior_n"] if "prior_n" in rates_pop else pd.Series(0.0, index=pop.index)).reindex(pop.index).fillna(0.0)
    provisional = ~gradeable & (sample + prior_n >= features.MIN_GRADE_SAMPLE) & has_fit
    pos_adj = group.map(model.positional_adjustment).fillna(0.0).where(role_series.eq("bat"), 0.0)
    replacement = group.map(model.replacement).fillna(0.0)
    full_runs = projected + pos_adj - replacement + two_way_runs  # per 600 PA or per 180/65 IP
    grade_raw = full_runs.where(gradeable)
    grade = features.to_grade_scale(grade_raw, group)
    grade = grade.fillna(features.to_grade_scale(full_runs.where(provisional), group, reference=grade_raw))
    k_grade = pd.Series([model.reliability_k.get(f"grade_{p}", 0.0) for p in np.where(group.isin(features.PITCHER_GROUPS), group, "hitter")], index=pop.index)
    band = features.reliability_band(sample, k_grade.astype(float).fillna(0.0))

    # Playing time and WAR by season.
    first_pt = playing_time[playing_time["season"].eq(next_season)] if "season" in playing_time else playing_time
    pt = features.role_rows(first_pt, group).set_index("player_id")["pa"].reindex(pop.index)
    pt = pt.fillna(pd.Series([PLAYING_TIME_NORM["RP"] if g == "RP" else PLAYING_TIME_NORM["SP"] if g == "SP" else PLAYING_TIME_NORM["hitter"] for g in group], index=pop.index))
    per_unit = pd.Series(np.where(role_series.eq("bat"), FULL_SEASON_PA, np.where(group.eq("SP"), STARTER_INNINGS, RELIEVER_INNINGS) * model.pa_per_ip), index=pop.index)
    future_pt = playing_time.set_index(["player_id", "role", "season"])["pa"] if "season" in playing_time else None
    future_shift = (playing_time.set_index(["player_id", "role", "season"])["run_shift"]
                    if "run_shift" in playing_time else None)
    future_war = (playing_time.set_index(["player_id", "role", "season"])["expected_war"]
                  if "expected_war" in playing_time else None)
    pricing_role = (playing_time.set_index(["player_id", "role", "season"])["pricing_role"]
                    if "pricing_role" in playing_time else None)
    profiles = None
    if model.price.get("profile_prices"):
        from v2.players.market_inputs import PROFILE_COLUMNS

        if not set(PROFILE_COLUMNS) <= set(playing_time):
            raise ValueError("Market pricing requires dated workload profiles")
        profiles = playing_time.groupby(["player_id", "season"])[PROFILE_COLUMNS].sum().to_dict("index")
    if price_roles := model.price.get("role_prices"):
        if pricing_role is None or set(price_roles) != {"bat", "SP", "RP"}:
            raise ValueError("Role-calibrated scoring requires dated production roles")
    if model.production and future_war is None:
        raise ValueError("WAR model requires dated production forecasts")
    if not model.production and model.production_aging and future_shift is None:
        raise ValueError("Production aging model requires dated horizon projections")
    rpw = model.runs_per_win
    age_next = pop["age"].astype("float64")
    curve_name = pd.Series(np.where(role_series.eq("bat"), "bat_runs_600", "pitch_runs_180"), index=pop.index)
    price = model.price_model()
    contracts = contracts.copy() if contracts is not None else pd.DataFrame()
    by_player = {pid: g for pid, g in contracts.groupby("player_id")} if len(contracts) else {}
    out_rows, trait_rows = [], []
    curves = model.age_curves
    gwf = model.grade_weight_frame()
    in_grade_traits = set(gwf.loc[gwf["coefficient"] > 0, "feature"]) if not gwf.empty else set(features.TRAITS)
    for pid in pop.index:
        g = str(group[pid])
        status = pop.at[pid, "status"]
        rows = by_player.get(pid)
        last_season = max(next_season, int(rows["season"].max())) if rows is not None and len(rows) else next_season
        years = list(range(next_season, last_season + 1))
        base = float(full_runs[pid]) if pd.notna(full_runs[pid]) else np.nan
        war_by_season, ages, scales = {}, {}, {}
        for y in years:
            a = int(age_next[pid]) + (y - next_season) if pd.notna(age_next[pid]) else 27
            ages[y] = a
            volume = float(future_pt.get((pid, role_series[pid], y), 0.0)) if future_pt is not None else float(pt[pid])
            scales[y] = volume / float(per_unit[pid])
            if model.production:
                primary = future_war.get((pid, role_series[pid], y), np.nan)
                other = future_war.get((pid, "pit", y), np.nan) if pid in two_way_ids else 0.0
                war_by_season[y] = max(0.0, float(primary + other)) if pd.notna(primary + other) else np.nan
                continue
            shift = 0.0
            if future_shift is not None:
                shift = float(future_shift.get((pid, role_series[pid], y), 0.0))
                if g == "RP":
                    shift *= RELIEVER_INNINGS / STARTER_INNINGS
            elif curves and curve_name[pid] in curves and pd.notna(age_next[pid]):
                shift = aging.apply(0.0, curve_name[pid], int(age_next[pid]), a, curves)
                if g == "RP":
                    shift *= RELIEVER_INNINGS / STARTER_INNINGS
            # Two-way pitching uses its own BF forecast, never the hitter's PA multiplier.
            pit_volume = (float(future_pt.get((pid, "pit", y), 0.0)) if future_pt is not None
                          else float(first_pt.loc[(first_pt["player_id"] == pid) & (first_pt["role"] == "pit"), "pa"].sum()))
            pit_shift = float(future_shift.get((pid, "pit", y), 0.0)) if future_shift is not None and pid in two_way_ids else 0.0
            pitching = (float(two_way_runs[pid]) + pit_shift) * pit_volume / (STARTER_INNINGS * model.pa_per_ip)
            batting_or_pitching = base - float(two_way_runs[pid])
            # A player projected below replacement is replaced: WAR floors at zero, as in the draws.
            war_by_season[y] = max(0.0, ((batting_or_pitching + shift) * scales[y] + pitching) / rpw) if pd.notna(base) else np.nan
        war_by_role = {y: {str(role_series[pid]): war_by_season[y]} for y in years}
        if model.production:
            for y in years:
                roles = ("bat", "pit") if pid in two_way_ids else (str(role_series[pid]),)
                war_by_role[y] = {}
                for r in roles:
                    fallback_role = "bat" if r == "bat" else g if g in ("SP", "RP") else "SP"
                    pr = pricing_role.get((pid, r, y), fallback_role) if pricing_role is not None else fallback_role
                    pr = fallback_role if pd.isna(pr) else pr
                    w = float(future_war.get((pid, r, y), np.nan))
                    war_by_role[y][pr] = max(0.0, w) if pd.notna(w) else np.nan
                war_by_season[y] = sum(war_by_role[y].values())
        production_known = all(pd.notna(w) for w in war_by_season.values())
        if production_known and profiles is not None and any((pid, y) not in profiles for y in years):
            raise ValueError("Supported WAR forecast is missing its market workload profile")
        market_profiles = ({y: profiles.get((pid, y), {"expected_saves": 0.0, "bat_pa": 0.0, "pit_pa": 0.0}) for y in years}
                           if profiles is not None else None)
        if projection_details is not None:
            # Unrounded production for market replay, including explicitly labelled
            # thin-sample estimates. This does not weaken the public grade gate.
            projection_details.append({"player_id": int(pid), "sample": float(sample[pid]),
                                       "war_by_season": war_by_season, "war_by_role": war_by_role, "production_supported": production_known, "market_profiles": market_profiles})
        contract_status, service_time, fa_year = "unknown", np.nan, None
        cash_metadata = {"basis": "Payroll schedule; actual cash timing not verified", "verified": False}
        if rows is not None and len(rows):
            contract = _contract_years(rows)
            if model.production:
                from v2.players import cash

                contract, cash_metadata = cash.apply(rows, contract, as_of)
            service_time = float(rows["service_time"].iloc[0]) if pd.notna(rows["service_time"].iloc[0]) else np.nan
            fa_year = int(rows["fa_year"].dropna().iloc[0]) if rows["fa_year"].notna().any() else None
            contract_status = "estimated" if (rows["how_acquired"] == "estimated").all() else "contract"
            if fa_year is not None and fa_year <= next_season:
                # Free agency arrives before the next season: no control years, a market value.
                contract = [value.ContractYear(next_season, "free_agent", None, None)]
                contract_status = "free_agent"
        elif status == "free_agent":
            contract = [value.ContractYear(next_season, "free_agent", None, None)]
            contract_status = "free_agent"
        else:
            contract = []
        if status == "free_agent":
            contract = [value.ContractYear(next_season, "free_agent", None, None)]
            contract_status = "free_agent"
        war_known = {y: (0.0 if pd.isna(w) else w) for y, w in war_by_season.items()}
        known_roles = {y: {r: max(0.0, w) if pd.notna(w) else 0.0 for r, w in wars.items()}
                       for y, wars in war_by_role.items()}
        graded = bool(gradeable[pid]) and pd.notna(grade[pid])
        grade_shown = graded or (bool(provisional[pid]) and pd.notna(grade[pid]))
        service_season = int(rows["season"].min()) if rows is not None and len(rows) and contract_status != "free_agent" else next_season
        # Preserve unsigned control when production is unknown; do not infer a non-tender
        # from a missing grade. Dollar projections require a grade.
        outlook = value.project_outlook(war_known, ages, contract, price, model.arb_fractions, service_time, next_season, service_season=service_season, allow_non_tender=graded and production_known, war_by_role=known_roles, market_profiles=market_profiles, arbitration_model=model.arbitration) if contract else []
        control_years = sum(1 for o in outlook if o.in_control)
        tv = value.trade_value(outlook) if outlook else np.nan
        # Market value is what his next season would fetch, whatever his contract says about it.
        mv = price.value_for_roles(known_roles[next_season], next_season, next_season,
                                   market_profiles[next_season] if market_profiles is not None else None)
        if pd.isna(base) or contract_status == "unknown" or not graded or not production_known:
            tv = mv = np.nan
        # Draws for the range.
        tv_low = tv_high = np.nan
        if not model.production and draws is not None and pid in draws.index and pd.notna(tv):
            d = draws.loc[pid]
            tv_draws = []
            for delta in d:
                war_d = {y: max(0.0, war_known[y] + float(delta) * scales[y] / rpw) for y in years}
                tv_draws.append(value.trade_value(value.project_outlook(war_d, ages, contract, price, model.arb_fractions, service_time, next_season, service_season=service_season)))
            tv_low, tv_high = float(np.percentile(tv_draws, 10)), float(np.percentile(tv_draws, 90))
        shown = [o for o in outlook if o.in_control or o.salary_kind == "buyout"]
        faded = [o for o in outlook if not (o.in_control or o.salary_kind == "buyout")][:FADED_FREE_AGENT_SEASONS]
        outlook_json = [
            {"season": o.season, "age": o.age, "war": round(o.war, 2) if graded and production_known else None, "salary_musd": round(o.salary_musd, 2), "salary_kind": o.salary_kind,
             "surplus_musd": round(o.surplus_musd, 1) if graded and production_known else None, "in_control": o.in_control}
            for o in shown + faded
        ]
        owed = float(sum(o.salary_musd for o in shown))
        # Next season's salary: the contract figure, else the estimated arbitration or minimum from the outlook.
        current_salary = None
        if rows is not None and len(rows):
            cur = rows[rows["season"] == next_season]
            if len(cur) and cur["salary_kind"].iloc[0] == "guaranteed" and pd.notna(cur["salary_musd"].iloc[0]):
                current_salary = float(cur["salary_musd"].iloc[0])
            elif len(cur) and cur["salary_kind"].iloc[0] == "pre_arb":
                current_salary = league_minimum(next_season)
        if current_salary is None and outlook and outlook[0].season == next_season and outlook[0].in_control:
            current_salary = round(outlook[0].salary_musd, 2)
        if cash_metadata["verified"] and outlook and outlook[0].season == next_season and outlook[0].in_control:
            current_salary = round(outlook[0].salary_musd, 2)
        impact_war = impact_runs = np.nan
        if actual is not None and pid in actual.index:
            impact_runs = float(actual.at[pid, "runs"])
            impact_war = impact_runs / rpw
        role_prices = {r: price.price_for(next_season, next_season, r) for r in known_roles[next_season]}
        total_war = sum(known_roles[next_season].values())
        effective_price = (sum(w * role_prices[r] for r, w in known_roles[next_season].items()) / total_war
                           if total_war > 0 else next(iter(role_prices.values())) if len(role_prices) == 1 else None)
        tip = {
            "production_source": model.production.get("method", "legacy_trait_runs"),
            "production_supported": production_known,
            "projected_runs": None if pd.isna(projected[pid]) else round(float(projected[pid]), 1),
            "positional_adjustment": round(float(pos_adj[pid]), 1), "replacement_runs": round(float(replacement[pid]), 1),
            "playing_time": round(float(pt[pid]), 0), "runs_per_win": round(rpw, 2),
            "price_per_win_musd": round(effective_price, 2) if effective_price is not None else None,
            "price_source": price.source, "inflation": round(price.inflation, 3), "discount": value.DISCOUNT_PER_YEAR,
            "prices_by_role_musd": {r: round(p, 2) for r, p in role_prices.items()},
            "market_usage": market_profiles[next_season] if market_profiles is not None else None,
            "relief_usage_price_musd": price.profile_prices.get("expected_saves"),
            "arbitration_source": model.arbitration.get("method", "fallback_market_fractions"),
            "contract_cash": cash_metadata,
            "sample": round(float(sample[pid]), 0), "shrunk_runs": None if pd.isna(shrunk[pid]) else round(float(shrunk[pid]), 1),
            "two_way_pitching_runs": round(float(two_way_runs[pid]), 1) if pid in two_way_ids else None,
        }
        out_rows.append({
            "player_id": int(pid), "player_name": pop.at[pid, "player_name"], "team": pop.at[pid, "team"],
            "team_id": None if pd.isna(pop.at[pid, "team_id"]) else int(pop.at[pid, "team_id"]), "level": "mlb",
            "position_group": g, "secondary_positions": list(pop.at[pid, "secondary_positions"]) if isinstance(pop.at[pid, "secondary_positions"], list) else [],
            "age": None if pd.isna(age_next[pid]) else int(age_next[pid]), "bats": pop.at[pid, "bats"], "throws": pop.at[pid, "throws"],
            "status": status, "as_of": as_of, "model_version": model.model_version, "season": next_season,
            "grade": round(float(grade[pid]), 1) if grade_shown else None,
            "grade_low": round(float(max(20.0, grade[pid] - band[pid])), 1) if grade_shown else None,
            "grade_high": round(float(min(80.0, grade[pid] + band[pid])), 1) if grade_shown else None,
            "grade_percentile": None, "grade_reason": None if graded else ("provisional_sample" if grade_shown else "insufficient_sample"),
            "projected_war": None if not graded or pd.isna(war_by_season.get(next_season, np.nan)) else round(float(war_by_season[next_season]), 2),
            "trade_value_musd": None if pd.isna(tv) else round(float(tv), 1),
            "trade_value_low": None if pd.isna(tv_low) else round(tv_low, 1), "trade_value_high": None if pd.isna(tv_high) else round(tv_high, 1),
            "market_value_musd": None if pd.isna(mv) or not graded else round(float(mv), 1),
            "impact_war": None if pd.isna(impact_war) else round(impact_war, 2), "impact_runs": None if pd.isna(impact_runs) else round(impact_runs, 1),
            "playing_time": round(float(pt[pid]), 0),
            "service_time": None if pd.isna(service_time) else service_time, "contract_status": contract_status, "fa_year": fa_year,
            "control_years": int(control_years), "salary_current_musd": current_salary, "owed_through_control_musd": round(owed, 2),
            "outlook": outlook_json, "tip": tip, "headshot_url": HEADSHOT.format(player_id=int(pid)),
        })
        # Traits.
        for trait, spec in features.TRAITS.items():
            if spec.pool != pool[pid]:
                continue
            comp = composites.at[pid, trait] if trait in composites.columns else np.nan
            tw = [(r["input"], float(r["weight"]), r["status"]) for r in model.trait_weights if r["pool"] == spec.pool and r["trait"] == trait]
            inputs_json = []
            trait_sample = 0.0
            for name, w, _status in tw:
                v = values_wide.at[pid, name] if name in values_wide.columns else np.nan
                if pd.isna(v):
                    continue
                inputs_json.append([name, round(w, 3), round(float(v), 4), round(float(pct.at[pid, name]), 1) if name in pct.columns and pd.notna(pct.at[pid, name]) else None])
                trait_sample = max(trait_sample, float(samples_wide.at[pid, name]) if name in samples_wide.columns and pd.notna(samples_wide.at[pid, name]) else 0.0)
            kind = "physical" if len(tw) == 1 and features.INPUT_KIND.get(tw[0][0]) == "physical" else "measured"
            trait_rows.append({
                "player_id": int(pid), "trait": trait, "panel": spec.panel, "value": None if pd.isna(comp) else round(float(comp), 4),
                "in_grade": bool(any(w > 0 for _, w, _ in tw)) and trait in in_grade_traits, "kind": kind, "sample": int(trait_sample), "inputs": inputs_json,
                "as_of": as_of, "model_version": model.model_version,
            })
    values_frame = pd.DataFrame(out_rows)
    traits_frame = pd.DataFrame(trait_rows)
    if not traits_frame.empty:
        comp_series = traits_frame.set_index(["player_id", "trait"])["value"]
        grp = traits_frame["player_id"].map(group)
        scaled = features.to_grade_scale(comp_series.reset_index(drop=True), grp)
        traits_frame["grade"] = scaled.round(1).values
        traits_frame["percentile"] = (comp_series.reset_index(drop=True).groupby(grp).rank(pct=True) * 100).round(1).values
        k_trait = traits_frame["trait"].map(lambda t: model.reliability_k.get(t, 0.0)).astype(float)
        tb = features.reliability_band(traits_frame["sample"].astype(float), k_trait.values)
        traits_frame["grade_low"] = (traits_frame["grade"] - tb).clip(lower=20).round(1)
        traits_frame["grade_high"] = (traits_frame["grade"] + tb).clip(upper=80).round(1)
        traits_frame.loc[traits_frame["value"].isna(), ["grade", "grade_low", "grade_high", "percentile"]] = None
    if not values_frame.empty:
        values_frame["grade_percentile"] = _grade_percentiles(values_frame)
    return values_frame, traits_frame


def _grade_percentiles(values_frame: pd.DataFrame) -> pd.Series:
    """Percentile among fully graded players of the group; a provisional grade takes the share of them it matches or beats."""
    full = values_frame["grade_reason"].isna() & values_frame["grade"].notna()
    out = pd.Series(np.nan, index=values_frame.index)
    for position, rows in values_frame.groupby("position_group"):
        reference = np.sort(rows.loc[full.reindex(rows.index), "grade"].to_numpy(float))
        if not len(reference):
            continue
        graded_rows = rows.index[full.reindex(rows.index)]
        out.loc[graded_rows] = rows.loc[graded_rows, "grade"].rank(pct=True) * 100
        provisional_rows = rows.index[rows["grade_reason"].eq("provisional_sample")]
        out.loc[provisional_rows] = np.searchsorted(reference, rows.loc[provisional_rows, "grade"].to_numpy(float), side="right") / len(reference) * 100
    return out.round(1)


def _actual_runs(season: int, model: ValueModel, groups: pd.Series, lev: pd.Series) -> pd.DataFrame:
    """This season's runs above replacement from actual outcomes, fielding, and running, per player."""
    counts = inputs.outcome_counts(season)
    board = inputs.season_inputs(season).pivot_table(index="player_id", columns="input", values="value", aggfunc="first")
    counts = features.role_rows(counts, groups).set_index("player_id")
    w = np.array([model.linear_weights[k] for k in OUTCOMES])
    c = counts[list(OUTCOMES)].to_numpy(float)
    n = c.sum(axis=1)
    total = c @ w
    group = pd.Series(counts.index.map(groups), index=counts.index)
    group = group.fillna(pd.Series(np.where(counts["role"].eq("pit"), "RP", "DH"), index=counts.index))
    is_bat = counts["role"].eq("bat").to_numpy()

    def col(name: str) -> np.ndarray:
        return board[name].reindex(counts.index).fillna(0.0).to_numpy(float) if name in board.columns else np.zeros(len(counts))

    catcher = group.eq("C").to_numpy()
    fld = RUNS_PER_OAA * col("range.oaa_per_600") + np.where(catcher, col("reliability.framing_runs") + col("reliability.blocking_runs") + 0.2 * col("arm.cs_above_average"), 0.0)
    run = col("baserunning.runs")
    pos = group.map(model.positional_adjustment).fillna(0.0).to_numpy(float) * n / FULL_SEASON_PA
    repl = group.map(model.replacement).fillna(0.0).to_numpy(float)
    hitter = total + fld + run + pos - repl * n / FULL_SEASON_PA
    innings = np.where(group.eq("SP"), STARTER_INNINGS, RELIEVER_INNINGS)
    mult = np.where(group.eq("SP"), 1.0, lev.reindex(counts.index).fillna(1.0).clip(0.5, 2.0).to_numpy(float))
    pitcher = -total * mult - repl * (n / model.pa_per_ip) / innings
    out = pd.DataFrame({
        "runs": np.where(is_bat, hitter, pitcher), "batting": np.where(is_bat, total, 0.0),
        "fielding": np.where(is_bat, fld, 0.0), "running": np.where(is_bat, run, 0.0), "n": n,
    }, index=counts.index)
    return out[out["n"] > 0]


def gather(as_of: date, model: ValueModel, season: int | None = None) -> dict:
    """Everything assemble_values needs, from the caches, the API, and the contracts table."""
    if not model.workload:
        raise ValueError("Model has no fitted workload projection; refit and validate before scoring")
    season = season or current_season(as_of)
    nxt = next_season(as_of, season)
    model.validate(nxt)
    seasons = [season - b for b in range(WINDOW_SEASONS)]
    weights = window_weights_by_season(model, season)
    population = bio.population(season, as_of)
    pos = bio.positions([s for s in seasons if (inputs.STATCAST_DIR / f"statcast_{s}.parquet").exists()])
    groups = bio.primary_position(pos, weights)
    # The current season is recomputed every run (Statcast cache and Savant boards); past seasons come from cache.
    frames = {s: inputs.season_inputs(s, refresh=(s == season)) for s in seasons if (inputs.STATCAST_DIR / f"statcast_{s}.parquet").exists()}
    counts = {s: inputs.outcome_counts(s, refresh=(s == season)) for s in frames}
    # Completed Triple-A seasons only: the current season's cache is not refreshed daily.
    _, season_end = bio.season_dates(season)
    completed = [s for s in frames if s < season or as_of > season_end]
    prior = minors.prior_counts({s: minors.pseudo_counts(s) for s in completed}, weights, set(frames), model.aaa_scale)
    players = pd.Index(pd.concat([f["player_id"] for f in frames.values()]).unique())
    for s in completed:
        frames[s] = pd.concat([frames[s], minors.translated_inputs(minors.season_inputs(s), model.aaa_input_offsets, model.shrink_strength, model.aaa_scale, players)], ignore_index=True)
    win = inputs.window(frames, weights)
    wc = inputs.window_counts(counts, weights)
    group_of = groups.set_index("player_id")["position_group"]
    rates = inputs.shrunk_rates(wc, group_of, model.shrink_strength, prior)
    il = bio.il_days(seasons, as_of)
    contracts = load_contracts(season)
    # Players Baseball-Reference does not list yet (this year's debuts) get an
    # estimated pre-arbitration schedule from their debut date.
    _, season_end = bio.season_dates(season)
    listed = set(contracts["player_id"])
    extra = [
        estimated_schedule(int(r.player_id), r.mlb_debut, season, season_end, as_of)
        for r in population.itertuples() if r.player_id not in listed and r.status != "free_agent"
    ]
    extra = [e for e in extra if len(e)]
    if extra:
        contracts = pd.concat([contracts, *extra], ignore_index=True)
    workload_lines = workload.load_lines(list(range(nxt - 5, nxt)))
    retained = set(population.loc[population["status"].ne("free_agent"), "player_id"])
    history = workload.history(workload_lines, nxt, retained)
    history = history[history["player_id"].isin(population["player_id"])]
    pt = workload.predict(history, population.set_index("player_id")["age"], model.workload,
                          nxt, max(nxt, int(contracts["season"].max())) if len(contracts) else nxt)
    if model.production:
        # Only the active season can change daily. Historical snapshots remain
        # stable unless explicitly refreshed for a new evaluation revision.
        current_source = production.source_paths(season)[-1]
        if not current_source.exists() or date.fromtimestamp(current_source.stat().st_mtime) < as_of:
            production.load_war(season, refresh=True)
        if "surgery_features" in model.production:
            surgery_source = bio.CACHE_DIR / "tommy_john_surgeries.csv"
            if not surgery_source.exists() or date.fromtimestamp(surgery_source.stat().st_mtime) < as_of:
                bio.surgery_dates(refresh=True)
        pt = production.add_forecasts(pt, model.production, nxt, int(pt["season"].max()))
        if model.price.get("profile_prices"):
            from v2.players.market_inputs import add_profiles

            pt = add_profiles(pt, nxt)
    else:
        pt = aging.add_projection_shifts(pt, history, population.set_index("player_id")["age"], model.production_aging, nxt)
    lev = win[win["input"] == "workload.leverage"].set_index("player_id")["value"]
    actual = (production.actual_war(season).mul(model.runs_per_win).rename("runs").to_frame() if model.production
              else _actual_runs(season, model, group_of, lev))
    # Draws of outcome rates, as run deltas against the shrunk point estimate, one row per player.
    wc_role = features.role_rows(wc, group_of).reset_index(drop=True)
    draws = features.dirichlet_draws(wc_role, group_of, model.shrink_strength, DRAWS, prior_counts=prior)
    lw = np.array([model.linear_weights[k] for k in OUTCOMES])
    point = rates.set_index(["player_id", "role"]).reindex(pd.MultiIndex.from_frame(wc_role[["player_id", "role"]]))
    base = point[list(OUTCOMES)].to_numpy(float) @ lw
    per = draws @ lw - base[:, None]
    is_bat = wc_role["role"].eq("bat").to_numpy()
    innings = np.where(wc_role["player_id"].map(group_of).eq("SP"), STARTER_INNINGS, RELIEVER_INNINGS)
    scale_draw = np.where(is_bat, FULL_SEASON_PA, -model.pa_per_ip * innings)
    draw_frame = pd.DataFrame(per * scale_draw[:, None], index=wc_role["player_id"].to_numpy())
    return {
        "population": population, "groups": groups, "rates": rates, "inputs": win, "il_days": il, "playing_time": pt,
        "contracts": contracts, "model": model, "next_season": nxt, "actual": actual, "draws": draw_frame, "as_of": as_of,
        "seasons": seasons,
    }


def score(as_of: date, model: ValueModel, *, write: bool, force: bool = False, players: list[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if write:
        model.require_market_validation()
    data = gather(as_of, model)
    seasons = data.pop("seasons")
    fp = fingerprint(as_of, data["population"], data["contracts"], model, seasons)
    if write and not force and LAST_SCORED.exists():
        last = json.loads(LAST_SCORED.read_text())
        if last.get("fingerprint") == fp:
            log.info(f"inputs unchanged since {last.get('as_of')}; skipping write")
            return pd.DataFrame(), pd.DataFrame()
    values_frame, traits_frame = assemble_values(**data)
    if players:
        values_frame = values_frame[values_frame["player_id"].isin(players)]
        traits_frame = traits_frame[traits_frame["player_id"].isin(players)]
    if write:
        from v2.players.publish import publish

        publish(values_frame, traits_frame)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        LAST_SCORED.write_text(json.dumps({"fingerprint": fp, "as_of": str(as_of)}))
    return values_frame, traits_frame


def _spot_checks(values_frame: pd.DataFrame) -> None:
    v = values_frame
    pre_arb = v[(v["contract_status"] == "contract") & (v["salary_current_musd"].fillna(99) < 1.0) & v["trade_value_musd"].notna()]
    if not pre_arb.empty:
        r = pre_arb.sort_values("trade_value_musd").iloc[-1]
        log.info(f"spot: largest pre-arb trade value: {r.player_name} ({r.position_group}, {r.team}) grade {r.grade} value ${r.trade_value_musd}M")
    long = v[(v["control_years"] >= 3) & v["trade_value_musd"].notna()]
    if not long.empty:
        r = long.sort_values("trade_value_musd").iloc[0]
        log.info(f"spot: most negative with 3+ years: {r.player_name} ({r.position_group}, {r.team}) grade {r.grade} value ${r.trade_value_musd}M owed ${r.owed_through_control_musd}M")
    fa = v[(v["contract_status"] == "free_agent") & v["grade"].notna()]
    if not fa.empty:
        r = fa.sort_values("grade").iloc[-1]
        log.info(f"spot: strongest pending free agent: {r.player_name} ({r.position_group}) grade {r.grade} market ${r.market_value_musd}M")
    log.info(f"rows: {len(v)} values, graded {v['grade'].notna().sum()}, free agents {(v['status'] == 'free_agent').sum()}, no contract {(v['contract_status'] == 'unknown').sum()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Score player values from the stored model.")
    ap.add_argument("--as-of", default=str(date.today()))
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--model", type=Path, help="Local model JSON instead of the database row")
    ap.add_argument("--player", type=int, nargs="*")
    args = ap.parse_args()
    as_of = date.fromisoformat(args.as_of)
    try:
        model = ValueModel.load(args.model) if args.model else load_latest_from_db()
    except RuntimeError as exc:
        log.warning(f"{exc}; nothing scored")
        return
    write = not args.no_write and writes_allowed() and not args.player
    if args.player and not args.no_write and writes_allowed():
        log.info("--player given; not writing")
    values_frame, traits_frame = score(as_of, model, write=write, force=args.force, players=args.player)
    if values_frame.empty:
        return
    _spot_checks(values_frame)
    if args.player:
        for r in values_frame.itertuples():
            print(json.dumps(r._asdict(), default=str, indent=1))
        print(traits_frame.to_string())
    else:
        print(f"{len(values_frame)} player_values rows, {len(traits_frame)} player_traits rows")


if __name__ == "__main__":
    setup_logging()
    main()
