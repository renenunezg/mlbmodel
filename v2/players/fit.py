"""The offseason fit: weights, curves, prices, and the evaluation report.

Development pairs end with the requested development season and its next-year
target. The following pair is the grade holdout, reported only when observed.
Market validation is separate. Publication requires both write authorization
and a validated model.

Usage:
    python -m v2.players.fit --last-dev-season 2024 [--no-publish]
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.stats import spearmanr

from backend.db import writes_allowed
from backend.log import setup_logging
from v2.data.pa_dataset import OUTCOMES
from v2.players import aging, bio, features, inputs, minors, pricing, production, workload
from v2.players import runs as runs_mod
from v2.players.inputs import INPUTS
from v2.players.model import MODELS_DIR, ValueModel, model_path, publish_model
from v2.players.savant import load_board

log = logging.getLogger(__name__)

PENALTIES = np.logspace(-2, 4, 61)
ZERO_COEFFICIENT = 1e-9
WINDOW_CANDIDATES = [(5.0, 4.0, 3.0), (6.0, 3.0, 1.0), (1.0, 0.0, 0.0), (4.0, 3.0, 2.0)]
SHRINK_CANDIDATES = [50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0]
MIN_FIT_SAMPLE = 100.0  # weighted PA or batters faced in season t and in t+1
MIN_TARGET_SAMPLE = 100.0
RELIABILITY_BINS = 5
MARCEL_WEIGHTS = (5.0, 4.0, 3.0)
MARCEL_REGRESSION_PA = 1200.0
WOBA = {"BB": 0.69, "HBP": 0.72, "1B": 0.89, "2B": 1.27, "3B": 1.62, "HR": 2.10}
# The OAA board filters by Statcast position code, not abbreviation.
POSITIONS_FOR_ADJUSTMENT = {position: str(slot) for slot, position in bio.FIELD_POSITIONS.items()}
# Trait target: (source, name, sign) where source is counts or inputs; sign orients higher-is-better.
TRAIT_TARGETS = {
    "contact": ("counts", "k_rate", -1.0), "power": ("counts", "iso", 1.0), "discipline": ("counts", "bb_rate", 1.0),
    "bbq": ("inputs", "bbq.xwobacon", 1.0), "platoon": ("inputs", "platoon.vs_lhp_delta", 1.0),
    "range": ("inputs", "range.oaa_per_600", 1.0), "arm": ("inputs", "arm.strength", 1.0),
    "reliability": ("inputs", "reliability.framing_runs", 1.0), "baserunning": ("inputs", "baserunning.runs", 1.0),
    "speed": ("inputs", "speed.sprint_speed", 1.0), "stuff": ("inputs", "stuff.whiff_rate", 1.0),
    "command": ("inputs", "command.bb_rate", -1.0), "batmiss": ("inputs", "batmiss.k_rate", 1.0),
    "contactmgmt": ("inputs", "contactmgmt.xwobacon_allowed", -1.0), "workload": ("counts", "n", 1.0),
}


# Ridge (the CFB grade method).


def _ridge(x: np.ndarray, y: np.ndarray, lower: np.ndarray, penalty: float) -> np.ndarray:
    width = x.shape[1]
    design = np.vstack([np.column_stack([np.ones(len(x)), x]), np.column_stack([np.zeros(width), np.sqrt(penalty) * np.eye(width)])])
    target = np.concatenate([y, np.zeros(width)])
    bounds = (np.concatenate([[-np.inf], lower]), np.full(width + 1, np.inf))
    return lsq_linear(design, target, bounds=bounds).x


def fit_ridge(pairs: dict[int, tuple[np.ndarray, np.ndarray]], lower: np.ndarray) -> tuple[np.ndarray, float]:
    """Intercept and coefficients; the penalty minimises leave-one-pair-out squared error."""
    seasons = sorted(pairs)
    if len(seasons) > 1:
        errors = []
        for penalty in PENALTIES:
            sse, n = 0.0, 0
            for held in seasons:
                x = np.vstack([pairs[s][0] for s in seasons if s != held])
                y = np.concatenate([pairs[s][1] for s in seasons if s != held])
                beta = _ridge(x, y, lower, penalty)
                xh, yh = pairs[held]
                residual = yh - beta[0] - xh @ beta[1:]
                sse += float(residual @ residual)
                n += len(residual)
            errors.append(sse / max(n, 1))
        penalty = float(PENALTIES[int(np.argmin(errors))])
    else:
        penalty = float(PENALTIES[0])
    x = np.vstack([pairs[s][0] for s in seasons])
    y = np.concatenate([pairs[s][1] for s in seasons])
    return _ridge(x, y, lower, penalty), penalty


# Season data.


def season_rates(counts: pd.DataFrame) -> pd.DataFrame:
    """Outcome rates, sample, and the summary rates each trait target uses, from an outcome-count frame."""
    c = counts
    n = c[list(OUTCOMES)].sum(axis=1)
    out = c[["player_id", "role"]].copy()
    for k in OUTCOMES:
        out[k] = c[k] / n.where(n > 0)
    out["n"] = n
    out["k_rate"], out["bb_rate"], out["hr_rate"] = out["K"], out["BB"], out["HR"]
    out["iso"] = out["2B"] + 2 * out["3B"] + 3 * out["HR"]
    out["woba"] = sum(WOBA[k] * out[k] for k in WOBA)
    return out


class SeasonData:
    """Cached per-season frames: inputs, counts, positions."""

    def __init__(self, seasons: list[int]):
        self.inputs = {s: inputs.season_inputs(s) for s in seasons}
        self.counts = {s: inputs.outcome_counts(s) for s in seasons}
        self.pseudo = {s: minors.pseudo_counts(s) for s in seasons}
        self.aaa_inputs = {s: minors.season_inputs(s) for s in seasons}
        self.positions = bio.positions(seasons)
        self.linear_weights = {s: runs_mod.linear_weights(s) for s in seasons}
        self.pa_per_ip = {s: runs_mod.pa_per_inning(s) for s in seasons}
        self._ages: dict[int, pd.Series] = {}

    def ages(self, season: int) -> pd.Series:
        if season not in self._ages:
            self._ages[season] = bio.ages(season)
        return self._ages[season]

    def window(self, t: int, model: ValueModel) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame | None]:
        """(inputs, MLB counts, position groups, Triple-A prior counts) over the window ending in ``t``.

        The model supplies the window weights, the Triple-A scale and input offsets, and the shrink strength.
        """
        weights = {t - back: w for back, w in model.window_weights.items() if t - back in self.inputs}
        players = pd.Index(pd.concat([self.inputs[s]["player_id"] for s in weights]).unique())
        frames = {s: pd.concat([self.inputs[s], minors.translated_inputs(self.aaa_inputs[s], model.aaa_input_offsets, model.shrink_strength, model.aaa_scale, players)],
                               ignore_index=True) for s in weights}
        win = inputs.window(frames, weights)
        counts = inputs.window_counts({s: self.counts[s] for s in weights}, weights)
        prior = minors.prior_counts(self.pseudo, weights, set(weights), model.aaa_scale)
        pos = self.positions[self.positions["season"].isin(weights)]
        groups = bio.primary_position(pos, weights).set_index("player_id")["position_group"]
        return win, counts, groups, prior

    def season_rates(self, s: int) -> pd.DataFrame:
        return season_rates(self.counts[s])

    def targets(self, t1: int, groups: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(trait targets wide by trait, group target) for season t1, indexed by (player_id, role)."""
        rates = self.season_rates(t1)
        lw = self.linear_weights[t1]
        w = np.array([lw[k] for k in OUTCOMES])
        rates["bat_runs"] = rates[list(OUTCOMES)].values @ w * runs_mod.FULL_SEASON_PA
        board = self.inputs[t1].pivot_table(index="player_id", columns="input", values="value", aggfunc="first")
        rates = rates.set_index("player_id")
        traits = pd.DataFrame(index=rates.index)
        for trait, (source, name, sign) in TRAIT_TARGETS.items():
            if source == "counts":
                traits[trait] = sign * rates[name]
            else:
                traits[trait] = sign * board[name].reindex(rates.index) if name in board.columns else np.nan
        if "reliability.blocking_runs" in board.columns:
            traits["reliability"] = traits["reliability"].fillna(0) + board["reliability.blocking_runs"].reindex(rates.index).fillna(0)
        group = groups.reindex(rates.index)
        role = rates["role"]
        fld = (runs_mod.RUNS_PER_OAA * board.get("range.oaa_per_600", pd.Series(dtype=float)).reindex(rates.index).fillna(0.0)
               + board.get("reliability.framing_runs", pd.Series(dtype=float)).reindex(rates.index).fillna(0.0).where(group.eq("C"), 0.0)
               + board.get("reliability.blocking_runs", pd.Series(dtype=float)).reindex(rates.index).fillna(0.0).where(group.eq("C"), 0.0))
        run = board.get("baserunning.runs", pd.Series(dtype=float)).reindex(rates.index).fillna(0.0)
        hitter_total = rates["bat_runs"] + (fld + run) * runs_mod.FULL_SEASON_PA / rates["n"].where(rates["n"] > 0)
        per_pa = rates[list(OUTCOMES)].values @ w
        innings = np.where(group.eq("SP"), runs_mod.STARTER_INNINGS, runs_mod.RELIEVER_INNINGS)
        pitcher_total = pd.Series(-per_pa * self.pa_per_ip[t1] * innings, index=rates.index)
        target = pd.DataFrame({"role": role, "n": rates["n"], "target": np.where(role.eq("bat"), hitter_total, pitcher_total),
                               "bat_runs": rates["bat_runs"], "woba": rates["woba"]})
        traits["role"] = role
        return traits.reset_index(), target.reset_index()


# Pair assembly.


def _by_role(frame: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    """One row per player: the role his position group implies (two-way players have both)."""
    expected = np.where(frame["player_id"].map(group).isin(features.PITCHER_GROUPS), "pit", "bat")
    return frame[frame["role"].values == expected].set_index("player_id")


def _z_within(values: pd.Series, key: pd.Series) -> pd.Series:
    g = values.groupby(key)
    mean, sd = g.transform("mean"), g.transform("std")
    return ((values - mean) / sd.where(sd > 0)).clip(-features.Z_CAP, features.Z_CAP)


def build_pair(data: SeasonData, t: int, model: ValueModel) -> dict:
    """Everything a development pair (t, t+1) contributes: features at t, targets at t+1."""
    win, counts, groups, prior = data.window(t, model)
    values, samples = features.wide_inputs(win)
    group = groups.reindex(values.index).fillna("DH")
    rates = inputs.shrunk_rates(counts, groups, model.shrink_strength, prior)
    pool = pd.Series(features.pool_of(group), index=values.index)
    z = features.oriented_z(values, pool)
    sample = features.role_rows(rates, group).set_index("player_id")["n"].reindex(values.index).fillna(0.0)
    traits_next, target_next = data.targets(t + 1, groups)
    leverage = values["workload.leverage"] if "workload.leverage" in values.columns else None
    shrunk = features.shrunk_runs(rates, model, group, leverage).reindex(values.index)
    return {"t": t, "values": values, "samples": samples, "z": z, "pool": pool, "group": group, "rates": rates,
            "sample": sample, "traits_next": traits_next, "target_next": target_next, "shrunk": shrunk, "counts": counts,
            "age": data.ages(t + 1).reindex(values.index)}


def choose_window_and_shrink(data: SeasonData, dev_first: list[int]) -> tuple[dict[int, float], float, float]:
    """Window, shrink, and Triple-A scale by next-season multinomial log-likelihood of shrunk rates over the development pairs."""
    best, best_ll = None, -np.inf
    candidate = ValueModel.empty([], (0, 0))
    # Every candidate is judged on the same players: those with counts in season t
    # (the single-season window's population) and a full sample in t+1.
    targets = {}
    for t in dev_first:
        present = data.counts[t].set_index(["player_id", "role"]).index
        nxt = data.counts[t + 1].set_index(["player_id", "role"])
        nxt = nxt[nxt[list(OUTCOMES)].sum(axis=1) >= MIN_TARGET_SAMPLE]
        targets[t] = nxt.loc[nxt.index.intersection(present)]
    for cand in WINDOW_CANDIDATES:
        offsets = {i: w for i, w in enumerate(cand)}
        for aaa_scale in minors.SCALE_CANDIDATES:
            candidate.window_weights, candidate.aaa_scale = offsets, aaa_scale
            windows = {t: data.window(t, candidate) for t in dev_first}
            for strength in SHRINK_CANDIDATES:
                ll = 0.0
                for t in dev_first:
                    _, counts, groups, prior = windows[t]
                    rates = inputs.shrunk_rates(counts, groups, strength, prior).set_index(["player_id", "role"])
                    nxt = targets[t]
                    r = rates.reindex(nxt.index)[list(OUTCOMES)].fillna(1.0 / len(OUTCOMES)).values
                    ll += float((nxt[list(OUTCOMES)].values * np.log(np.clip(r, 1e-9, 1))).sum())
                log.info(f"window {cand} triple-a {aaa_scale} shrink {strength}: log-lik {ll:,.0f}")
                if ll > best_ll:
                    best, best_ll = (offsets, strength, aaa_scale), ll
    return best


def fit_trait_weights(pairs: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for trait, spec in features.TRAITS.items():
        cols = [name for name, s in INPUTS.items() if s.trait == trait]
        design = {}
        for p in pairs:
            members = p["pool"] == spec.pool
            present = [c for c in cols if c in p["z"].columns and p["z"].loc[members, c].notna().mean() >= features.MIN_INPUT_AVAILABILITY]
            if not present:
                continue
            tn = _by_role(p["traits_next"], p["group"])
            y = tn[trait].reindex(p["z"].index[members])
            ok = members.copy()
            ok.loc[members] = (p["sample"][members] >= MIN_FIT_SAMPLE) & y.notna().values
            nxt_sample = _by_role(p["target_next"], p["group"])["n"].reindex(p["z"].index).fillna(0)
            ok &= nxt_sample >= MIN_TARGET_SAMPLE
            if trait in ("reliability",) :
                ok &= p["group"].eq("C")
            if ok.sum() < 30:
                continue
            x = p["z"].loc[ok, present].fillna(0.0).reindex(columns=cols).fillna(0.0).to_numpy(float)
            yz = _z_within(tn[trait].reindex(p["z"].index[ok]), pd.Series("all", index=p["z"].index[ok])).fillna(0.0).to_numpy(float)
            design[p["t"]] = (x, yz)
        if not design:
            for c in cols:
                rows.append({"pool": spec.pool, "trait": trait, "input": c, "weight": 0.0, "status": "unmeasured"})
            continue
        beta, penalty = fit_ridge(design, np.zeros(len(cols)))
        coef = pd.Series(beta[1:], index=cols).where(lambda s: s.abs() > ZERO_COEFFICIENT, 0.0)
        total = coef.abs().sum()
        for c in cols:
            w = float(coef[c] / total) if total > 0 else 0.0
            rows.append({"pool": spec.pool, "trait": trait, "input": c, "weight": w, "status": "weighted" if w > 0 else "unweighted", "penalty": penalty})
    return rows


def _fit_pools(p: dict) -> pd.Series:
    return pd.Series(np.where(p["group"].isin(features.PITCHER_GROUPS), p["group"], features.HITTER_POOL), index=p["group"].index)


def grade_design(p: dict, model: ValueModel) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    comps, _ = features.trait_composites(p["z"], p["pool"], model)
    x = features.grade_features(comps, p["shrunk"], p["pool"], p["age"])
    tn = _by_role(p["target_next"], p["group"])
    y = tn["target"].reindex(x.index)
    ok = (p["sample"] >= MIN_FIT_SAMPLE) & (tn["n"].reindex(x.index).fillna(0) >= MIN_TARGET_SAMPLE) & y.notna() & x["shrunk_runs"].notna()
    return x, y, ok


def fit_grade_weights(pairs: list[dict], model: ValueModel) -> list[dict]:
    rows: list[dict] = []
    feature_cols = [t for t in features.TRAITS] + ["shrunk_runs", *features.AGE_FEATURES]
    for pool_name in (features.HITTER_POOL, "SP", "RP"):
        design = {}
        cols = [c for c in feature_cols if c not in features.TRAITS
                or features.TRAITS[c].pool == (features.PITCHER_POOL if pool_name != features.HITTER_POOL else features.HITTER_POOL)]
        for p in pairs:
            x, y, ok = grade_design(p, model)
            ok &= _fit_pools(p) == pool_name
            if ok.sum() < 30:
                continue
            design[p["t"]] = (x.loc[ok, cols].fillna(0.0).to_numpy(float), y[ok].to_numpy(float))
        if not design:
            continue
        # Traits and shrunk runs are oriented higher-is-better; age may slope either way.
        lower = np.array([-np.inf if c in features.AGE_FEATURES else 0.0 for c in cols])
        beta, penalty = fit_ridge(design, lower)
        rows.append({"pool": pool_name, "feature": "intercept", "coefficient": float(beta[0]), "penalty": penalty})
        for c, b in zip(cols, beta[1:], strict=True):
            rows.append({"pool": pool_name, "feature": c, "coefficient": float(b) if abs(b) > ZERO_COEFFICIENT else 0.0})
    return rows


def spearman(a: pd.Series, b: pd.Series) -> float:
    joined = pd.concat([a, b], axis=1).dropna()
    if len(joined) < 10:
        return float("nan")
    return float(spearmanr(joined.iloc[:, 0], joined.iloc[:, 1]).correlation)


def reliability_k(pairs: list[dict], model: ValueModel) -> dict[str, float]:
    """k per trait and grade from year-over-year rank correlation by sample bucket: r = (n/(n+k))^2."""
    series: dict[str, list[tuple[pd.Series, pd.Series, pd.Series]]] = {}
    for a, b in zip(pairs[:-1], pairs[1:], strict=True):
        if b["t"] != a["t"] + 1:
            continue
        ca, _ = features.trait_composites(a["z"], a["pool"], model)
        cb, _ = features.trait_composites(b["z"], b["pool"], model)
        xa = features.grade_features(ca, a["shrunk"], a["pool"], a["age"])
        xb = features.grade_features(cb, b["shrunk"], b["pool"], b["age"])
        ga, gb = features.project_runs(xa, a["group"], model), features.project_runs(xb, b["group"], model)
        pools_a = _fit_pools(a)
        for trait in features.TRAITS:
            series.setdefault(trait, []).append((ca[trait], cb[trait].reindex(ca.index), np.minimum(a["sample"], b["sample"].reindex(a["sample"].index))))
        for pool_name in (features.HITTER_POOL, "SP", "RP"):
            m = pools_a == pool_name
            series.setdefault(f"grade_{pool_name}", []).append((ga[m], gb.reindex(ga.index)[m], np.minimum(a["sample"], b["sample"].reindex(a["sample"].index))[m]))
    out: dict[str, float] = {}
    for name, parts in series.items():
        x = pd.concat([p[0] for p in parts]).reset_index(drop=True)
        y = pd.concat([p[1] for p in parts]).reset_index(drop=True)
        n = pd.concat([p[2] for p in parts]).reset_index(drop=True)
        frame = pd.DataFrame({"x": x, "y": y, "n": n}).dropna()
        frame = frame[frame["n"] >= MIN_FIT_SAMPLE]
        if len(frame) < 50:
            out[name] = 0.0
            continue
        frame["bin"] = pd.qcut(frame["n"], RELIABILITY_BINS, labels=False, duplicates="drop")
        ks = []
        for _, g in frame.groupby("bin"):
            r = spearman(g["x"], g["y"])
            if r == r and r > 0.02:
                ks.append(float(g["n"].median()) * (1 / np.sqrt(r) - 1))
        out[name] = float(np.median(ks)) if ks else 0.0
    return out


def marcel(rates_by_season: dict[int, pd.DataFrame], linear_weights: dict[int, dict[str, float]], t: int, role: str) -> pd.Series:
    """Three-season weighted wOBA (hitters) or runs allowed per PA (pitchers), regressed to 1,200 PA of league average.

    ``rates_by_season`` holds ``season_rates`` frames; season ``t`` may be a partial season.
    """
    num, den = None, None
    for back, w in enumerate(MARCEL_WEIGHTS):
        s = t - back
        if s not in rates_by_season:
            continue
        r = rates_by_season[s]
        r = r[r["role"] == role].set_index("player_id")
        lw = linear_weights[s]
        stat = r["woba"] if role == "bat" else -(r[list(OUTCOMES)].values @ np.array([lw[k] for k in OUTCOMES]))
        stat = pd.Series(np.asarray(stat), index=r.index)
        contrib = w * stat * r["n"]
        num = contrib if num is None else num.add(contrib, fill_value=0.0)
        den = (w * r["n"]) if den is None else den.add(w * r["n"], fill_value=0.0)
    league = float((num / den).mean())
    return (num + MARCEL_REGRESSION_PA * league) / (den + MARCEL_REGRESSION_PA)


def evaluate(pairs: list[dict], data: SeasonData, model: ValueModel, holdout: dict | None, xwoba_boards: dict[int, pd.Series]) -> dict:
    """Leave-one-pair-out rank correlations of the Grade and baselines with next-season runs."""
    report: dict = {"dev_pairs": {}, "holdout": {}}
    for pool_name in (features.HITTER_POOL, "SP", "RP"):
        rows = []
        for held in pairs:
            others = [p for p in pairs if p is not held]
            sub = ValueModel.from_payload(model.to_payload())
            sub.trait_weights = fit_trait_weights(others)
            sub.grade_weights = fit_grade_weights(others, sub)
            rows.append(_score_pair(held, data, sub, pool_name, xwoba_boards))
        frame = pd.DataFrame(rows)
        report["dev_pairs"][pool_name] = {c: float(frame[c].mean()) for c in frame.columns if c != "t"} | {"pairs": len(frame)}
        if holdout is not None:
            report["holdout"][pool_name] = _score_pair(holdout, data, model, pool_name, xwoba_boards)
    return report


def _score_pair(p: dict, data: SeasonData, model: ValueModel, pool_name: str, xwoba_boards: dict[int, pd.Series]) -> dict:
    x, y, ok = grade_design(p, model)
    ok &= _fit_pools(p) == pool_name
    proj = features.project_runs(x, p["group"], model)
    role = "pit" if pool_name != features.HITTER_POOL else "bat"
    prior = data.season_rates(p["t"])
    prior = prior[prior["role"] == role].set_index("player_id")
    lw = data.linear_weights[p["t"]]
    prior_runs = pd.Series(np.asarray(prior[list(OUTCOMES)].values @ np.array([lw[k] for k in OUTCOMES])), index=prior.index)
    if role == "pit":
        prior_runs = -prior_runs
    out = {
        "t": p["t"], "n": int(ok.sum()),
        "grade": spearman(proj[ok], y[ok]),
        "shrunk_only": spearman(x.loc[ok, "shrunk_runs"], y[ok]),
        "prior_production": spearman(prior_runs.reindex(x.index[ok]), y[ok]),
        "marcel": spearman(marcel({s: data.season_rates(s) for s in data.counts if p["t"] - 2 <= s <= p["t"]},
                                  data.linear_weights, p["t"], role).reindex(x.index[ok]), y[ok]),
    }
    if role == "bat" and p["t"] in xwoba_boards:
        out["prior_xwoba"] = spearman(xwoba_boards[p["t"]].reindex(x.index[ok]), y[ok])
    return out


def fit_positional_adjustment(data: SeasonData, seasons: list[int]) -> tuple[dict[str, float], int]:
    boards = []
    for season in seasons:
        for pos, code in POSITIONS_FOR_ADJUSTMENT.items():
            try:
                b = load_board("oaa", season, pos=code)
            except Exception as exc:  # a missing position board must not stop the fit
                log.warning(f"oaa {pos} {season}: {exc}")
                continue
            if b.empty:
                continue
            boards.append(pd.DataFrame({"player_id": b["player_id"], "season": season, "position": pos,
                                        "outs_above_average": pd.to_numeric(b["outs_above_average"], errors="coerce")}))
    if not boards:
        return dict(runs_mod.PUBLISHED_POSITIONAL_RUNS), 0
    return runs_mod.positional_adjustment(data.positions[data.positions["season"].isin(seasons)], pd.concat(boards, ignore_index=True))


def fit(last_dev_season: int = 2024, holdout_pair: tuple[int, int] | None = None, *,
        calibration_model: ValueModel | None = None, market_cohort: dict | None = None,
        calibration_history: list[tuple[ValueModel, dict]] | None = None) -> tuple[ValueModel, dict]:
    holdout_pair = holdout_pair or (last_dev_season+1, last_dev_season+2)
    if holdout_pair[0] <= last_dev_season or holdout_pair[1] != holdout_pair[0]+1:
        raise ValueError("Grade development and holdout pairs overlap or are not consecutive")
    if calibration_history and calibration_model is not None:
        raise ValueError("Choose either chronological calibration history or a single cohort")
    dev_first = list(range(2021, last_dev_season + 1))
    dev_seasons = list(range(2021, last_dev_season + 2))
    seasons = list(range(2021 - 2, holdout_pair[1] + 1))
    seasons = [s for s in seasons if (inputs.STATCAST_DIR / f"statcast_{s}.parquet").exists()]
    data = SeasonData(seasons)
    model = ValueModel.empty(dev_seasons, holdout_pair)
    model.fitted_at = datetime.now(UTC).isoformat()

    offsets, strength, aaa_scale = choose_window_and_shrink(data, dev_first)
    model.window_weights, model.shrink_strength, model.aaa_scale = offsets, strength, aaa_scale
    model.aaa_input_offsets = minors.input_offsets({s: data.inputs[s] for s in dev_seasons}, {s: data.aaa_inputs[s] for s in dev_seasons})
    model.linear_weights = data.linear_weights[last_dev_season]
    model.pa_per_ip = data.pa_per_ip[last_dev_season]
    model.runs_per_win = runs_mod.runs_per_win(last_dev_season)
    log.info(f"window {offsets}, shrink {strength}, triple-a {aaa_scale}, runs/win {model.runs_per_win:.2f}")

    pairs = [build_pair(data, t, model) for t in dev_first]
    model.trait_weights = fit_trait_weights(pairs)
    model.grade_weights = fit_grade_weights(pairs, model)
    model.reliability_k = reliability_k(pairs, model)
    model.age_curves = aging.fit_curves_from_cache(last_dev_season)
    model.positional_adjustment, model.positional_seasons = fit_positional_adjustment(data, dev_first)

    # Replacement level: what the fringe actually produced in the last development season,
    # pooled over the development pairs so one season's noise does not set the bar.
    repl_parts = []
    for p in pairs:
        tn = _by_role(p["target_next"], p["group"])
        g = p["group"].reindex(tn.index)
        pos_adj = g.map(model.positional_adjustment).fillna(0.0).where(~g.isin(features.PITCHER_GROUPS), 0.0)
        repl_parts.append(pd.DataFrame({"runs": tn["target"] + pos_adj, "pt": tn["n"], "group": g}).dropna())
    fringe = pd.concat(repl_parts)
    measured_replacement = runs_mod.replacement_level(fringe["runs"], fringe["pt"], fringe["group"])
    model.replacement = {g: runs_mod.CONVENTIONAL_REPLACEMENT.get(g, runs_mod.CONVENTIONAL_REPLACEMENT["hitter"]) for g in measured_replacement}
    # OAA is already relative to each position's average fielder, so the multi-position
    # fit cannot see the spectrum; the published scale stands and the fit is reported.
    fitted_positional, model.positional_seasons = model.positional_adjustment, model.positional_seasons
    model.positional_adjustment = dict(runs_mod.PUBLISHED_POSITIONAL_RUNS)

    # Economic forecasts use completed development targets, with absent players
    # retained as zeros. The following grade holdout target remains excluded.
    lines = workload.load_lines(list(range(workload.FIRST_HISTORY_SEASON, last_dev_season + 2)))
    rosters = {year: set(bio.october_roster(year-1)["player_id"])
               for year in range(workload.FIRST_ORIGIN, last_dev_season + 2)}
    panel = workload.training_frame(lines, rosters, last_dev_season + 1)
    model.workload = workload.fit(panel, last_dev_season + 1)
    model.production_aging = {}
    war_panel = production.panel(production.load_lines(last_dev_season + 1), last_dev_season + 1)
    model.production = production.fit(war_panel, last_dev_season + 1)
    model.model_version = "player-value-v2-fwar"

    # Contract transactions and frozen forecasts are distinct from current payroll.
    # Never infer free agency or arbitration from how the team acquired the player.
    calibration = {"status": "uncalibrated", "reason": "Supply a frozen pre-signing model and verified signing cohort"}
    model.price["source"] = "uncalibrated"
    model.arb_fractions = dict(pricing.FALLBACK_ARB_FRACTIONS)
    if calibration_model is not None and market_cohort is not None:
        from v2.players.market import calibrate

        if market_cohort["first_season"] >= holdout_pair[1]:
            raise ValueError("Dollar calibration cannot consume the market holdout season")
        price, calibration = calibrate(calibration_model, market_cohort)
        model.price = asdict(price)
    elif calibration_history:
        from v2.players.arbitration import calibrate as calibrate_arbitration
        from v2.players.market import calibrate_history

        cohorts = [cohort for _, cohort in calibration_history]
        if any(c["first_season"] >= holdout_pair[1] for c in cohorts):
            raise ValueError("Dollar calibration cannot consume the market holdout season")
        if len({c["first_season"] for c in cohorts}) != len(cohorts):
            raise ValueError("Duplicate signing winter in calibration history")
        frozen = {c["first_season"]: m for m, c in calibration_history}
        price, calibration = calibrate_history(cohorts, frozen)
        model.arbitration, calibration["arbitration"] = calibrate_arbitration(frozen)
        model.price = asdict(price)

    # Evaluation: leave-one-pair-out on development, then the holdout once.
    xwoba_boards = {}
    for s in dev_first + [holdout_pair[0]]:
        try:
            b = load_board("expected_statistics_batter", s)
            xwoba_boards[s] = pd.to_numeric(b.set_index("player_id")["est_woba"], errors="coerce")
        except Exception as exc:
            log.warning(f"xwoba board {s}: {exc}")
    holdout = build_pair(data, holdout_pair[0], model) if holdout_pair[1] in data.inputs else None
    report = evaluate(pairs, data, model, holdout, xwoba_boards)
    report.update({
        "window_weights": offsets, "shrink_strength": strength, "aaa_scale": aaa_scale, "runs_per_win": model.runs_per_win,
        "price": model.price, "arb_fractions": model.arb_fractions, "dollar_calibration": calibration,
        "replacement": model.replacement, "measured_fringe_replacement": measured_replacement,
        "positional_adjustment": model.positional_adjustment, "fitted_positional_runs": fitted_positional,
        "positional_seasons": model.positional_seasons, "reliability_k": model.reliability_k,
        "zero_weight_inputs": [f"{r['pool']}:{r['trait']}:{r['input']}" for r in model.trait_weights if r["weight"] == 0],
        "grade_weights": model.grade_weights, "trait_weights": model.trait_weights,
    })
    model.evaluation = {k: report[k] for k in ("dev_pairs", "holdout", "dollar_calibration")}
    model.evaluation["market"] = {"status": "not_validated"}
    return model, report


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit the player valuation model.")
    ap.add_argument("--last-dev-season", type=int, default=2024)
    ap.add_argument("--no-publish", action="store_true")
    ap.add_argument("--calibration-model", type=Path, help="Frozen model trained strictly before the signing cohort")
    ap.add_argument("--calibration-history", type=Path, help="JSON list of {cohort, model} paths with frozen pre-signing forecasts")
    ap.add_argument("--market-contracts", type=Path, default=Path(__file__).parent / "data/free_agent_signings_2025.json")
    args = ap.parse_args()
    if args.calibration_history and args.calibration_model:
        ap.error("Choose --calibration-history or --calibration-model")
    history = ([(ValueModel.load(Path(item["model"])), json.loads(Path(item["cohort"]).read_text()))
                for item in json.loads(args.calibration_history.read_text())] if args.calibration_history else None)
    model, report = fit(args.last_dev_season,
                        calibration_model=ValueModel.load(args.calibration_model) if args.calibration_model else None,
                        market_cohort=json.loads(args.market_contracts.read_text()) if args.calibration_model else None,
                        calibration_history=history)
    path = model.save(model_path(args.last_dev_season))
    (MODELS_DIR / f"evaluation_{args.last_dev_season}.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    log.info(f"saved {path}")
    for pool_name, rows in report["dev_pairs"].items():
        log.info(f"dev {pool_name}: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in rows.items()))
    for pool_name, rows in report["holdout"].items():
        log.info(f"holdout {pool_name}: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in rows.items()))
    log.info(f"price {model.price}, arb {model.arb_fractions}, replacement {model.replacement}, positional {model.positional_adjustment} ({model.positional_seasons} seasons)")
    if not args.no_publish and writes_allowed():
        publish_model(model)


if __name__ == "__main__":
    setup_logging()
    main()
