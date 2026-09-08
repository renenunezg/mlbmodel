"""Chronological acceptance using frozen pregame predictive distributions.

Compare two JSON forecast exports with --candidate and --baseline. Exports use
the columns returned by residual.load_frozen_games, including both contexts.
No actual bullpen sequences or retrospectively fitted posteriors are allowed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from backend.db import engine
from v2.market_model.residual import validated_forecasts


def load_export(path: Path) -> pd.DataFrame:
    """Join local score exports to results, retaining their original provenance."""
    frame = pd.DataFrame(json.loads(path.read_text()))
    if "prediction_updated_at" not in frame:
        return frame  # Already a joined frozen-forecast export.
    if frame.empty or frame.duplicated(["game_pk", "team"]).any():
        raise ValueError("Export must contain unique per-team forecasts")
    with engine.connect() as conn:
        games = pd.read_sql(text("""
            SELECT game_pk, game_date, start_time, home_team, away_team, home_score, away_score
            FROM games WHERE game_pk=ANY(:ids) AND status='Final'
              AND home_score IS NOT NULL AND away_score IS NOT NULL
        """), conn, params={"ids": [int(g) for g in frame.game_pk.unique()]})
    rows = []
    for game in games.to_dict("records"):
        pair = frame[frame.game_pk == game["game_pk"]].set_index("team")
        if any(t not in pair.index for t in (game["home_team"], game["away_team"])):
            continue
        h, a = pair.loc[game["home_team"]], pair.loc[game["away_team"]]
        if any(pd.Timestamp(r.date).date() != pd.Timestamp(game["game_date"]).date() for r in (h, a)):
            continue
        rows.append({**game, "home_prediction_at": h.prediction_updated_at,
                     "away_prediction_at": a.prediction_updated_at,
                     "prediction_context": h.prediction_context,
                     "away_prediction_context": a.prediction_context})
    return pd.DataFrame(rows)


def _distribution(context: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    histogram = context[name]
    values = np.array([int(v) for v in histogram], dtype=int)
    probabilities = np.array(list(histogram.values()), dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1):
        raise ValueError(f"Invalid {name}")
    return values, probabilities


def _crps(values: np.ndarray, p: np.ndarray, actual: float) -> float:
    return float(p @ np.abs(values - actual) - .5 * (p[:, None] * p * np.abs(values[:, None] - values)).sum())


def forecast_losses(games: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in games.to_dict("records"):
        c = row["prediction_context"]
        h, hp = _distribution(c, "home_run_distribution")
        a, ap = _distribution(c, "away_run_distribution")
        m, mp = _distribution(c, "margin_distribution")
        margin = row["home_score"] - row["away_score"]
        home_prob = np.clip(c["raw_home_win_prob"], 1e-7, 1 - 1e-7)
        won = float(margin > 0)
        records.append({
            "game_pk": row["game_pk"], "game_date": row["game_date"], "opener": bool(c["opener"]),
            "runs_mae": (abs(h @ hp - row["home_score"]) + abs(a @ ap - row["away_score"])) / 2,
            "runs_crps": (_crps(h, hp, row["home_score"]) + _crps(a, ap, row["away_score"])) / 2,
            "margin_crps": _crps(m, mp, margin),
            "runline_brier": float((mp[m > 1.5].sum() - (margin > 1.5)) ** 2),
            "tail_brier": float(((hp[h >= 10].sum() - (row["home_score"] >= 10)) ** 2
                                 + (ap[a >= 10].sum() - (row["away_score"] >= 10)) ** 2) / 2),
            "win_brier": (home_prob - won) ** 2,
            "win_log_loss": -(won * np.log(home_prob) + (1 - won) * np.log(1 - home_prob)),
        })
    return pd.DataFrame(records)


def compare_forecasts(candidate: pd.DataFrame, baseline: pd.DataFrame, min_games: int = 100) -> dict:
    candidate = validated_forecasts(candidate)
    baseline = validated_forecasts(baseline)
    if len(candidate) < min_games or len(baseline) < min_games:
        raise ValueError(f"Need at least {min_games} forecasts per model with verified pregame provenance")
    for frame in (candidate, baseline):
        if frame.game_pk.duplicated().any() or frame.model_version.nunique() != 1:
            raise ValueError("Use one forecast per game from one identified model version")
    # Same games and results, rather than comparing a sampled slate with a
    # different season-wide run distribution.
    pairs = candidate.merge(baseline, on=["game_pk", "game_date", "home_team", "away_team",
                                          "home_score", "away_score"], suffixes=("_c", "_b"))
    if len(pairs) < min_games:
        raise ValueError("Too few common games for a paired comparison")
    ids = set(pairs.game_pk)
    candidate = candidate[candidate.game_pk.isin(ids)].sort_values("game_pk")
    baseline = baseline[baseline.game_pk.isin(ids)].sort_values("game_pk")
    c = forecast_losses(candidate).set_index("game_pk")
    b = forecast_losses(baseline).set_index("game_pk")
    metrics = [k for k in c if k not in ("game_date", "opener")]
    result = {}
    rng = np.random.default_rng(20260908)
    for name, mask in (("all", np.ones(len(c), dtype=bool)), ("opener", c.opener | b.opener)):
        cc, bb = c.loc[mask], b.loc[mask]
        report = {"games": len(cc), "metrics": {}, "sufficient": len(cc) >= (min_games if name == "all" else 20)}
        if len(cc):
            delta = cc[metrics] - bb[metrics]
            delta["game_date"] = cc.game_date
            groups = [g[metrics].to_numpy() for _, g in delta.groupby("game_date")]
            bootstrap = np.array([
                np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean(axis=0)
                for _ in range(2000)
            ])
            intervals = np.quantile(bootstrap, [.025, .975], axis=0)
            for i, metric in enumerate(metrics):
                report["metrics"][metric] = {
                    "candidate": float(cc[metric].mean()), "baseline": float(bb[metric].mean()),
                    "delta_ci_95": intervals[:, i].tolist(),
                }
        result[name] = report
    # Insufficient opener evidence is inconclusive, never a promotion pass.
    result["all_pass"] = bool(all(r["sufficient"] and all(
        m["delta_ci_95"][1] <= 0 for m in r["metrics"].values()
    ) for r in (result["all"], result["opener"])))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    args = parser.parse_args()
    report = compare_forecasts(load_export(args.candidate), load_export(args.baseline))
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["all_pass"] else 1)


if __name__ == "__main__":
    main()
