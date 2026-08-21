"""Chronological market-offset evaluation for candidate pregame feature blocks."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss
from sqlalchemy import text

from backend.db import engine
from backend.strategy import EV_THRESHOLDS
from v2.market_model.residual import american_to_prob


TEAM_WINDOW = 20
TEAM_PRIOR_GAMES = 10
MIN_EDGE_BETS = 50

FEATURE_BLOCKS = {
    "market_calibration": ["market_logit_correction"],
    "simulator": ["sim_disagreement"],
    "team_residuals": ["offense_residual_diff", "defense_residual_diff"],
    "team_form": ["run_margin_form_diff", "win_form_diff"],
    "bullpen_rest": ["bullpen_rest_diff"],
    "uncertainty": ["uncertainty_width"],
    "lineup_queue_state": ["lineup_live", "queue_live"],
    "posterior_age": ["posterior_age_days"],
    "prediction_context": [
        "uncertainty_width",
        "lineup_live",
        "queue_live",
        "posterior_age_days",
    ],
    "all": [
        "sim_disagreement",
        "offense_residual_diff",
        "defense_residual_diff",
        "run_margin_form_diff",
        "win_form_diff",
        "bullpen_rest_diff",
        "uncertainty_width",
        "lineup_live",
        "queue_live",
        "posterior_age_days",
    ],
}


def _logit(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_feature_games(start: str, end: str) -> pd.DataFrame:
    query = text("""
        WITH paired_candidates AS (
            SELECT g.game_pk, oh.book,
                   GREATEST(oh.scraped_at, oa.scraped_at) AS market_quote_at,
                   ABS(EXTRACT(EPOCH FROM (oh.scraped_at - oa.scraped_at))) AS pair_lag_seconds,
                   ROW_NUMBER() OVER (
                       PARTITION BY g.game_pk, oh.book
                       ORDER BY GREATEST(oh.scraped_at, oa.scraped_at) DESC,
                                ABS(EXTRACT(EPOCH FROM (oh.scraped_at - oa.scraped_at)))
                   ) AS pair_rank,
                   CASE WHEN oh.moneyline < 0
                        THEN -oh.moneyline::double precision / (-oh.moneyline + 100.0)
                        ELSE 100.0 / (oh.moneyline + 100.0)
                   END AS home_implied,
                   CASE WHEN oa.moneyline < 0
                        THEN -oa.moneyline::double precision / (-oa.moneyline + 100.0)
                        ELSE 100.0 / (oa.moneyline + 100.0)
                   END AS away_implied
            FROM games g
            JOIN odds oh
              ON oh.game_pk = g.game_pk AND oh.team = g.home_team
            JOIN odds oa
              ON oa.game_pk = g.game_pk
             AND oa.team = g.away_team
             AND oa.book = oh.book
            WHERE g.game_date BETWEEN :start AND :end
              AND oh.moneyline IS NOT NULL
              AND oa.moneyline IS NOT NULL
              AND oh.scraped_at < g.start_time
              AND oa.scraped_at < g.start_time
              AND ABS(EXTRACT(EPOCH FROM (oh.scraped_at - oa.scraped_at))) <= 5
        ), market_consensus AS (
            SELECT game_pk,
                   AVG(home_implied / (home_implied + away_implied)) AS home_market_prob,
                   COUNT(*) AS paired_books,
                   MAX(pair_lag_seconds) AS max_pair_lag_seconds,
                   MAX(market_quote_at) AS market_quote_at
            FROM paired_candidates
            WHERE pair_rank = 1
            GROUP BY game_pk
        )
        SELECT g.game_pk, g.game_date, g.start_time,
               g.home_team, g.away_team, g.home_score, g.away_score,
               h.win_prob AS home_model_prob,
               h.prediction_updated_at AS home_prediction_at,
               a.prediction_updated_at AS away_prediction_at,
               h.expected_runs AS home_expected_runs,
               a.expected_runs AS away_expected_runs,
               h.moneyline AS home_moneyline,
               a.moneyline AS away_moneyline,
               market.home_market_prob,
               market.paired_books,
               market.max_pair_lag_seconds,
               market.market_quote_at,
               h.win_prob_p10 AS home_win_prob_p10,
               h.win_prob_p90 AS home_win_prob_p90,
               h.lineup_source,
               h.posterior_age_days,
               COALESCE((
                   SELECT SUM(b.reliever_outs)
                   FROM bullpen_daily b
                   WHERE b.team = g.home_team
                     AND b.game_date BETWEEN g.game_date - 2 AND g.game_date - 1
               ), 0) AS home_bp_outs_2d,
               COALESCE((
                   SELECT SUM(b.reliever_outs)
                   FROM bullpen_daily b
                   WHERE b.team = g.away_team
                     AND b.game_date BETWEEN g.game_date - 2 AND g.game_date - 1
               ), 0) AS away_bp_outs_2d
        FROM games g
        JOIN model_outputs_season h
          ON h.game_pk = g.game_pk AND h.team = g.home_team
        JOIN model_outputs_season a
          ON a.game_pk = g.game_pk AND a.team = g.away_team
        JOIN market_consensus market ON market.game_pk = g.game_pk
        WHERE g.game_date BETWEEN :start AND :end
          AND g.status = 'Final'
          AND g.home_score IS NOT NULL
          AND g.away_score IS NOT NULL
          AND h.win_prob IS NOT NULL
          AND h.expected_runs IS NOT NULL
          AND a.expected_runs IS NOT NULL
          AND h.moneyline IS NOT NULL
          AND a.moneyline IS NOT NULL
          AND h.date::date = g.game_date
          AND a.date::date = g.game_date
          AND h.prediction_updated_at < g.start_time
          AND a.prediction_updated_at < g.start_time
        ORDER BY g.game_date, g.start_time, g.game_pk
    """)
    with engine.begin() as conn:
        return pd.read_sql(query, conn, params={"start": start, "end": end})


def _shrunk_mean(values: deque[float]) -> float:
    return float(sum(values) / (len(values) + TEAM_PRIOR_GAMES))


def build_feature_frame(games: pd.DataFrame) -> pd.DataFrame:
    frame = games.sort_values(["game_date", "start_time", "game_pk"]).reset_index(drop=True).copy()
    if "home_market_prob" not in frame:
        raise ValueError("home_market_prob must come from paired same-book pregame odds")
    frame["market_logit"] = _logit(frame["home_market_prob"])
    frame["market_logit_correction"] = frame["market_logit"]
    frame["sim_disagreement"] = _logit(frame["home_model_prob"]) - frame["market_logit"]
    frame["home_win"] = (frame["home_score"] > frame["away_score"]).astype(int)
    frame["bullpen_rest_diff"] = frame["away_bp_outs_2d"] - frame["home_bp_outs_2d"]
    frame["uncertainty_width"] = frame["home_win_prob_p90"] - frame["home_win_prob_p10"]
    sources = frame["lineup_source"].fillna("")
    frame["lineup_live"] = sources.str.startswith("lineup_live").astype(float)
    frame["queue_live"] = sources.str.endswith("queue_live").astype(float)
    frame["posterior_age_days"] = frame["posterior_age_days"].fillna(0).astype(float)

    histories = defaultdict(lambda: {
        "offense": deque(maxlen=TEAM_WINDOW),
        "defense": deque(maxlen=TEAM_WINDOW),
        "margin": deque(maxlen=TEAM_WINDOW),
        "wins": deque(maxlen=TEAM_WINDOW),
    })
    offense_diffs = np.zeros(len(frame), dtype=float)
    defense_diffs = np.zeros(len(frame), dtype=float)
    margin_diffs = np.zeros(len(frame), dtype=float)
    win_diffs = np.zeros(len(frame), dtype=float)

    for _, day in frame.groupby("game_date", sort=True):
        for index, row in day.iterrows():
            home = histories[row.home_team]
            away = histories[row.away_team]
            offense_diffs[index] = _shrunk_mean(home["offense"]) - _shrunk_mean(away["offense"])
            defense_diffs[index] = _shrunk_mean(home["defense"]) - _shrunk_mean(away["defense"])
            margin_diffs[index] = _shrunk_mean(home["margin"]) - _shrunk_mean(away["margin"])
            win_diffs[index] = _shrunk_mean(home["wins"]) - _shrunk_mean(away["wins"])

        for _, row in day.iterrows():
            home = histories[row.home_team]
            away = histories[row.away_team]
            home["offense"].append(float(row.home_score - row.home_expected_runs))
            away["offense"].append(float(row.away_score - row.away_expected_runs))
            home["defense"].append(float(row.away_expected_runs - row.away_score))
            away["defense"].append(float(row.home_expected_runs - row.home_score))
            margin = float(row.home_score - row.away_score)
            home["margin"].append(margin)
            away["margin"].append(-margin)
            home_win = float(row.home_score > row.away_score)
            home["wins"].append(home_win - 0.5)
            away["wins"].append(0.5 - home_win)

    frame["offense_residual_diff"] = offense_diffs
    frame["defense_residual_diff"] = defense_diffs
    frame["run_margin_form_diff"] = margin_diffs
    frame["win_form_diff"] = win_diffs
    frame["uncertainty_width"] = frame["uncertainty_width"].fillna(0.0)
    return frame


@dataclass
class OffsetLogit:
    columns: list[str]
    mean: np.ndarray
    scale: np.ndarray
    intercept: float
    coefficients: np.ndarray

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.columns].to_numpy(dtype=float)
        standardized = (values - self.mean) / self.scale
        logits = frame["market_logit"].to_numpy(dtype=float)
        logits = logits + self.intercept + standardized @ self.coefficients
        return 1.0 / (1.0 + np.exp(-logits))


def fit_offset_logit(frame: pd.DataFrame, columns: list[str], l2: float = 1.0) -> OffsetLogit:
    values = frame[columns].to_numpy(dtype=float)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale == 0] = 1.0
    x = (values - mean) / scale
    offset = frame["market_logit"].to_numpy(dtype=float)
    actual = frame["home_win"].to_numpy(dtype=float)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = parameters[0]
        coefficients = parameters[1:]
        logits = offset + intercept + x @ coefficients
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        loss = np.logaddexp(0.0, logits).sum() - np.dot(actual, logits)
        loss += 0.5 * l2 * np.dot(coefficients, coefficients)
        error = probabilities - actual
        gradient = np.concatenate((
            np.array([error.sum()]),
            x.T @ error + l2 * coefficients,
        ))
        return float(loss), gradient

    result = minimize(
        objective,
        np.zeros(len(columns) + 1),
        jac=True,
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(f"offset logistic fit failed: {result.message}")
    return OffsetLogit(columns, mean, scale, float(result.x[0]), result.x[1:])


def _metrics(actual: np.ndarray, probability: np.ndarray) -> dict:
    return {
        "brier": round(float(brier_score_loss(actual, probability)), 6),
        "log_loss": round(float(log_loss(actual, probability, labels=[0, 1])), 6),
    }


def _metric_delta_intervals(
    frame: pd.DataFrame,
    model_probability: np.ndarray,
    market_probability: np.ndarray,
    n_boot: int = 2000,
) -> dict:
    actual = frame["home_win"].to_numpy(dtype=float)
    model = np.clip(model_probability, 1e-7, 1 - 1e-7)
    market = np.clip(market_probability, 1e-7, 1 - 1e-7)
    deltas = pd.DataFrame({
        "game_date": frame["game_date"].to_numpy(),
        "brier": (model - actual) ** 2 - (market - actual) ** 2,
        "log_loss": (
            -(actual * np.log(model) + (1 - actual) * np.log(1 - model))
            + actual * np.log(market)
            + (1 - actual) * np.log(1 - market)
        ),
    })
    daily = deltas.groupby("game_date")[["brier", "log_loss"]].mean().to_numpy()
    rng = np.random.default_rng(20260722)
    samples = daily[rng.integers(0, len(daily), size=(n_boot, len(daily)))].mean(axis=1)
    return {
        "brier": [round(float(value), 6) for value in np.quantile(samples[:, 0], [0.025, 0.975])],
        "log_loss": [round(float(value), 6) for value in np.quantile(samples[:, 1], [0.025, 0.975])],
    }


def _ledger(frame: pd.DataFrame, home_probability: np.ndarray, threshold: float) -> dict:
    home_raw = american_to_prob(frame["home_moneyline"])
    away_raw = american_to_prob(frame["away_moneyline"])
    bets = []
    for index, row in frame.reset_index(drop=True).iterrows():
        candidates = (
            (home_probability[index], home_raw[index], row.home_moneyline, row.home_win, row.home_team),
            (1.0 - home_probability[index], away_raw[index], row.away_moneyline, 1 - row.home_win, row.away_team),
        )
        for probability, book_probability, odds, won, team in candidates:
            edge = float(probability - book_probability)
            if edge < threshold:
                continue
            profit = odds / 100.0 if won and odds > 0 else 100.0 / -odds if won else -1.0
            bets.append((edge, float(odds), int(won), float(profit), team, row.game_date))

    if not bets:
        return {
            "n_bets": 0,
            "units": 0.0,
            "roi": None,
            "roi_ci_95": None,
            "max_team_share": None,
            "edge_buckets": [],
        }
    ledger = pd.DataFrame(
        bets,
        columns=["edge", "odds", "won", "profit", "team", "game_date"],
    )
    daily = [group["profit"].to_numpy() for _, group in ledger.groupby("game_date")]
    rng = np.random.default_rng(20260722)
    roi_samples = []
    for _ in range(2000):
        sampled = [daily[index] for index in rng.integers(0, len(daily), size=len(daily))]
        roi_samples.append(float(np.concatenate(sampled).mean()))
    labels = ["4.5-6%", "6-8%", "8-10%", "10%+"]
    ledger["bucket"] = pd.cut(
        ledger["edge"],
        bins=[threshold, 0.06, 0.08, 0.10, np.inf],
        labels=labels,
        include_lowest=True,
        duplicates="drop",
    )
    buckets = []
    for label, group in ledger.groupby("bucket", observed=True):
        buckets.append({
            "bucket": str(label),
            "n_bets": int(len(group)),
            "roi": round(float(group["profit"].mean()), 6),
        })
    return {
        "n_bets": int(len(ledger)),
        "units": round(float(ledger["profit"].sum()), 4),
        "roi": round(float(ledger["profit"].mean()), 6),
        "roi_ci_95": [
            round(float(value), 6)
            for value in np.quantile(roi_samples, [0.025, 0.975])
        ],
        "max_team_share": round(float(ledger["team"].value_counts(normalize=True).max()), 6),
        "edge_buckets": buckets,
    }


def _edge_monotonic(ledger: dict) -> bool:
    supported = [bucket for bucket in ledger["edge_buckets"] if bucket["n_bets"] >= 10]
    if len(supported) < 2:
        return False
    rois = [bucket["roi"] for bucket in supported]
    return all(right >= left for left, right in zip(rois, rois[1:])) and rois[-1] > 0


def evaluate_feature_blocks(
    games: pd.DataFrame,
    threshold: float = EV_THRESHOLDS["ml"],
    n_folds: int = 4,
) -> dict:
    frame = build_feature_frame(games)
    if len(frame) < 100:
        raise ValueError("at least 100 completed games are required")

    cut_points = np.linspace(0.5, 1.0, n_folds + 1)
    tests = []
    predictions = {name: [] for name in FEATURE_BLOCKS}
    for fold in range(n_folds):
        train_end = int(len(frame) * cut_points[fold])
        test_end = int(len(frame) * cut_points[fold + 1])
        train = frame.iloc[:train_end]
        test = frame.iloc[train_end:test_end]
        tests.append(test)
        for name, columns in FEATURE_BLOCKS.items():
            model = fit_offset_logit(train, columns)
            predictions[name].append(model.predict(test))

    test_frame = pd.concat(tests, ignore_index=True)
    actual = test_frame["home_win"].to_numpy(dtype=int)
    market_probability = test_frame["home_market_prob"].to_numpy(dtype=float)
    market_metrics = _metrics(actual, market_probability)
    results = {}
    for name, columns in FEATURE_BLOCKS.items():
        probability = np.concatenate(predictions[name])
        metrics = _metrics(actual, probability)
        metric_delta_ci = _metric_delta_intervals(
            test_frame,
            probability,
            market_probability,
        )
        ledger = _ledger(test_frame, probability, threshold)
        fold_metrics = []
        for test, fold_probability in zip(tests, predictions[name]):
            fold_actual = test["home_win"].to_numpy(dtype=int)
            fold_metrics.append({
                "games": int(len(test)),
                "market": _metrics(
                    fold_actual,
                    test["home_market_prob"].to_numpy(dtype=float),
                ),
                "model": _metrics(fold_actual, fold_probability),
            })
        full_model = fit_offset_logit(frame, columns)
        coefficients = {
            column: round(float(value), 6)
            for column, value in zip(columns, full_model.coefficients)
        }
        gates = {
            "beats_market_brier": metrics["brier"] < market_metrics["brier"],
            "beats_market_log_loss": metrics["log_loss"] < market_metrics["log_loss"],
            "credible_brier_improvement": metric_delta_ci["brier"][1] < 0,
            "credible_log_loss_improvement": metric_delta_ci["log_loss"][1] < 0,
            "enough_flagged_bets": ledger["n_bets"] >= MIN_EDGE_BETS,
            "positive_flagged_roi": ledger["roi"] is not None and ledger["roi"] > 0,
            "credible_positive_roi": (
                ledger["roi_ci_95"] is not None and ledger["roi_ci_95"][0] > 0
            ),
            "edge_monotonic": _edge_monotonic(ledger),
            "not_team_concentrated": (
                ledger["max_team_share"] is not None and ledger["max_team_share"] <= 0.20
            ),
        }
        gates["all_pass"] = all(gates.values())
        results[name] = {
            "metrics": metrics,
            "metric_delta_ci_95": metric_delta_ci,
            "fold_metrics": fold_metrics,
            "ledger": ledger,
            "coefficients_standardized": coefficients,
            "intercept": round(full_model.intercept, 6),
            "gates": gates,
        }

    return {
        "games": int(len(frame)),
        "rolling_test_games": int(len(test_frame)),
        "market_metrics": market_metrics,
        "feature_blocks": results,
        "passing_blocks": [name for name, result in results.items() if result["gates"]["all_pass"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--threshold", type=float, default=EV_THRESHOLDS["ml"])
    args = parser.parse_args()

    report = evaluate_feature_blocks(
        load_feature_games(args.start, args.end),
        threshold=args.threshold,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["passing_blocks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
