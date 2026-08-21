"""Measure whether simulator probabilities add signal beyond the market."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sqlalchemy import text

from backend.db import engine
from backend.strategy import EV_THRESHOLDS


MIN_EDGE_BETS = 50


def american_to_prob(odds: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(odds, dtype=float)
    probabilities = np.empty_like(values)
    negative = values < 0
    probabilities[negative] = -values[negative] / (-values[negative] + 100.0)
    probabilities[~negative] = 100.0 / (values[~negative] + 100.0)
    return probabilities


def _logit(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_games(start: str, end: str) -> pd.DataFrame:
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
               h.win_prob AS home_model_prob,
               h.prediction_updated_at AS home_prediction_at,
               a.prediction_updated_at AS away_prediction_at,
               h.moneyline AS home_moneyline,
               a.moneyline AS away_moneyline,
               market.home_market_prob,
               market.paired_books,
               market.max_pair_lag_seconds,
               market.market_quote_at,
               CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END AS home_win
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
          AND h.moneyline IS NOT NULL
          AND a.moneyline IS NOT NULL
          AND h.date::date = g.game_date
          AND a.date::date = g.game_date
          AND h.prediction_updated_at < g.start_time
          AND a.prediction_updated_at < g.start_time
        ORDER BY g.game_date, g.game_pk
    """)
    with engine.begin() as conn:
        return pd.read_sql(query, conn, params={"start": start, "end": end})


def load_runline_games(start: str, end: str) -> pd.DataFrame:
    query = text("""
        WITH paired_candidates AS (
            SELECT g.game_pk, oh.book, oh.spread AS home_spread,
                   GREATEST(oh.scraped_at, oa.scraped_at) AS market_quote_at,
                   ABS(EXTRACT(EPOCH FROM (oh.scraped_at - oa.scraped_at))) AS pair_lag_seconds,
                   ROW_NUMBER() OVER (
                       PARTITION BY g.game_pk, oh.book, oh.spread
                       ORDER BY GREATEST(oh.scraped_at, oa.scraped_at) DESC,
                                ABS(EXTRACT(EPOCH FROM (oh.scraped_at - oa.scraped_at)))
                   ) AS pair_rank,
                   CASE WHEN oh.spread_odds < 0
                        THEN -oh.spread_odds::double precision / (-oh.spread_odds + 100.0)
                        ELSE 100.0 / (oh.spread_odds + 100.0)
                   END AS home_implied,
                   CASE WHEN oa.spread_odds < 0
                        THEN -oa.spread_odds::double precision / (-oa.spread_odds + 100.0)
                        ELSE 100.0 / (oa.spread_odds + 100.0)
                   END AS away_implied
            FROM games g
            JOIN odds oh
              ON oh.game_pk = g.game_pk AND oh.team = g.home_team
            JOIN odds oa
              ON oa.game_pk = g.game_pk
             AND oa.team = g.away_team
             AND oa.book = oh.book
            WHERE g.game_date BETWEEN :start AND :end
              AND oh.spread_odds IS NOT NULL
              AND oa.spread_odds IS NOT NULL
              AND ABS(oh.spread) = 1.5
              AND ABS(oa.spread) = 1.5
              AND ABS(oh.spread + oa.spread) < 0.001
              AND oh.scraped_at < g.start_time
              AND oa.scraped_at < g.start_time
              AND ABS(EXTRACT(EPOCH FROM (oh.scraped_at - oa.scraped_at))) <= 5
        ), market_consensus AS (
            SELECT game_pk, home_spread,
                   AVG(home_implied / (home_implied + away_implied)) AS home_market_prob,
                   COUNT(*) AS paired_books,
                   MAX(pair_lag_seconds) AS max_pair_lag_seconds,
                   MAX(market_quote_at) AS market_quote_at
            FROM paired_candidates
            WHERE pair_rank = 1
            GROUP BY game_pk, home_spread
        )
        SELECT g.game_pk, g.game_date, g.start_time,
               h.p_cover AS home_model_prob,
               h.prediction_updated_at AS home_prediction_at,
               a.prediction_updated_at AS away_prediction_at,
               h.spread_odds AS home_moneyline,
               a.spread_odds AS away_moneyline,
               market.home_market_prob,
               market.paired_books,
               market.max_pair_lag_seconds,
               market.market_quote_at,
               CASE WHEN g.home_score - g.away_score + h.spread > 0
                    THEN 1 ELSE 0 END AS home_win
        FROM games g
        JOIN model_outputs_season h
          ON h.game_pk = g.game_pk AND h.team = g.home_team
        JOIN model_outputs_season a
          ON a.game_pk = g.game_pk AND a.team = g.away_team
        JOIN market_consensus market
          ON market.game_pk = g.game_pk
         AND ABS(market.home_spread - h.spread) < 0.001
        WHERE g.game_date BETWEEN :start AND :end
          AND g.status = 'Final'
          AND g.home_score IS NOT NULL
          AND g.away_score IS NOT NULL
          AND h.p_cover IS NOT NULL
          AND h.spread_odds IS NOT NULL
          AND a.spread_odds IS NOT NULL
          AND ABS(h.spread) = 1.5
          AND ABS(a.spread) = 1.5
          AND ABS(h.spread + a.spread) < 0.001
          AND h.date::date = g.game_date
          AND a.date::date = g.game_date
          AND h.prediction_updated_at < g.start_time
          AND a.prediction_updated_at < g.start_time
        ORDER BY g.game_date, g.game_pk
    """)
    with engine.begin() as conn:
        return pd.read_sql(query, conn, params={"start": start, "end": end})


def prepare_games(games: pd.DataFrame) -> pd.DataFrame:
    frame = games.sort_values(["game_date", "game_pk"]).reset_index(drop=True).copy()
    if "home_market_prob" not in frame:
        raise ValueError("home_market_prob must come from paired same-book pregame odds")
    frame["model_logit"] = _logit(frame["home_model_prob"].to_numpy())
    frame["market_logit"] = _logit(frame["home_market_prob"].to_numpy())
    return frame


def _fit(train: pd.DataFrame, columns: list[str]) -> LogisticRegression:
    model = LogisticRegression(C=10.0, solver="lbfgs")
    model.fit(train[columns], train["home_win"])
    return model


def _probability_metrics(actual: np.ndarray, probability: np.ndarray) -> dict:
    return {
        "brier": round(float(brier_score_loss(actual, probability)), 6),
        "log_loss": round(float(log_loss(actual, probability, labels=[0, 1])), 6),
    }


def _flat_bet_ledger(games: pd.DataFrame, home_prob: np.ndarray, threshold: float) -> dict:
    home_raw = american_to_prob(games["home_moneyline"])
    away_raw = american_to_prob(games["away_moneyline"])
    records = []
    for index, row in games.reset_index(drop=True).iterrows():
        candidates = (
            (float(home_prob[index]), home_raw[index], float(row.home_moneyline), int(row.home_win)),
            (1.0 - float(home_prob[index]), away_raw[index], float(row.away_moneyline), 1 - int(row.home_win)),
        )
        for probability, book_probability, odds, won in candidates:
            if probability - book_probability < threshold:
                continue
            profit = odds / 100.0 if won and odds > 0 else 100.0 / -odds if won else -1.0
            records.append((odds, won, profit))
    if not records:
        return {"n_bets": 0, "n_underdogs": 0, "units": 0.0, "roi": None}
    ledger = pd.DataFrame(records, columns=["odds", "won", "profit"])
    return {
        "n_bets": int(len(ledger)),
        "n_underdogs": int((ledger["odds"] >= 100).sum()),
        "units": round(float(ledger["profit"].sum()), 4),
        "roi": round(float(ledger["profit"].mean()), 6),
    }


def evaluate_market_residual(
    games: pd.DataFrame,
    n_folds: int = 4,
    threshold: float = EV_THRESHOLDS["ml"],
) -> dict:
    frame = prepare_games(games)
    if len(frame) < 100:
        raise ValueError("at least 100 completed games are required")

    cut_points = np.linspace(0.5, 1.0, n_folds + 1)
    tests = []
    stack_probabilities = []
    for fold in range(n_folds):
        train_end = int(len(frame) * cut_points[fold])
        test_end = int(len(frame) * cut_points[fold + 1])
        train = frame.iloc[:train_end]
        test = frame.iloc[train_end:test_end]
        stack = _fit(train, ["model_logit", "market_logit"])
        tests.append(test)
        stack_probabilities.append(stack.predict_proba(test[["model_logit", "market_logit"]])[:, 1])

    test_frame = pd.concat(tests, ignore_index=True)
    stack_probability = np.concatenate(stack_probabilities)
    actual = test_frame["home_win"].to_numpy(dtype=int)
    model_probability = test_frame["home_model_prob"].to_numpy(dtype=float)
    market_probability = test_frame["home_market_prob"].to_numpy(dtype=float)

    full_stack = _fit(frame, ["model_logit", "market_logit"])
    metrics = {
        "simulator": _probability_metrics(actual, model_probability),
        "market": _probability_metrics(actual, market_probability),
        "market_plus_simulator": _probability_metrics(actual, stack_probability),
    }
    ledgers = {
        "simulator": _flat_bet_ledger(test_frame, model_probability, threshold),
        "market_plus_simulator": _flat_bet_ledger(test_frame, stack_probability, threshold),
    }
    simulator_coefficient = float(full_stack.coef_[0][0])
    stack_ledger = ledgers["market_plus_simulator"]
    gates = {
        "beats_market_brier": metrics["market_plus_simulator"]["brier"] < metrics["market"]["brier"],
        "beats_market_log_loss": metrics["market_plus_simulator"]["log_loss"] < metrics["market"]["log_loss"],
        "positive_simulator_coefficient": simulator_coefficient > 0,
        "enough_flagged_bets": stack_ledger["n_bets"] >= MIN_EDGE_BETS,
        "positive_flagged_roi": stack_ledger["roi"] is not None and stack_ledger["roi"] > 0,
    }
    gates["all_pass"] = all(gates.values())

    return {
        "games": int(len(frame)),
        "rolling_test_games": int(len(test_frame)),
        "stack_coefficients": {
            "intercept": round(float(full_stack.intercept_[0]), 6),
            "simulator_logit": round(simulator_coefficient, 6),
            "market_logit": round(float(full_stack.coef_[0][1]), 6),
        },
        "metrics": metrics,
        "ledgers": ledgers,
        "gates": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--market", choices=("ml", "rl"), default="ml")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    loader = load_games if args.market == "ml" else load_runline_games
    threshold = args.threshold if args.threshold is not None else EV_THRESHOLDS[args.market]
    report = evaluate_market_residual(loader(args.start, args.end), threshold=threshold)
    report["market"] = args.market
    report["threshold"] = threshold
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n")
    return 0 if report["gates"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
