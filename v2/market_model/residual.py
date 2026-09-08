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


def load_frozen_games(start: str, end: str) -> pd.DataFrame:
    """Read frozen forecasts, including provenance, without mutable odds joins.

    to_jsonb keeps the read safe before the additive schema migration: legacy
    rows have no context and are excluded, never mislabeled as raw forecasts.
    """
    query = text("""
        SELECT g.game_pk, g.game_date, g.start_time, g.home_team, g.away_team,
               g.home_score, g.away_score,
               h.win_prob AS home_published_prob,
               h.expected_runs AS home_expected_runs, a.expected_runs AS away_expected_runs,
               h.win_prob_p10 AS home_win_prob_p10, h.win_prob_p90 AS home_win_prob_p90,
               h.lineup_source, h.posterior_age_days,
               h.prediction_updated_at AS home_prediction_at,
               a.prediction_updated_at AS away_prediction_at,
               h.moneyline AS home_moneyline, a.moneyline AS away_moneyline,
               h.spread AS home_spread, a.spread AS away_spread,
               h.spread_odds AS home_spread_odds, a.spread_odds AS away_spread_odds,
               h.p_cover AS home_cover_prob,
               h.runs_hist AS home_runs_hist, a.runs_hist AS away_runs_hist,
               to_jsonb(h)->'prediction_context' AS prediction_context,
               to_jsonb(a)->'prediction_context' AS away_prediction_context
        FROM games g
        JOIN model_outputs_season h ON h.game_pk=g.game_pk AND h.team=g.home_team
        JOIN model_outputs_season a ON a.game_pk=g.game_pk AND a.team=g.away_team
        WHERE g.game_date BETWEEN :start AND :end AND g.status='Final'
          AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
          AND h.date::date=g.game_date AND a.date::date=g.game_date
          AND h.prediction_updated_at < g.start_time
          AND a.prediction_updated_at < g.start_time
        ORDER BY g.game_date,g.start_time,g.game_pk
    """)
    with engine.begin() as conn:
        games = pd.read_sql(query, conn, params={"start": start, "end": end})
    return validated_forecasts(games)


def validated_forecasts(games: pd.DataFrame) -> pd.DataFrame:
    """Only genuine pregame forecasts with matching raw/model/input provenance."""
    records = []
    for row in games.to_dict("records"):
        context = row.get("prediction_context")
        if not isinstance(context, dict) or context != row.get("away_prediction_context"):
            continue
        try:
            start = pd.to_datetime(row["start_time"], utc=True)
            day = pd.Timestamp(row["game_date"]).date()
            dates = [context["forecast_at"], context["inputs_as_of"],
                     row["home_prediction_at"], row["away_prediction_at"]]
            if pd.isna(start) or any(pd.isna(pd.to_datetime(d, utc=True)) or pd.to_datetime(d, utc=True) >= start for d in dates):
                continue
            cutoffs = [pd.Timestamp(context[key]) for key in ("training_max_date", "tables_training_max_date")]
            if pd.isna(day) or any(pd.isna(cutoff) or cutoff.date() >= day for cutoff in cutoffs):
                continue
            raw = float(context["raw_home_win_prob"])
            if not np.isfinite(raw) or not 0 <= raw <= 1 or not context.get("model_version"):
                continue
            row.update(home_model_prob=raw, probability_source="raw_simulator",
                       model_version=context["model_version"],
                       home_win=int(row["home_score"] > row["away_score"]))
        except (KeyError, TypeError, ValueError):
            continue
        records.append(row)
    return pd.DataFrame(records, columns=list(dict.fromkeys([*games.columns,
        "home_model_prob", "probability_source", "model_version", "home_win",
    ])))


def _snapshot_market(row: dict, market: str) -> dict | None:
    context = row["prediction_context"]
    pairs = []
    forecast = pd.to_datetime(context["forecast_at"], utc=True)
    for pair in context.get("market_pairs", []):
        try:
            htime = pd.to_datetime(pair["home_quoted_at"], utc=True)
            atime = pd.to_datetime(pair["away_quoted_at"], utc=True)
            if not pair.get("book") or pd.isna(htime) or pd.isna(atime):
                continue
            if max(htime, atime) > forecast or abs((htime - atime).total_seconds()) > 5:
                continue
            if market == "rl":
                spread = float(row["home_spread"])
                if abs(spread) != 1.5 or float(row["away_spread"]) != -spread:
                    continue
                if float(pair["home_spread"]) != spread or float(pair["away_spread"]) != -spread:
                    continue
                prices = [pair["home_spread_odds"], pair["away_spread_odds"]]
            else:
                prices = [pair["home_moneyline"], pair["away_moneyline"]]
            if not all(np.isfinite(float(p)) and abs(float(p)) >= 100 for p in prices):
                continue
            implied = american_to_prob(prices)
            pairs.append((implied[0] / implied.sum(), pair["book"], max(htime, atime), abs((htime-atime).total_seconds())))
        except (KeyError, TypeError, ValueError):
            continue
    if not pairs:
        return None
    return {"home_market_prob": float(np.mean([p[0] for p in pairs])),
            "paired_books": len(pairs), "market_quote_at": max(p[2] for p in pairs),
            "max_pair_lag_seconds": max(p[3] for p in pairs)}


def _load_market_games(start: str, end: str, market: str) -> pd.DataFrame:
    games = load_frozen_games(start, end)
    records = []
    for row in games.to_dict("records"):
        snapshot = _snapshot_market(row, market)
        if snapshot is None:
            continue
        if market == "rl":
            row.update(home_model_prob=row["home_cover_prob"],
                       home_moneyline=row["home_spread_odds"], away_moneyline=row["away_spread_odds"],
                       home_win=int(row["home_score"]-row["away_score"]+row["home_spread"] > 0))
        if not all(pd.notna(row[k]) and np.isfinite(float(row[k])) for k in ("home_model_prob", "home_moneyline", "away_moneyline")):
            continue
        row.update(snapshot)
        records.append(row)
    return pd.DataFrame(records, columns=list(games.columns) + [
        "home_market_prob", "paired_books", "market_quote_at", "max_pair_lag_seconds",
    ])


def load_games(start: str, end: str) -> pd.DataFrame:
    return _load_market_games(start, end, "ml")


def load_runline_games(start: str, end: str) -> pd.DataFrame:
    return _load_market_games(start, end, "rl")


def chronological_folds(frame: pd.DataFrame, n_folds: int):
    """Keep whole game dates together so no fold trains on same-day outcomes."""
    dates = np.array(sorted(frame["game_date"].unique()))
    cuts = (np.linspace(.5, 1, n_folds + 1) * len(dates)).astype(int)
    if len(set(cuts)) != len(cuts) or cuts[0] < 1:
        raise ValueError("Not enough distinct dates for chronological folds")
    for lo, hi in zip(cuts, cuts[1:]):
        train = frame[frame.game_date.isin(dates[:lo])]
        test = frame[frame.game_date.isin(dates[lo:hi])]
        yield train, test


def prepare_games(games: pd.DataFrame) -> pd.DataFrame:
    frame = games.sort_values(["game_date", "game_pk"]).reset_index(drop=True).copy()
    if "home_market_prob" not in frame:
        raise ValueError("home_market_prob must come from paired same-book pregame odds")
    if "probability_source" not in frame or not frame["probability_source"].eq("raw_simulator").all():
        raise ValueError("Research requires explicitly identified raw simulator probabilities")
    if "model_version" not in frame or frame["model_version"].isna().any() or frame["model_version"].nunique() != 1:
        raise ValueError("Evaluate one identified model version at a time")
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
    if len(games) < 100:
        raise ValueError("At least 100 frozen pregame forecasts with raw probability provenance are required")
    frame = prepare_games(games)
    if len(frame) < 100:
        raise ValueError("at least 100 completed games are required")

    tests = []
    stack_probabilities = []
    for train, test in chronological_folds(frame, n_folds):
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
