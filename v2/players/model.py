"""The fitted valuation model: every weight, curve, and constant scoring needs.

Fitted once per offseason, stored as one JSON row in ``mlb.player_value_models``
and optionally on disk. Scoring refuses a model fitted on the season it scores.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from backend.db import engine
from v2.players.pricing import (
    FALLBACK_ARB_FRACTIONS,
    FALLBACK_INFLATION,
    FALLBACK_PRICE_MUSD,
    FALLBACK_PRICE_SOURCE,
    PriceModel,
)
from v2.players.runs import PUBLISHED_POSITIONAL_RUNS

log = logging.getLogger(__name__)

MODEL_VERSION = "player-value-v1"
MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_WINDOW = {0: 5.0, 1: 4.0, 2: 3.0}
DEFAULT_SHRINK = 100.0
DEFAULT_RUNS_PER_WIN = 10.0
DEFAULT_PA_PER_IP = 4.37
DEFAULT_LINEAR_WEIGHTS = {"K": -0.23, "BB": 0.23, "HBP": 0.36, "1B": 0.44, "2B": 0.72, "3B": 0.97, "HR": 1.52, "OUT": -0.25}


@dataclass
class ValueModel:
    model_version: str = MODEL_VERSION
    fitted_at: str = ""
    dev_seasons: list[int] = field(default_factory=list)
    holdout_pair: tuple[int, int] = (0, 0)
    window_weights: dict[int, float] = field(default_factory=lambda: dict(DEFAULT_WINDOW))  # seasons back -> weight
    shrink_strength: float = DEFAULT_SHRINK
    # One row per (pool, trait, input): weight within the trait, status.
    trait_weights: list[dict] = field(default_factory=list)
    # One row per (pool, feature): coefficient on the z-scored feature, plus the intercept rows.
    grade_weights: list[dict] = field(default_factory=list)
    # trait or grade name -> k in r(n) = n / (n + k).
    reliability_k: dict[str, float] = field(default_factory=dict)
    age_curves: dict[str, dict[int, float]] = field(default_factory=dict)
    positional_adjustment: dict[str, float] = field(default_factory=lambda: dict(PUBLISHED_POSITIONAL_RUNS))
    positional_seasons: int = 0
    replacement: dict[str, float] = field(default_factory=dict)
    linear_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LINEAR_WEIGHTS))
    pa_per_ip: float = DEFAULT_PA_PER_IP
    runs_per_win: float = DEFAULT_RUNS_PER_WIN
    price: dict = field(default_factory=lambda: asdict(PriceModel(FALLBACK_PRICE_MUSD, float("nan"), FALLBACK_INFLATION, 0, FALLBACK_PRICE_SOURCE)))
    arb_fractions: dict[int, float] = field(default_factory=lambda: dict(FALLBACK_ARB_FRACTIONS))
    evaluation: dict = field(default_factory=dict)

    @classmethod
    def empty(cls, dev_seasons: list[int], holdout_pair: tuple[int, int]) -> ValueModel:
        """Fallback constants and no fitted weights, for tests and dry runs."""
        return cls(dev_seasons=list(dev_seasons), holdout_pair=tuple(holdout_pair), fitted_at=datetime.now(UTC).isoformat())

    def validate(self, season: int) -> None:
        """Refuse to score a season the model was developed on."""
        if season in self.dev_seasons:
            raise ValueError(f"model trained on {season}; refusing to score it")

    def price_model(self) -> PriceModel:
        p = self.price
        return PriceModel(float(p["price_musd"]), float(p["se"]) if p.get("se") == p.get("se") else float("nan"),
                          float(p["inflation"]), int(p["n"]), str(p["source"]))

    def trait_weight_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.trait_weights, columns=["pool", "trait", "input", "weight", "status"])

    def grade_weight_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.grade_weights, columns=["pool", "feature", "coefficient"])

    def to_payload(self) -> dict:
        payload = asdict(self)
        payload["holdout_pair"] = list(self.holdout_pair)
        payload["window_weights"] = {str(k): v for k, v in self.window_weights.items()}
        payload["age_curves"] = {c: {str(a): v for a, v in curve.items()} for c, curve in self.age_curves.items()}
        payload["arb_fractions"] = {str(k): v for k, v in self.arb_fractions.items()}
        return payload

    @classmethod
    def from_payload(cls, payload: dict) -> ValueModel:
        p = dict(payload)
        p["holdout_pair"] = tuple(p.get("holdout_pair", (0, 0)))
        p["window_weights"] = {int(k): float(v) for k, v in p.get("window_weights", {}).items()}
        p["age_curves"] = {c: {int(a): float(v) for a, v in curve.items()} for c, curve in p.get("age_curves", {}).items()}
        p["arb_fractions"] = {int(k): float(v) for k, v in p.get("arb_fractions", {}).items()}
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in p.items() if k in known})

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_payload(), indent=1, allow_nan=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> ValueModel:
        return cls.from_payload(json.loads(path.read_text(encoding="utf-8")))


def model_path(last_dev_season: int) -> Path:
    return MODELS_DIR / f"player_value_model_{last_dev_season}.json"


def publish_model(model: ValueModel) -> None:
    """Upsert the model row; scoring in CI reads the newest ``fitted_at``."""
    payload = json.dumps(model.to_payload(), allow_nan=False)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO player_value_models (model_version, last_dev_season, payload, fitted_at, updated_at) "
            "VALUES (:v, :s, CAST(:p AS jsonb), :f, now()) "
            "ON CONFLICT (model_version, last_dev_season) DO UPDATE SET payload = EXCLUDED.payload, "
            "fitted_at = EXCLUDED.fitted_at, updated_at = now()"
        ), {"v": model.model_version, "s": max(model.dev_seasons), "p": payload, "f": model.fitted_at})
    log.info(f"model: published {model.model_version} through {max(model.dev_seasons)}")


def load_latest_from_db() -> ValueModel:
    with engine.connect() as conn:
        row = conn.execute(text("SELECT payload FROM player_value_models ORDER BY fitted_at DESC LIMIT 1")).fetchone()
    if row is None:
        raise RuntimeError("No player value model published; run python -m v2.players.fit first")
    payload = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    return ValueModel.from_payload(payload)
