"""The fitted prospect model: boosters, baselines, scale references and the evaluation, as one JSON payload.

Stored as one jsonb row in ``mlb.prospect_models`` (boosters as gzipped base64 UBJSON) and optionally
on disk. Scoring in CI reads the newest ``fitted_at`` row and never fits.
"""
from __future__ import annotations

import base64
import gzip
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import text

from backend.db import engine

log = logging.getLogger(__name__)

MODEL_VERSION = "prospect-v1"
STUFF_VERSION = "prospect-stuff-v1"  # the stuff model's sibling row in prospect_models
MODELS_DIR = Path(__file__).resolve().parent / "models"
POOLS = ("bat", "pit")
QUANTILES = (0.25, 0.75)
RISK_LABELS = ("safe", "medium", "high", "extreme")


@dataclass
class ProspectModel:
    model_version: str = MODEL_VERSION
    fitted_at: str = ""
    dev_seasons: list[int] = field(default_factory=list)
    holdout_seasons: list[int] = field(default_factory=list)
    last_season: int = 0
    feature_names: dict[str, list[str]] = field(default_factory=dict)  # pool -> features
    boosters: dict[str, dict[str, str]] = field(default_factory=dict)  # pool -> target -> base64 ubj
    rounds: dict[str, dict[str, int]] = field(default_factory=dict)
    baseline: dict[str, dict] = field(default_factory=dict)  # pool -> KATOH-style coefficients
    grade_reference: dict[str, dict[str, float]] = field(default_factory=dict)  # pool -> mean, sd of MLB projected WAR
    risk_cutpoints: dict[str, list[float]] = field(default_factory=dict)  # pool -> quartile cut points on 1 - p_mlb
    eta_band: dict[str, int] = field(default_factory=dict)  # pool -> holdout mean absolute error of the arrival year, rounded
    regular_platt: dict[str, list[float]] = field(default_factory=dict)  # pool -> [slope, intercept] on the logit of P(regular)
    validated: dict[str, bool] = field(default_factory=dict)  # pool -> passed the publish gate
    price: dict = field(default_factory=dict)  # copied from the player model at fit time
    arb_fractions: dict[int, float] = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)

    def booster(self, pool: str, target: str) -> xgb.Booster:
        b = xgb.Booster()
        b.load_model(bytearray(gzip.decompress(base64.b64decode(self.boosters[pool][target]))))
        return b

    def set_booster(self, pool: str, target: str, booster: xgb.Booster) -> None:
        self.boosters.setdefault(pool, {})[target] = base64.b64encode(gzip.compress(bytes(booster.save_raw("ubj")), 9)).decode("ascii")

    def predict(self, frame: pd.DataFrame, pool: str, *, contributions: bool = False) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """Predictions for one pool's rows: probabilities, expected WARs, control quantiles, ETA years.

        With ``contributions`` the per-feature SHAP contributions of the peak WAR booster come back too.
        """
        names = self.feature_names[pool]
        X = frame.reindex(columns=names).astype(float)
        dm = xgb.DMatrix(X, feature_names=names, missing=np.nan)
        out = pd.DataFrame(index=frame.index)
        out["p_mlb"] = self.booster(pool, "mlb").predict(dm)
        out["p_regular"] = calibrate(self.booster(pool, "regular").predict(dm), self.regular_platt.get(pool))
        out["peak_war"] = np.clip(self.booster(pool, "peak_war").predict(dm), 0.0, None)
        out["control_war"] = np.clip(self.booster(pool, "control_war").predict(dm), 0.0, None)
        q = self.booster(pool, "control_q").predict(dm)
        q = q.reshape(len(frame), -1) if q.ndim == 1 and len(QUANTILES) > 1 else q
        out["control_q25"] = np.clip(q[:, 0], 0.0, None)
        out["control_q75"] = np.clip(np.maximum(q[:, -1], q[:, 0]), 0.0, None)
        out["eta_years"] = np.clip(self.booster(pool, "eta_years").predict(dm), 0.5, None)
        contribs: dict[str, pd.DataFrame] = {}
        if contributions:
            c = self.booster(pool, "peak_war").predict(dm, pred_contribs=True)
            contribs["peak_war"] = pd.DataFrame(c[:, :-1], index=frame.index, columns=names)
        return out, contribs

    def to_grade(self, peak_war: pd.Series, pool: str) -> pd.Series:
        """Peak WAR on the 20-80 scale of the published MLB projected WAR for the pool."""
        ref = self.grade_reference[pool]
        return (50 + 10 * (peak_war - ref["mean"]) / ref["sd"]).clip(20, 80)

    def risk(self, pred: pd.DataFrame, pool: str) -> pd.Series:
        """Risk band from the chance of never reaching the majors, cut at the holdout quartiles."""
        idx = np.searchsorted(np.array(self.risk_cutpoints[pool]), risk_score(pred).to_numpy(float), side="right")
        return pd.Series([RISK_LABELS[min(i, 3)] for i in idx], index=pred.index)

    def to_payload(self) -> dict:
        payload = asdict(self)
        payload["arb_fractions"] = {str(k): v for k, v in self.arb_fractions.items()}
        return payload

    @classmethod
    def from_payload(cls, payload: dict) -> ProspectModel:
        p = dict(payload)
        p["arb_fractions"] = {int(k): float(v) for k, v in p.get("arb_fractions", {}).items()}
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in p.items() if k in known})

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_payload(), allow_nan=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> ProspectModel:
        return cls.from_payload(json.loads(path.read_text(encoding="utf-8")))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibrate(p: np.ndarray, platt: list[float] | None) -> np.ndarray:
    """Platt-scaled probabilities; a model fitted without calibration passes through unchanged."""
    if platt is None:
        return p
    slope, intercept = platt
    return 1 / (1 + np.exp(-(slope * logit(p) + intercept)))


def risk_score(pred: pd.DataFrame) -> pd.Series:
    """Risk is the chance of never reaching the majors; the control-WAR quartiles sit at zero for most prospects, so they cannot band it."""
    return 1.0 - pred["p_mlb"]


def model_path(last_dev_season: int) -> Path:
    return MODELS_DIR / f"prospect_model_{last_dev_season}.json"


def _clean(obj):
    """JSON without NaN: NaN and infinities become null."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def publish_model(model: ProspectModel) -> None:
    if not any(model.validated.values()):
        raise ValueError("refusing to publish a prospect model with no validated pool")
    payload = json.dumps(_clean(model.to_payload()), allow_nan=False)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO prospect_models (model_version, last_dev_season, payload, fitted_at, updated_at) "
            "VALUES (:v, :s, CAST(:p AS jsonb), :f, now()) "
            "ON CONFLICT (model_version, last_dev_season) DO UPDATE SET payload = EXCLUDED.payload, "
            "fitted_at = EXCLUDED.fitted_at, updated_at = now()"
        ), {"v": model.model_version, "s": max(model.dev_seasons), "p": payload, "f": model.fitted_at})
    log.info(f"model: published {model.model_version} through {max(model.dev_seasons)}")


def _latest_payload(version: str) -> dict | None:
    with engine.connect() as conn:
        row = conn.execute(text("SELECT payload FROM prospect_models WHERE model_version = :v ORDER BY fitted_at DESC LIMIT 1"),
                           {"v": version}).fetchone()
    if row is None:
        return None
    return row[0] if isinstance(row[0], dict) else json.loads(row[0])


def load_latest_from_db() -> ProspectModel:
    payload = _latest_payload(MODEL_VERSION)
    if payload is None:
        raise RuntimeError("No prospect model published; run python -m v2.prospects.fit first")
    return ProspectModel.from_payload(payload)


def load_stuff_from_db():
    """The published stuff model, or None before one is published (pitchers then carry no stuff columns)."""
    from v2.prospects.stuff import StuffModel
    payload = _latest_payload(STUFF_VERSION)
    return None if payload is None else StuffModel.from_payload(payload)


def publish_stuff(model) -> None:
    payload = json.dumps(_clean(model.to_payload()), allow_nan=False)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO prospect_models (model_version, last_dev_season, payload, fitted_at, updated_at) "
            "VALUES (:v, :s, CAST(:p AS jsonb), :f, now()) "
            "ON CONFLICT (model_version, last_dev_season) DO UPDATE SET payload = EXCLUDED.payload, "
            "fitted_at = EXCLUDED.fitted_at, updated_at = now()"
        ), {"v": STUFF_VERSION, "s": max(model.seasons), "p": payload, "f": model.fitted_at})
    log.info(f"model: published {STUFF_VERSION} through {max(model.seasons)}")
