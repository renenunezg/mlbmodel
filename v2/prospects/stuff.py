"""Stuff and command: what a pitch's shape and a pitcher's locations are worth, learned on MLB pitches.

Stuff: one gradient-boosted model per pitch family predicts a pitch's run value (Savant
``delta_run_exp``, batting side) from its shape alone (``pitches.SHAPE``), never its location or
count. Command: a location-and-count model minus the MLB mean run value of the count (a 72-cell table
of balls, strikes, platoon side and pitch family), so a pitch's command value is what its location
added beyond the count it was thrown in. Per pitcher, season and pitch type the
mean prediction per 100 pitches is mapped to 20-80 against the MLB distribution of that pitch type
(50 average, 10 per standard deviation, lower run value better); command against all MLB pitchers.

Minor-league and spring training readings and grades are shifted to the MLB scale by level offsets
fitted on pitchers measured at both levels in a season. A pitch family is released for publishing
only if its grades carry over to the same pitchers' MLB grades (``carry``): Triple-A's check releases
Triple-A and Florida State League pitches, spring training has its own.
"""
from __future__ import annotations

import base64
import gzip
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from v2.prospects import pitches

READINGS = ("velo", "ivb", "hb", "spin", "ext")
GRADE_MIN_PITCHES = 100  # a pitcher-season-type in the MLB grade scale
COMMAND_MIN_PITCHES = 300


@dataclass
class StuffModel:
    fitted_at: str = ""
    seasons: list[int] = field(default_factory=list)
    boosters: dict[str, str] = field(default_factory=dict)  # family, "command" -> gzipped base64 UBJSON
    count_table: dict[str, float] = field(default_factory=dict)  # "balls-strikes-same_side-family_code" -> MLB mean run value
    scale: dict[str, list[float]] = field(default_factory=dict)  # pitch type -> [mean, sd] of MLB rv100
    command_scale: list[float] = field(default_factory=list)
    floor: dict[str, int] = field(default_factory=dict)  # pitch type or "command" -> minimum pitches for a grade
    offsets: dict = field(default_factory=dict)  # level -> {reading or grade key -> MLB minus level}
    carry: dict = field(default_factory=dict)  # level -> family or "command" -> {"r", "bias", "n", "passed"}
    adjustment: dict = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)

    def booster(self, key: str) -> xgb.Booster:
        b = xgb.Booster()
        b.load_model(bytearray(gzip.decompress(base64.b64decode(self.boosters[key]))))
        return b

    def set_booster(self, key: str, booster: xgb.Booster) -> None:
        self.boosters[key] = base64.b64encode(gzip.compress(bytes(booster.save_raw("ubj")), 9)).decode("ascii")

    def to_payload(self) -> dict:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict) -> StuffModel:
        return cls(**{k: v for k, v in payload.items() if k in cls.__dataclass_fields__})

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_payload(), allow_nan=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> StuffModel:
        return cls.from_payload(json.loads(path.read_text(encoding="utf-8")))


def _matrix(frame: pd.DataFrame, cols: list[str], y=None) -> xgb.DMatrix:
    return xgb.DMatrix(frame[cols].astype(float), label=y, feature_names=cols, missing=np.nan)


def pitch_values(model: StuffModel, frame: pd.DataFrame) -> pd.DataFrame:
    """The frame with ``stuff_rv`` (shape-only expected run value) and ``command_rv`` (location value beyond the count) per pitch."""
    out = frame.copy()
    out["stuff_rv"] = np.nan
    for fam in pitches.FAMILIES:
        m = out["family"].eq(fam)
        if m.any():
            out.loc[m, "stuff_rv"] = model.booster(fam).predict(_matrix(out[m], pitches.SHAPE))
    ok = out["px"].notna() & out["pz"].notna()
    out["command_rv"] = np.nan
    if ok.any():
        sub = out[ok]
        out.loc[ok, "command_rv"] = (model.booster("command").predict(_matrix(sub, pitches.LOCATION))
                                     - count_key(sub).map(model.count_table).to_numpy(float))
    return out


def count_key(frame: pd.DataFrame) -> pd.Series:
    return frame[pitches.COUNT].astype(int).astype(str).agg("-".join, axis=1)


def stuff_grade(rv100: pd.Series, pitch_type: pd.Series, scale: dict[str, list[float]]) -> pd.Series:
    mean = pitch_type.map(lambda t: scale[t][0])
    sd = pitch_type.map(lambda t: scale[t][1])
    return (50 - 10 * (rv100 - mean) / sd).clip(20, 80)


def command_grade(rv100: pd.Series, scale: list[float]) -> pd.Series:
    return (50 - 10 * (rv100 - scale[0]) / scale[1]).clip(20, 80)


def by_type(values: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Per key and pitch type: pitch count, mean readings and stuff run value per 100 pitches."""
    g = values.groupby([*keys, "pitch_type"])
    out = g[list(READINGS)].mean()
    out["n"] = g.size()
    out["stuff_rv100"] = 100 * g["stuff_rv"].mean()
    return out.reset_index()


def by_pitcher_command(values: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = values[values["command_rv"].notna()].groupby(keys)
    return pd.DataFrame({"command_n": g.size(), "command_rv100": 100 * g["command_rv"].mean()}).reset_index()


PITCH_CODES = ("FF", "SI", "FC", "SL", "ST", "SV", "CU", "KC", "CS", "CH", "FS", "FO", "SC")  # published code = position + 1
LEVEL_CODES = {"aaa": 1, "fsl": 2, "mixed": 3, "spring": 4}
TRACKED_FLOOR = 200  # released pitches in a pitcher's window before he counts as tracked
MINOR_LEVELS = ("aaa", "fsl", "spring")
RELEASED_BY = {"aaa": "aaa", "fsl": "aaa", "spring": "spring"}  # level -> the carry-over check that releases it


def released(model: StuffModel, level: str) -> set[str]:
    """Pitch families and "command" released at a level by its carry-over check; only these are published or used.

    Command is released only where it was checked at the level itself (Triple-A); Florida State League and
    spring samples never reach the command floor in a pair, so their command is never shown.
    """
    checked = model.carry.get(RELEASED_BY[level], {})
    out = {k for k, v in checked.items() if v.get("passed")}
    return out if RELEASED_BY[level] == level else out - {"command"}


def adjusted(model: StuffModel, level: str) -> bool:
    return level == "aaa" or bool(model.offsets.get(f"{level}_adjusted"))


def _level_shift(frame: pd.DataFrame, model: StuffModel, level: str) -> pd.DataFrame:
    """Readings and the stuff grade on the MLB scale; a level whose offsets could not be fitted stays unshifted."""
    if not adjusted(model, level):
        return frame
    off = model.offsets[level]
    out = frame.copy()
    for r in READINGS:
        out[r] = out[r] + off[r]["value"]
    out["stuff_grade"] = (out["stuff_grade"] + out["family"].map(lambda f: off[f"stuff_grade:{f}"]["value"])).clip(20, 80)
    return out


def window_arsenal(model: StuffModel, season: int, weights: dict[int, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tracked prospect-level pitches over the window ending at ``season``, pooled with the window weights.

    Returns per pitcher and pitch type (released families only, at or above the type's floor):
    ``n, usage, readings, stuff_grade``; and per tracked pitcher: ``pitches, command_grade`` (null below
    the command floor or where command is not released), ``stuff_level``, ``fsl_share``, ``spring_share`` and
    ``stuff_adjusted``.
    """
    types, cmds = [], []
    for back, w in weights.items():
        for level in MINOR_LEVELS:
            pitches.require_complete(season - back, level)
            if not pitches.path(season - back, level).exists():
                continue
            ok = released(model, level)
            values = pitch_values(model, pitches.load(season - back, level))
            t = by_type(values, ["pitcher"])
            t = t[t["pitch_type"].isin(model.scale)]
            t["family"] = t["pitch_type"].map(pitches.FAMILY)
            t = t[t["family"].isin(ok)]
            t["stuff_grade"] = stuff_grade(t["stuff_rv100"], t["pitch_type"], model.scale)
            t = _level_shift(t, model, level)
            types.append(t.assign(w=w, level=level, adjusted=adjusted(model, level)))
            if "command" in ok:
                c = by_pitcher_command(values, ["pitcher"])
                c["command_grade"] = (command_grade(c["command_rv100"], model.command_scale)
                                      + model.offsets[level]["command_grade"]["value"]).clip(20, 80)
                cmds.append(c.assign(w=w))
    if not types:
        return pd.DataFrame(), pd.DataFrame()
    t = pd.concat(types, ignore_index=True)
    t["wn"] = t["w"] * t["n"]
    g = t.groupby(["pitcher", "pitch_type"])
    arsenal = pd.DataFrame({"n": g["n"].sum(), "family": g["family"].first()})
    for c in (*READINGS, "stuff_grade"):
        arsenal[c] = (t[c] * t["wn"]).groupby([t["pitcher"], t["pitch_type"]]).sum() / g["wn"].sum()
    arsenal = arsenal.reset_index()
    arsenal["usage"] = arsenal["n"] / arsenal.groupby("pitcher")["n"].transform("sum")
    arsenal = arsenal[arsenal["n"] >= arsenal["pitch_type"].map(model.floor)]
    by_pitcher = t.groupby("pitcher")
    pitchers = pd.DataFrame({"pitches": by_pitcher["n"].sum(),
                             "stuff_level": by_pitcher["level"].agg(lambda s: s.iloc[0] if s.nunique() == 1 else "mixed"),
                             "stuff_adjusted": by_pitcher["adjusted"].all()})
    for level in ("fsl", "spring"):
        share = t[t["level"].eq(level)].groupby("pitcher")["n"].sum()
        pitchers[f"{level}_share"] = (share.reindex(pitchers.index).fillna(0.0) / pitchers["pitches"]).astype(float)
    pitchers = pitchers[pitchers["pitches"] >= TRACKED_FLOOR]
    if cmds:
        c = pd.concat(cmds, ignore_index=True)
        c["wn"] = c["w"] * c["command_n"]
        cg = c.groupby("pitcher")
        grade = (c["command_grade"] * c["wn"]).groupby(c["pitcher"]).sum() / cg["wn"].sum()
        pitchers["command_grade"] = grade.where(cg["command_n"].sum() >= model.floor["command"]).reindex(pitchers.index)
    else:
        pitchers["command_grade"] = np.nan
    arsenal = arsenal[arsenal["pitcher"].isin(pitchers.index)]
    return arsenal.sort_values(["pitcher", "usage"], ascending=[True, False]).reset_index(drop=True), pitchers.reset_index()
