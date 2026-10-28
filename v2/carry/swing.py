"""Swing physics per ball: from the pitch, the measured swing and the observed launch, solve the collision for spin and fly
the ball, then compare with the same launch carrying the league's typical backspin for its launch angle. The difference
(feet) is what this ball's spin, as the swing produced it, added or cost."""
from __future__ import annotations

import numpy as np
import pandas as pd

from v2.carry import physics as ph

SPIN_LAUNCH = (10.0, 40.0)       # above 40 degrees the ball is a pop-up; low liners move little with spin but their solved
                                 # spin is the best read on a hitter's tendency (10-15 degree balls alone predict carry at .6)
BELT_HIGH_FT = 2.5


def pitch_height_curves(bip: pd.DataFrame) -> dict[str, list[float]]:
    """League tilt and attack angle as quadratics in pitch height: low pitches force steeper, more upward swings."""
    tracked = bip[bip.swing_path_tilt.notna() & bip.attack_angle.notna() & bip.plate_z.notna()]
    return {col: np.polyfit(tracked.plate_z, tracked[col], 2).tolist() for col in ("swing_path_tilt", "attack_angle")}


def at_belt_high(frame: pd.DataFrame, curves: dict[str, list[float]]) -> pd.DataFrame:
    """Each swing's tilt and attack angle restated at a belt-high pitch, and tilt relative to the league at its height."""
    out = pd.DataFrame(index=frame.index)
    for col, name in (("swing_path_tilt", "tilt"), ("attack_angle", "attack")):
        rel = frame[col] - np.polyval(curves[col], frame.plate_z)
        out[f"{name}_rel"] = rel
        out[name] = rel + np.polyval(curves[col], BELT_HIGH_FT)
    return out


def spin_balls(air: pd.DataFrame) -> pd.DataFrame:
    keep = (air.launch_angle.between(*SPIN_LAUNCH) & air.air_density.notna() & air.bat_speed.notna()
            & air.attack_angle.notna() & air.attack_direction.notna() & air.swing_path_tilt.notna() & air.pvy.notna())
    return air[keep].reset_index(drop=True)


def _launch_bins(launch: np.ndarray) -> pd.Categorical:
    return pd.cut(launch, np.arange(SPIN_LAUNCH[0] - 0.1, SPIN_LAUNCH[1] + 1.1, 1.0))


def solve(balls: pd.DataFrame, tilt_rel: np.ndarray, d_tilt: float = 0.0, d_attack: float = 0.0,
          typical: pd.Series | None = None, detail: bool = False,
          tilt_correction: float = ph.TILT_ATTACK_CORRECTION) -> tuple[pd.DataFrame, pd.Series]:
    """Spin effect per ball for the measured swing shifted by d_tilt and d_attack degrees, holding the launch fixed.

    typical is the league's median backspin (rad/s) by launch-angle degree; computed from this call when not given.
    detail adds the sidespin's own effect on distance and the hook or slice (landing minus launch direction, degrees).
    """
    right = (balls.stand == "R").to_numpy()
    tilt = balls.swing_path_tilt.to_numpy() + d_tilt
    attack = balls.attack_angle.to_numpy() + d_attack + tilt_correction * (tilt_rel + d_tilt)
    direction = np.where(right, 1, -1) * balls.attack_direction.to_numpy()
    pitch_v = balls[["pvx", "pvy", "pvz"]].to_numpy()
    v_out, spin, q = ph.collide(pitch_v, balls.bat_speed.to_numpy(), attack, direction, tilt, right,
                                balls.launch_speed.to_numpy(), balls.launch_angle.to_numpy())
    spin = spin * ph.SPIN_SCALE
    axis = ph.backspin_axis(v_out)
    back = (spin * axis).sum(1)
    bins = _launch_bins(balls.launch_angle.to_numpy())
    if typical is None:
        typical = pd.Series(back).groupby(bins, observed=True).median()
    typ = pd.Series(bins).map(typical).astype(float).to_numpy()
    density = balls.air_density.to_numpy()
    dist, landing = ph.fly(v_out, spin, density)
    dist_typ, _ = ph.fly(v_out, typ[:, None] * axis, density)
    rpm = 60 / (2 * np.pi)
    out = pd.DataFrame({"spin_effect": dist - dist_typ, "backspin": (back - typ) * rpm, "sidespin": np.abs(spin[:, 2]) * rpm,
                        "q": q, "pred_distance": dist}, index=balls.index)
    if detail:
        out[["spin_x", "spin_y", "spin_z"]] = spin
        out["launch_dir"] = np.degrees(np.arctan2(v_out[:, 0], v_out[:, 1]))
    if detail:
        dist_back, _ = ph.fly(v_out, back[:, None] * axis, density)
        out["sidespin_effect"] = dist - dist_back
        out["hook"] = landing - out.launch_dir
    return out, typical
