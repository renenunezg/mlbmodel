"""Ball flight and bat-ball collision physics.

Flight: drag and Magnus lift with Nathan's 2017 coefficients (Analysis of Baseball Trajectories, fitted on Trackman
trajectories with measured spin), plus a drag rise below about 78 mph fitted on 2026 distances and validated on 2025.
Collision: the Cross and Nathan (2006) oblique collision with the ball rolling off the bat (no grip, which fit best),
the bat treated as a cylinder whose axis is tilted by a quarter of the swing path tilt. Air density comes from
elevation and game temperature at standard pressure.

Frame: x toward first base, y toward center field, z up; meters and seconds.
"""
from __future__ import annotations

import numpy as np

MPH = 0.44704
FT = 0.3048
BALL_R = 0.0369
BALL_M = 0.145
BALL_A = np.pi * BALL_R ** 2
G = 9.81
CONTACT_HEIGHT = 0.9

CD_HIGH_SPEED = 0.299            # Nathan's 0.297 at the speeds he fitted
CD_LOW_SPEED_RISE = 0.10         # extra drag below the transition speed
CD_TRANSITION = 35.0             # m/s, about 78 mph
CD_TRANSITION_WIDTH = 3.0
CD_PER_KRPM = 0.0292
SPIN_DECAY_S = 30.0

ROLL = 5 / 7                     # tangential speed kept and spin produced by a ball rolling off the bat
BAT_AXIS_TILT_SHARE = 0.25       # bat axis tilt as a share of swing path tilt; larger shares bias direction to the opposite field
TILT_ATTACK_CORRECTION = 0.6     # degrees of extra rise at contact per degree of tilt above the league at that pitch height
SPIN_SCALE = 0.5                 # collision spin scaled to the joint fit with distance; absolute rpm is not identified


def air_density(elevation_ft: np.ndarray, temp_f: np.ndarray) -> np.ndarray:
    pressure = 101325 * (1 - 2.25577e-5 * np.asarray(elevation_ft) * FT) ** 5.25588
    kelvin = (np.asarray(temp_f) - 32) * 5 / 9 + 273.15
    return pressure / (287.05 * kelvin)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(1)


def fly(v0: np.ndarray, spin: np.ndarray, density: np.ndarray, dt: float = 0.01, t_max: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """Carry distance (ft) and landing direction (degrees, + toward right field) for launch velocities v0 (n x 3, m/s)
    and spin vectors (n x 3, rad/s). Midpoint integration to field level."""
    r = np.zeros_like(v0)
    r[:, 2] = CONTACT_HEIGHT
    v = v0.copy()
    k = (density * BALL_A / (2 * BALL_M))[:, None]
    w_mag = np.linalg.norm(spin, axis=1)
    w_hat = spin / np.maximum(w_mag, 1e-9)[:, None]
    rpm = w_mag * 60 / (2 * np.pi)
    landed = np.full((len(v0), 2), np.nan)

    def accel(v: np.ndarray, t: float) -> np.ndarray:
        s = np.linalg.norm(v, axis=1)
        v_hat = v / s[:, None]
        decay = np.exp(-t / SPIN_DECAY_S)
        w_perp = w_hat - _dot(w_hat, v_hat)[:, None] * v_hat
        w_perp_mag = np.linalg.norm(w_perp, axis=1)
        spin_factor = BALL_R * w_mag * decay * w_perp_mag / s
        cl = 1.12 * spin_factor / (0.583 + 2.333 * spin_factor)
        cd = (CD_HIGH_SPEED + CD_LOW_SPEED_RISE / (1 + np.exp((s - CD_TRANSITION) / CD_TRANSITION_WIDTH))
              + CD_PER_KRPM * rpm * decay / 1000)
        lift_dir = np.cross(w_perp / np.maximum(w_perp_mag, 1e-9)[:, None], v_hat)
        return -k * (cd * s)[:, None] * v + k * (cl * s * s)[:, None] * lift_dir - np.array([0.0, 0.0, G])

    t = 0.0
    while t < t_max:
        a1 = accel(v, t)
        v_mid = v + a1 * dt / 2
        a2 = accel(v_mid, t + dt / 2)
        r_next = r + v_mid * dt
        v = v + a2 * dt
        down = (r_next[:, 2] <= 0) & np.isnan(landed[:, 0])
        frac = (r[down, 2] / (r[down, 2] - r_next[down, 2]))[:, None]
        landed[down] = r[down, :2] + frac * (r_next[down, :2] - r[down, :2])
        r = r_next
        t += dt
        if not np.isnan(landed[:, 0]).any():
            break
    x, y = landed[:, 0], landed[:, 1]
    return np.hypot(x, y) / FT, np.degrees(np.arctan2(x, y))


def launch_velocity(exit_mph: np.ndarray, launch_deg: np.ndarray, spray_deg: np.ndarray) -> np.ndarray:
    ev = np.asarray(exit_mph) * MPH
    th, sp = np.radians(launch_deg), np.radians(spray_deg)
    return np.column_stack([ev * np.cos(th) * np.sin(sp), ev * np.cos(th) * np.cos(sp), ev * np.sin(th)])


def backspin_axis(velocity: np.ndarray) -> np.ndarray:
    """Unit spin axis of pure backspin for a ball moving along `velocity` (horizontal, to the right of the heading)."""
    up = np.broadcast_to(np.array([0.0, 0.0, 1.0]), velocity.shape)
    return _unit(np.cross(_unit(velocity), up))


def collide(pitch_v: np.ndarray, bat_speed_mph: np.ndarray, attack_deg: np.ndarray, attack_dir_deg: np.ndarray,
            tilt_deg: np.ndarray, right_handed: np.ndarray, exit_mph: np.ndarray, launch_deg: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve each collision for the contact offset that reproduces the observed exit speed and launch angle.

    attack_dir_deg is the bat's horizontal direction in the field frame (+ toward right field). Returns the launch
    velocity (n x 3), the spin vector before SPIN_SCALE (n x 3, rad/s) and the collision efficiency q.
    """
    vb = np.asarray(bat_speed_mph) * MPH
    al, be = np.radians(attack_deg), np.radians(attack_dir_deg)
    v_bat = np.column_stack([vb * np.cos(al) * np.sin(be), vb * np.cos(al) * np.cos(be), vb * np.sin(al)])
    u = pitch_v - v_bat
    e = -_unit(u)
    up = np.broadcast_to(np.array([0.0, 0.0, 1.0]), e.shape)
    bat_dir = _unit(v_bat)
    heading = _unit(np.column_stack([bat_dir[:, 0], bat_dir[:, 1], np.zeros(len(bat_dir))]))
    left = np.cross(up, heading)
    toward_hitter = np.where(np.asarray(right_handed)[:, None], left, -left)
    side = _unit(toward_hitter - _dot(toward_hitter, bat_dir)[:, None] * bat_dir)
    lift = _unit(up - _dot(up, bat_dir)[:, None] * bat_dir)
    tau = BAT_AXIS_TILT_SHARE * np.radians(tilt_deg)[:, None]
    axis = _unit(np.cos(tau) * side + np.sin(tau) * lift)
    e_perp = _unit(e - _dot(e, axis)[:, None] * axis)
    f = _unit(np.cross(e_perp, axis))
    f = np.where((f[:, 2] < 0)[:, None], -f, f)
    exit_ms = np.asarray(exit_mph) * MPH
    target = np.radians(launch_deg)

    def outgoing(offset: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = _unit(np.cos(offset)[:, None] * e_perp + np.sin(offset)[:, None] * f)
        u_n = _dot(u, n)
        u_t = u - u_n[:, None] * n
        a = v_bat + ROLL * u_t
        c = (-u_n)[:, None] * n
        ac, cc = _dot(a, c), _dot(c, c)
        q = (-ac + np.sqrt(np.maximum(ac ** 2 - cc * (_dot(a, a) - exit_ms ** 2), 0))) / cc
        spin = (ROLL * np.linalg.norm(u_t, axis=1) / BALL_R)[:, None] * _unit(np.cross(n, _unit(u_t)))
        return a + q[:, None] * c, spin, q

    lo = np.full(len(u), np.radians(-40.0))
    hi = np.full(len(u), np.radians(70.0))
    for _ in range(40):
        mid = (lo + hi) / 2
        v_out, _, _ = outgoing(mid)
        rising = np.arcsin(np.clip(v_out[:, 2] / exit_ms, -1, 1)) < target
        lo, hi = np.where(rising, mid, lo), np.where(rising, hi, mid)
    return outgoing((lo + hi) / 2)
