"""Load Phase 2 NetCDF traces and assemble per-actor offsets for the PA sim.

Reads `batter_skill.nc`, `pitcher_skill.nc`, `park_effects.nc`, takes posterior
means over (chain, draw), and returns a `PosteriorMeans` dataclass with arrays
laid out for vectorized inference.

Unknown actor fallback: each per-actor table has an extra trailing row of zeros.
Lookups for unknown ids resolve to that row, which sets the actor's offset to
zero (i.e., the league-mean actor).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np

from v2.bayesian._common import POSTERIORS_DIR, WOBA_WEIGHTS
from v2.data.pa_dataset import OUTCOMES

REF_IDX = OUTCOMES.index("OUT")
NON_REF_IDX = np.array([i for i in range(len(OUTCOMES)) if i != REF_IDX], dtype=np.int64)
K_FREE = len(NON_REF_IDX)
WOBA_VEC_FREE = np.array(
    [WOBA_WEIGHTS[OUTCOMES[i]] for i in NON_REF_IDX], dtype=np.float64
)
INTERCEPT_TOLERANCE = 0.30  # batter and pitcher fits use slightly different
# PA pools (pitcher fit drops position-player-pitchers), so per-outcome
# intercepts can diverge a bit. The park-effect stage already averages the two
# and the 1pp acceptance gate is the binding check.


@dataclass(frozen=True)
class PosteriorMeans:
    intercept: np.ndarray            # (K_FREE,) shared league logit baseline
    batter_offset: np.ndarray        # (n_b + 1, K_FREE), last row is zeros (fallback)
    platoon_offset: np.ndarray       # (n_b + 1, K_FREE), last row is zeros
    pitcher_offset: np.ndarray       # (n_p + 1, 2, K_FREE), last row+role is zeros
    park_log: np.ndarray             # (n_v + 1,), last entry is zero (fallback)
    batter_ids: np.ndarray           # sorted (n_b,), int64
    pitcher_ids: np.ndarray          # sorted (n_p,), int64
    venue_codes: np.ndarray          # (n_v,) array of str
    intercept_diff: float            # max|intercept_b - intercept_p|

    def encode_batter(self, ids: np.ndarray) -> np.ndarray:
        return _encode_int(ids, self.batter_ids)

    def encode_pitcher(self, ids: np.ndarray) -> np.ndarray:
        return _encode_int(ids, self.pitcher_ids)

    def encode_venue(self, codes: np.ndarray) -> np.ndarray:
        return _encode_str(codes, self.venue_codes)


def _encode_int(ids: np.ndarray, sorted_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    pos = np.searchsorted(sorted_ids, ids)
    pos_clipped = np.clip(pos, 0, len(sorted_ids) - 1)
    hit = (pos < len(sorted_ids)) & (sorted_ids[pos_clipped] == ids)
    out = np.where(hit, pos_clipped, len(sorted_ids))
    return out.astype(np.int64)


def _encode_str(codes: np.ndarray, sorted_codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes)
    # The venue list isn't huge (~30); a vectorized membership lookup via
    # np.where over the sorted codes is fine.
    order = np.argsort(sorted_codes)
    sorted_view = sorted_codes[order]
    pos = np.searchsorted(sorted_view, codes)
    pos_clipped = np.clip(pos, 0, len(sorted_view) - 1)
    hit = (pos < len(sorted_view)) & (sorted_view[pos_clipped] == codes)
    out = np.where(hit, order[pos_clipped], len(sorted_codes))
    return out.astype(np.int64)


def _posterior_mean(idata: az.InferenceData, var: str) -> np.ndarray:
    return idata.posterior[var].mean(("chain", "draw")).values


def _assemble(
    *,
    intercept_b: np.ndarray,
    intercept_p: np.ndarray,
    sigma_batter: np.ndarray,
    z_batter: np.ndarray,
    sigma_platoon: np.ndarray,
    z_platoon: np.ndarray,
    sigma_pitcher: np.ndarray,
    z_pitcher: np.ndarray,
    park_log_real: np.ndarray,
    batter_ids: np.ndarray,
    pitcher_ids: np.ndarray,
    venue_codes: np.ndarray,
    enforce_intercept_tolerance: bool,
) -> PosteriorMeans:
    intercept_diff = float(np.abs(intercept_b - intercept_p).max())
    if enforce_intercept_tolerance and intercept_diff > INTERCEPT_TOLERANCE:
        raise RuntimeError(
            f"batter and pitcher intercepts disagree by {intercept_diff:.4f} > "
            f"{INTERCEPT_TOLERANCE}; investigate before using the simulator."
        )

    beta_main = sigma_batter[None, :] * z_batter
    beta_platoon = sigma_platoon[None, :] * z_platoon
    n_b = beta_main.shape[0]
    batter_offset = np.vstack([beta_main, np.zeros((1, K_FREE))])
    platoon_offset = np.vstack([beta_platoon, np.zeros((1, K_FREE))])

    # ROLES = ("SP", "RP"); index 0 = SP, 1 = RP per pitcher_skill.py.
    beta_pitcher = sigma_pitcher[None, :, :] * z_pitcher[:, None, :]
    n_p = beta_pitcher.shape[0]
    pitcher_offset = np.concatenate([beta_pitcher, np.zeros((1, 2, K_FREE))], axis=0)

    park_log = np.concatenate([park_log_real, [0.0]])

    # searchsorted requires sorted ids; defensive resort if upstream order changes.
    batter_ids = batter_ids.copy()
    pitcher_ids = pitcher_ids.copy()
    if not np.all(np.diff(batter_ids) > 0):
        order = np.argsort(batter_ids)
        batter_ids = batter_ids[order]
        batter_offset[:n_b] = batter_offset[:n_b][order]
        platoon_offset[:n_b] = platoon_offset[:n_b][order]
    if not np.all(np.diff(pitcher_ids) > 0):
        order = np.argsort(pitcher_ids)
        pitcher_ids = pitcher_ids[order]
        pitcher_offset[:n_p] = pitcher_offset[:n_p][order]

    intercept = 0.5 * (intercept_b + intercept_p)

    return PosteriorMeans(
        intercept=intercept,
        batter_offset=batter_offset,
        platoon_offset=platoon_offset,
        pitcher_offset=pitcher_offset,
        park_log=park_log,
        batter_ids=batter_ids,
        pitcher_ids=pitcher_ids,
        venue_codes=venue_codes.copy(),
        intercept_diff=intercept_diff,
    )


def load_posteriors(posteriors_dir: Path = POSTERIORS_DIR) -> PosteriorMeans:
    bat = az.from_netcdf(posteriors_dir / "batter_skill.nc")
    pit = az.from_netcdf(posteriors_dir / "pitcher_skill.nc")
    park = az.from_netcdf(posteriors_dir / "park_effects.nc")

    return _assemble(
        intercept_b=_posterior_mean(bat, "intercept"),
        intercept_p=_posterior_mean(pit, "intercept"),
        sigma_batter=_posterior_mean(bat, "sigma_batter"),
        z_batter=_posterior_mean(bat, "z_batter"),
        sigma_platoon=_posterior_mean(bat, "sigma_platoon"),
        z_platoon=_posterior_mean(bat, "z_platoon"),
        sigma_pitcher=_posterior_mean(pit, "sigma_pitcher"),
        z_pitcher=_posterior_mean(pit, "z_pitcher"),
        park_log_real=_posterior_mean(park, "park_log"),
        batter_ids=bat.posterior["batter"].values.astype(np.int64),
        pitcher_ids=pit.posterior["pitcher"].values.astype(np.int64),
        venue_codes=park.posterior["venue"].values.astype(str),
        enforce_intercept_tolerance=True,
    )


def load_posterior_draws(
    rng: np.random.Generator,
    K: int = 30,
    posteriors_dir: Path = POSTERIORS_DIR,
) -> list[PosteriorMeans]:
    """Sample K random posterior realizations from the chain×draw axes.

    Each returned PosteriorMeans is one draw (not a mean), so consuming it via
    simulate_pa_batch / simulate_game propagates parameter uncertainty into the
    resulting run distribution. K~30 is enough for stable p10/p90 win-prob bands;
    the simulator's per-PA randomness still dominates total variance.

    Intercept tolerance is NOT enforced per-draw — the per-draw |int_b - int_p|
    is much noisier than on means, but the means already pass the check via
    load_posteriors().
    """
    bat = az.from_netcdf(posteriors_dir / "batter_skill.nc")
    pit = az.from_netcdf(posteriors_dir / "pitcher_skill.nc")
    park = az.from_netcdf(posteriors_dir / "park_effects.nc")

    bat_s = bat.posterior.stack(sample=("chain", "draw"))
    pit_s = pit.posterior.stack(sample=("chain", "draw"))
    park_s = park.posterior.stack(sample=("chain", "draw"))

    n_samples = bat_s.sizes["sample"]
    if K > n_samples:
        raise ValueError(f"K={K} exceeds available draws {n_samples}")
    indices = rng.choice(n_samples, size=K, replace=False)

    bat_intercept = bat_s["intercept"].isel(sample=indices).values    # (K_FREE, K)
    bat_sigma = bat_s["sigma_batter"].isel(sample=indices).values     # (K_FREE, K)
    bat_z = bat_s["z_batter"].isel(sample=indices).values             # (n_b, K_FREE, K)
    bat_sigma_pt = bat_s["sigma_platoon"].isel(sample=indices).values # (K_FREE, K)
    bat_z_pt = bat_s["z_platoon"].isel(sample=indices).values         # (n_b, K_FREE, K)

    pit_intercept = pit_s["intercept"].isel(sample=indices).values    # (K_FREE, K)
    pit_sigma = pit_s["sigma_pitcher"].isel(sample=indices).values    # (role=2, K_FREE, K)
    pit_z = pit_s["z_pitcher"].isel(sample=indices).values            # (n_p, K_FREE, K)

    park_log = park_s["park_log"].isel(sample=indices).values         # (n_v, K)

    batter_ids = bat.posterior["batter"].values.astype(np.int64)
    pitcher_ids = pit.posterior["pitcher"].values.astype(np.int64)
    venue_codes = park.posterior["venue"].values.astype(str)

    out: list[PosteriorMeans] = []
    for k in range(K):
        out.append(_assemble(
            intercept_b=bat_intercept[..., k],
            intercept_p=pit_intercept[..., k],
            sigma_batter=bat_sigma[..., k],
            z_batter=bat_z[..., k],
            sigma_platoon=bat_sigma_pt[..., k],
            z_platoon=bat_z_pt[..., k],
            sigma_pitcher=pit_sigma[..., k],
            z_pitcher=pit_z[..., k],
            park_log_real=park_log[..., k],
            batter_ids=batter_ids,
            pitcher_ids=pitcher_ids,
            venue_codes=venue_codes,
            enforce_intercept_tolerance=False,
        ))
    return out
