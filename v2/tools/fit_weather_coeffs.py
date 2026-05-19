"""Empirical weather coefficients (Option A) for the per-PA logit shift.

For each target in (HR, 3B, 2B) fit a logistic regression on per-PA outcomes:

    logit P(target) ~ wind_signal + temp_f_c + is_dome
                       + C(park) + batter_rate + pitcher_rate

wind_signal = wind_speed_mph * wind_out_component (park-relative, signed).
batter_rate / pitcher_rate are each actor's overall target rate - cheap
controls that absorb most batter/pitcher quality without 1800 FE dummies.
temp_f_c is centered at 70F so the intercept stays interpretable.

The wind_signal and temp coefficients are log-odds shifts per unit, directly
usable as additive logit shifts in pa_sim. Prints a constants block to paste
into v2/simulator/weather_effects.py. Does NOT write any code itself - the
coefficients get a human checkpoint before wiring.

    env/bin/python -m v2.tools.fit_weather_coeffs --years 2024 2025
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sqlalchemy import text

from backend.db import engine
from v2.data.pa_dataset import load_pa_dataset

TARGETS = ("HR", "3B", "2B")
TEMP_CENTER = 70.0


def _load_weather() -> pd.DataFrame:
    w = pd.read_sql(
        text("SELECT game_pk, wind_speed_mph, wind_out_component, temp_f, is_dome FROM weather"),
        engine,
    )
    w["wind_signal"] = w["wind_speed_mph"].astype(float) * w["wind_out_component"].astype(float)
    w["temp_f_c"] = w["temp_f"].astype(float) - TEMP_CENTER
    w["is_dome"] = w["is_dome"].astype(int)
    return w[["game_pk", "wind_signal", "temp_f_c", "is_dome"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2024, 2025])
    args = ap.parse_args()

    pa = load_pa_dataset(min(args.years), max(args.years))
    w = _load_weather()
    df = pa.merge(w, on="game_pk", how="inner")
    # dome games have no wind signal; temp indoors is climate-controlled noise.
    df.loc[df["is_dome"] == 1, ["wind_signal", "temp_f_c"]] = 0.0
    n_total = len(df)
    df = df.dropna(subset=["wind_signal", "temp_f_c", "home_team"])
    print(f"PAs: {len(df):,} of {n_total:,} after weather join (years {args.years})")
    print(f"wind_signal: mean={df['wind_signal'].mean():.2f} "
          f"sd={df['wind_signal'].std():.2f} range=[{df['wind_signal'].min():.0f},"
          f"{df['wind_signal'].max():.0f}]")

    park = pd.get_dummies(df["home_team"], prefix="pk", drop_first=True).astype(float)

    print("\ntarget |  wind_signal (p)   |   temp_f (p)      |  dome (p)   |  base%")
    print("-" * 78)
    out = {}
    for tgt in TARGETS:
        y = (df["outcome"] == tgt).astype(int).to_numpy()
        b_rate = df.assign(_y=y).groupby("batter")["_y"].transform("mean")
        p_rate = df.assign(_y=y).groupby("pitcher")["_y"].transform("mean")
        X = pd.concat([
            pd.Series(df["wind_signal"].to_numpy(), name="wind_signal"),
            pd.Series(df["temp_f_c"].to_numpy(), name="temp_f_c"),
            pd.Series(df["is_dome"].to_numpy(), name="is_dome"),
            pd.Series(b_rate.to_numpy(), name="batter_rate"),
            pd.Series(p_rate.to_numpy(), name="pitcher_rate"),
            park.reset_index(drop=True),
        ], axis=1)
        names = list(X.columns)
        Xm = X.to_numpy().astype(float)
        clf = LogisticRegression(penalty=None, max_iter=500, solver="lbfgs")
        clf.fit(Xm, y)
        beta = clf.coef_.ravel()
        # analytic SE via inv(X' W X), W = diag(p(1-p)); add intercept col
        Xi = np.column_stack([np.ones(len(Xm)), Xm])
        p = clf.predict_proba(Xm)[:, 1]
        Wd = p * (1 - p)
        cov = np.linalg.inv((Xi * Wd[:, None]).T @ Xi)
        se = np.sqrt(np.diag(cov))[1:]  # drop intercept
        z = beta / se
        pv = 2 * (1 - stats.norm.cdf(np.abs(z)))
        coef = dict(zip(names, beta))
        pval = dict(zip(names, pv))
        out[tgt] = {"wind_signal": coef["wind_signal"], "temp_f_c": coef["temp_f_c"],
                    "is_dome": coef["is_dome"]}
        print(f"{tgt:>4}   | {coef['wind_signal']:+.5f} ({pval['wind_signal']:.3f}) "
              f"| {coef['temp_f_c']:+.5f} ({pval['temp_f_c']:.3f}) "
              f"| {coef['is_dome']:+.3f} ({pval['is_dome']:.3f}) "
              f"| {100*y.mean():.2f}")

    print("\n# paste into v2/simulator/weather_effects.py (after rene checkpoint):")
    print("WEATHER_COEF = {")
    for tgt, c in out.items():
        print(f'    "{tgt}": {{"wind_signal": {c["wind_signal"]:+.6f}, '
              f'"temp_f_c": {c["temp_f_c"]:+.6f}}},')
    print("}")
    print(f"TEMP_CENTER = {TEMP_CENTER}")


if __name__ == "__main__":
    main()
