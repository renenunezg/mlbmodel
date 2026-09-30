"""Chronological, game-weighted backtest of live win expectancy. Never writes a DB.

Run: python -m v2.live.backtest --schedule-dir /tmp/mlb-live-wp --output /tmp/mlb-live-wp
Schedules are MLB Stats API schedule responses with hydrate=linescore, saved as
schedule-YYYY.json. Statcast inputs are the existing regular-season parquet caches.
2024 fits, 2025 selects count shrinkage, and 2026 is excluded from both.
Use --test-cache to evaluate a separate, untouched 2026 temporal holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v2.live.model import WinExpectancy

COLUMNS = ["game_pk", "game_date", "game_type", "inning", "inning_topbot", "outs_when_up",
           "on_1b", "on_2b", "on_3b", "balls", "strikes", "home_score", "away_score",
           "post_home_score", "post_away_score", "at_bat_number", "pitch_number"]


def load_states(cache: Path, schedule_path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_parquet(cache, columns=COLUMNS)
    raw = raw.loc[raw.game_type == "R"].sort_values(["game_pk", "at_bat_number", "pitch_number"])
    source_games = raw.game_pk.nunique()
    duplicated = int(raw.duplicated(["game_pk", "at_bat_number", "pitch_number"]).sum())
    if duplicated:
        raise ValueError(f"Duplicate source pitches: {duplicated}")
    games, innings = [], []
    for day in json.loads(schedule_path.read_text())["dates"]:
        for game in day["games"]:
            ls = game.get("linescore", {})
            if (game.get("gameType") != "R" or game["status"]["abstractGameState"] != "Final"
                    or ls.get("scheduledInnings", 9) != 9 or ls.get("currentInning", 0) < 9
                    or game.get("resumeDate") or game.get("resumedFrom")):
                continue
            h, a = (game["teams"][side].get("score") for side in ("home", "away"))
            if h is None or a is None or h == a:
                continue
            games.append({"game_pk": game["gamePk"], "final_home": h, "final_away": a})
            totals = {"home": 0, "away": 0}
            for inning in ls.get("innings", []):
                for side in ("away", "home"):
                    runs = inning.get(side, {}).get("runs")
                    if runs is not None:
                        innings.append({"game_pk": game["gamePk"], "inning": inning["num"],
                                        "top": side == "away", "half_runs": runs,
                                        "half_start_score": totals[side]})
                        totals[side] += runs
    final = pd.DataFrame(games).drop_duplicates("game_pk")
    # Reject incomplete captures or disagreements instead of deriving a winner from
    # the last cached pitch of an unfinished game.
    last = raw.groupby("game_pk", sort=False).tail(1).merge(final, on="game_pk", validate="one_to_one")
    valid = last.loc[(last.post_home_score == last.final_home) & (last.post_away_score == last.final_away)]
    frame = raw.loc[raw.game_pk.isin(valid.game_pk)].copy()
    frame["top"] = frame.inning_topbot.eq("Top")
    frame["outs"] = frame.outs_when_up
    frame["bases"] = sum(frame[f"on_{b}b"].notna().astype(int) * (1 << (b - 1)) for b in (1, 2, 3))
    frame = frame.merge(final, on="game_pk", validate="many_to_one")
    frame = frame.merge(pd.DataFrame(innings).drop_duplicates(["game_pk", "inning", "top"]),
                        on=["game_pk", "inning", "top"], validate="many_to_one")
    frame["remaining_runs"] = frame.half_runs - (
        np.where(frame.top, frame.away_score, frame.home_score) - frame.half_start_score
    )
    frame["home_win"] = (frame.final_home > frame.final_away).astype(int)
    frame["automatic_runner"] = True
    required = ["inning", "outs", "bases", "balls", "strikes", "home_score", "away_score", "remaining_runs"]
    good = frame[required].notna().all(axis=1) & (frame.remaining_runs >= 0)
    good &= frame.balls.between(0, 3) & frame.strikes.between(0, 2) & frame.outs.between(0, 2)
    invalid_states = int((~good).sum())
    frame = frame.loc[good].copy()
    frame[required] = frame[required].astype(int)
    frame["game_date"] = pd.to_datetime(frame.game_date).dt.strftime("%Y-%m-%d")
    # Repeated foul pitches at an identical state do not receive extra weight.
    frame = frame.drop_duplicates(["game_pk", "at_bat_number", "inning", "top", "outs", "bases",
                                   "balls", "strikes", "home_score", "away_score"])
    report = {"source_games": int(source_games), "eligible_games": int(frame.game_pk.nunique()),
              "excluded_games": int(source_games - frame.game_pk.nunique()), "states": len(frame),
              "invalid_states": invalid_states, "min_date": frame.game_date.min(), "max_date": frame.game_date.max(),
              "exclusions": "Non-final, resumed, shortened, incomplete score captures or unavailable official inning totals"}
    return frame, report


def game_losses(frame, probability):
    p = np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)
    y = frame.home_win.to_numpy()
    return pd.DataFrame({"game_pk": frame.game_pk.to_numpy(), "brier": (p - y) ** 2,
                         "log_loss": -(y * np.log(p) + (1 - y) * np.log1p(-p))}).groupby("game_pk").mean()


def metrics(frame, probability):
    losses = game_losses(frame, probability)
    weighted = frame[["game_pk", "home_win"]].copy()
    weighted["p"] = probability
    weighted["weight"] = 1 / weighted.groupby("game_pk").game_pk.transform("size")
    weighted["bucket"] = pd.cut(weighted.p, bins=np.linspace(0, 1, 11), include_lowest=True)
    calibration = []
    for bucket, group in weighted.groupby("bucket", observed=True):
        calibration.append({"bucket": str(bucket), "states": len(group), "games": int(group.game_pk.nunique()),
                            "predicted": float(np.average(group.p, weights=group.weight)),
                            "observed": float(np.average(group.home_win, weights=group.weight))})
    return {"games": len(losses), "states": len(frame), **losses.mean().to_dict(), "calibration": calibration}


def paired_comparison(frame, baseline, candidate):
    delta = game_losses(frame, candidate) - game_losses(frame, baseline)
    rng = np.random.default_rng(20260930)
    draws = np.array([delta.to_numpy()[rng.integers(0, len(delta), len(delta))].mean(axis=0) for _ in range(2000)])
    intervals = np.quantile(draws, [.025, .975], axis=0)
    return {name: {"delta": float(delta[name].mean()), "ci95": intervals[:, i].tolist()}
            for i, name in enumerate(delta.columns)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, default=Path("cache"))
    ap.add_argument("--schedule-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--test-cache", type=Path, help="Separate 2026 holdout cache; never used for fitting or selection")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train, train_quality = load_states(args.cache_dir / "statcast_2024.parquet", args.schedule_dir / "schedule-2024.json")
    dev, dev_quality = load_states(args.cache_dir / "statcast_2025.parquet", args.schedule_dir / "schedule-2025.json")
    if (not train.game_date.str.startswith("2024-").all()
            or not dev.game_date.str.startswith("2025-").all()
            or set(train.game_pk) & set(dev.game_pk)):
        raise ValueError("Fit and development sets must be separate 2024 and 2025 games")
    print(f"Fit {train_quality}; development {dev_quality}", flush=True)
    baseline = WinExpectancy.fit(train, use_count=False)
    dev_baseline = baseline.predict_frame(dev)
    selection = {"no_count": metrics(dev, dev_baseline)}
    # Small, declared grid, selected exclusively on 2025 log loss.
    best_strength, best_loss = None, selection["no_count"]["log_loss"]
    for strength in (50, 200, 800):
        model = WinExpectancy.fit(train, prior_strength=strength)
        p = model.predict_frame(dev)
        selection[str(strength)] = metrics(dev, p)
        print("Development", strength, {k: v for k, v in selection[str(strength)].items() if k != "calibration"}, flush=True)
        if selection[str(strength)]["log_loss"] < best_loss:
            best_strength, best_loss = strength, selection[str(strength)]["log_loss"]
    # The selection is now frozen. No 2026 data was used to fit or select.
    all_train = pd.concat([train, dev], ignore_index=True)
    baseline = WinExpectancy.fit(all_train, use_count=False)
    chosen = WinExpectancy.fit(all_train, use_count=best_strength is not None, prior_strength=best_strength or 100)
    chosen.metadata["selected_on"] = "2025; trained on 2024"
    chosen.metadata["locked_test"] = "2026; no tuning"
    chosen.save(args.output / "win_expectancy.json")
    baseline.save(args.output / "baseline.json")
    test_cache = args.test_cache or args.cache_dir / "statcast_2026.parquet"
    test, test_quality = load_states(test_cache,
                                    args.schedule_dir / "schedule-2026.json")
    if (all_train.game_date.max() >= test.game_date.min() or not test.game_date.str.startswith("2026-").all()
            or set(all_train.game_pk) & set(test.game_pk)):
        raise ValueError("Training and test dates overlap")
    p0, p1 = baseline.predict_frame(test), chosen.predict_frame(test)
    masks = {"all": np.ones(len(test), dtype=bool), "innings_1_3": test.inning <= 3,
             "innings_4_6": test.inning.between(4, 6), "innings_7_plus": test.inning >= 7,
             "late_close": (test.inning >= 7) & ((test.home_score - test.away_score).abs() <= 2),
             "extras": test.inning >= 10}
    holdout = {}
    for name, mask in masks.items():
        f = test.loc[mask]
        holdout[name] = {"baseline": metrics(f, p0[mask]), "candidate": metrics(f, p1[mask]),
                         "paired_game_bootstrap": paired_comparison(f, p0[mask], p1[mask])}
    report = {"data": {"train": train_quality, "development": dev_quality, "test": test_quality},
              "selection": selection, "selected_count_prior": best_strength,
              "holdout": holdout, "artifact": chosen.metadata,
              "limitations": ["Regular-season holdout only; postseason rules supported but accuracy unvalidated",
                              "Neutral player strength; no live market comparison",
                              "Historical replay uses corrected final source records, not measured feed latency"]}
    inputs = [args.cache_dir / f"statcast_{year}.parquet" for year in (2024, 2025)]
    inputs += [test_cache, *[args.schedule_dir / f"schedule-{year}.json" for year in (2024, 2025, 2026)]]
    report["inputs"] = {}
    for path in inputs:
        with path.open("rb") as source:
            report["inputs"][str(path)] = hashlib.file_digest(source, "sha256").hexdigest()
    (args.output / "backtest.json").write_text(json.dumps(report, indent=2))
    test.assign(baseline=p0, probability=p1).to_parquet(args.output / "holdout.parquet", index=False)
    print(json.dumps({"selected_count_prior": best_strength,
                      "holdout": {k: {"baseline": v["baseline"]["log_loss"],
                                      "candidate": v["candidate"]["log_loss"],
                                      "paired": v["paired_game_bootstrap"]} for k, v in holdout.items()}}, indent=2))


if __name__ == "__main__":
    main()
