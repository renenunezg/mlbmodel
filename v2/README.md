# MLB Model v2 - Hierarchical Bayesian + Monte Carlo

Parallel rebuild of the modeling layer. v1 (XGBoost) remains in production at
renenunez.dev until v2 passes the Phase 6 acceptance gates.

## Architecture

```
DATA LAYER (reuses backend/data/*)
  parquet PA cache  ─┐
  MLB Stats API     ─┤
  bullpen_daily     ─┼─►  v2/data/pa_dataset.py  (per-PA frame)
  park_factors      ─┤
  odds              ─┘
          │
          ▼
BAYESIAN SKILL LAYER (PyMC + numpyro)
  batter_skill.py    - hierarchical wOBA by handedness
  pitcher_skill.py   - hierarchical FIP by role (SP / RP)
  park_effects.py    - multiplicative park factors, shrunk to neutral
  posteriors/*.nc    - saved traces (gitignored)
          │
          ▼
PA SIMULATOR
  pa_simulator.py    - (batter_draw, pitcher_draw, park, count) → outcome
          │
          ▼
GAME SIMULATOR (Monte Carlo, ≥10k sims/game)
  game_simulator.py  - inning-by-inning, lineup turns, bullpen state
  bullpen_state.py   - reliever queue + rest weighting
  lineup.py          - batting order
          │
          ▼
MARKET LAYER (integrals over sim distribution)
  moneyline.py / totals.py / run_line.py / kelly.py
          │
          ▼
EVALUATION
  baseline_v1/       - frozen XGBoost snapshot for benchmarking
  metrics.py         - Brier, log loss, calibration, Sharpe, CLV
  backtester.py      - walk-forward harness
```

## Layout

```
v2/
├── data/pa_dataset.py
├── bayesian/
│   ├── batter_skill.py
│   ├── pitcher_skill.py
│   ├── park_effects.py
│   └── posteriors/             # NetCDF traces (gitignored)
├── simulator/
│   ├── pa_simulator.py
│   ├── game_simulator.py
│   ├── bullpen_state.py
│   └── lineup.py
├── markets/{moneyline,totals,run_line,kelly}.py
├── evaluation/
│   ├── baseline_v1/            # frozen v1 model.py + simulation.py + calibrator
│   ├── metrics.py
│   └── backtester.py
├── pipeline/
│   ├── daily_run.py            # 5 AM PT full predict
│   ├── train.py                # full or incremental Bayesian fit
│   └── refresh_lineups.py      # hourly predict-only refresh
├── tests/
├── requirements.txt
└── README.md
```

## Coexistence with v1

- v1 is untouched; `daily-pipeline.yml` keeps calling `pipeline.py`.
- v2 writes to **`model_outputs_v2`** and **`model_outputs_season_v2`** -
  parallel tables, same schema as v1 plus quantile + lineup-status columns.
- Frontend continues reading v1 tables until cutover.
- Cutover after Phase 6 passes: rename v2 tables → v1, sunset XGBoost
  pipeline, update frontend types if quantile columns are surfaced.

## Setup (Mac)

```
python3 -m venv env                                # NOT myenv
source env/bin/activate
pip install -r requirements.txt -r v2/requirements.txt
pip install -e .
```

`numpyro==0.20.1` is pinned against `jax==0.7.2` / `jaxlib==0.7.2` -
newer JAX versions remove `xla_pmap_p` and break numpyro at import.
Re-pin only after testing a known-good combination.

## Running

```
python -m v2.pipeline.train --mode full          # weekly: fresh NUTS fit
python -m v2.pipeline.train --mode incremental   # daily: warm-start from prev posterior
python -m v2.pipeline.daily_run                  # 5 AM PT predict
python -m v2.pipeline.refresh_lineups            # hourly predict-only refresh
pytest v2/tests/
```

## Acceptance gates (non-negotiable)

| Phase | Gate |
|---|---|
| 2 | All R-hat < 1.01 and ESS > 400 on key parameters |
| 3 | Simulated league-wide K%/BB%/HR%/BABIP within **1pp** of MLB actuals |
| 4 | Simulated runs/game distribution mean & variance within **5%** of MLB |
| 6 | v2 max calibration-curve decile deviation < v1's, AND v2 log-loss ≤ v1 on moneyline |

## Style (per CLAUDE.md and project plan)

- Address user as "rene" (lowercase). Direct over deferential. Pushback welcome.
- Mac-first, Safari-friendly. `env`, never `myenv`. No nano.
- Before any non-trivial edit: state what it does and wait for approval.
- Two failed fix attempts → escalate, don't loop.
- Bayesian models replace XGBoost; XGBoost stays frozen as the v1 baseline.
