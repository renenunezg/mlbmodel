# MLB Expected Runs Model

A daily pipeline that predicts a per-team run distribution for every MLB game,
derives win, run-line, and totals probabilities by Monte Carlo simulation, and
publishes the output to a public dashboard each morning of the season. The
betting markets function as a calibration benchmark, not as a gambling
application: sharp participants push lines toward true probabilities quickly,
which makes them a higher-quality signal than most independently constructed
models.

The site is at [renenunez.dev](https://renenunez.dev) and lives in its own
repository, [momentumweb](https://github.com/renenunezg/momentumweb). Supabase
is the only interface between the two: this pipeline writes tables, the site
reads them. The methodology page describes the model in detail; this README
covers what's in this repository and how to run it.

## What's in the model

The current model (v2, live since 2026-05-12) is a two-layer system:

1. **Hierarchical Bayesian skill layer.**
   Batter and pitcher skill use partially pooled multinomial-logit models over eight outcomes: K, BB, HBP, 1B, 2B, 3B, HR, and OUT.
   Batters have a platoon effect; pitcher shrinkage uses the fitted SP/RP classification, with explicit protection for legitimate two-way pitchers.
   Park coefficients are fitted against the nonlinear expected-wOBA response of the same softmax transformation used by the simulator.
   The park prior is neutral in logit units, and the park observation likelihood remains in wOBA units.
   All three models use NUTS sampling and record their training cutoff and diagnostic gates with the saved traces.
2. **Per-PA Monte Carlo simulator.**
   Thirty posterior realizations share the configured simulation budget, which defaults to 10,000 simulations per game.
   Baserunner advancement uses empirical transitions with smoothing restricted to the same base/out state and explicit runner-conservation checks.
   Bullpens use prior role and rest evidence from the active roster.
   Rotation pitchers cannot enter the relief queue merely because they are rested.
   Workloads are sampled per appearance; unspecified opener plans receive short workloads, and unknown relief coverage uses a neutral arm.
   Explicit, timestamped opener/bulk reports can supply separate workloads.
   Relievers with at least 6 outs yesterday or 9 outs over two days are skipped.

Win, total, and run-line probabilities are computed from the simulated run distributions.
Published moneyline probabilities blend the home-field-adjusted simulation logit with the paired, de-vigged market consensus at a model weight of 0.2.
Raw simulation probabilities and the exact market snapshot are stored separately in prediction context.
The p10/p90 win-probability band estimates between-posterior variation after subtracting estimated binomial simulation variance.
Bands are unavailable when the parameter spread is below Monte Carlo resolution.
Prediction context records simulation error and posterior integration error separately.
Recommendations require known starters, complete posted lineups, and sufficient pitching-usage evidence; moneyline recommendations also require a paired market anchor.
Moneyline and run-line thresholds remain 4.5 percentage points above the executable price's implied probability, with quarter-Kelly sizing.
Totals recommendations remain disabled.

v2 replaced an XGBoost regressor (v1) after a 542-game head-to-head backtest:
Brier −6.9%, log-loss −7.3%, max calibration gap from 41.9% down to 3.2%, ROI
up on every market. The comparison tooling was retired after cutover. v1
predictions before 2026-05-12 still live in `model_outputs_v1_archive` and
`model_outputs_season_v1_archive`.

## Repository layout

```
pipeline.py             Shared schedule and nightly evaluation orchestrator.
backend/
  data/                 Fetchers: MLB Stats API, Statcast (pybaseball), Savant,
                        The Odds API, per-pitcher workload from boxscores.
  db.py                 SQLAlchemy engine pointed at Supabase via DATABASE_URL;
                        blocks writes outside CI unless MLBMODEL_DB_WRITES=1.
  log.py                Logging setup for the CLI entry points.
  team_mappings.py      3-letter codes + MLB team-id lookup table.
  kelly.py, simulation.py, metrics.py, strategy.py
                        Kelly sizing, odds conversions, Brier and log-loss,
                        EV thresholds and market anchoring constants.
v2/
  bayesian/             Batter, pitcher, and park models + fit_all
                        orchestrator. Posteriors saved to
                        v2/bayesian/posteriors/*.nc (gitignored).
  simulator/            posteriors loader, vectorized PA sampler, empirical
                        baserunner table, rest-aware bullpen, game loop.
  markets/              Empirical market probs, EV flags, Kelly, writer to
                        Supabase model_outputs.
  market_model/         Market-relative feature and residual research.
  pipeline/             daily_run, train, score_games, refresh_lineups,
                        verify, write_posterior_summaries.
  data/                 Multi-year Statcast cache builder + per-PA dataset.
  tools/                Frozen-forecast variance comparisons and weather calibration.
tests/                  Production-critical suite; see below.
```

## Running locally

```
# Backend (v2)
pip install -r requirements.txt
pip install -r v2/requirements.txt
pip install -e .

# Refit the Bayesian skill layer (writes NetCDF traces to v2/bayesian/posteriors/)
python -m v2.bayesian.fit_all --start-year 2024 --end-year 2026 --save-traces

# Daily v2 scoring run (assumes posteriors and statcast cache are populated)
python -m v2.pipeline.daily_run

# Score a specific date
python -m v2.pipeline.score_games --date 2026-09-09 --n-sims 10000

# Intraday lineup refresh (re-scores games whose posted lineup changed)
python -m v2.pipeline.refresh_lineups

# Prepare the additive prediction_context column before deploying this scoring version:
# backend/sql/prediction_context.sql (apply only with production approval)

# Export a no-write pregame forecast for later chronological acceptance
python -m v2.pipeline.score_games --date 2026-09-09 --no-write --export cache/candidate.json

# Compare frozen exports after those games finish
python -m v2.market_model.acceptance --candidate cache/candidate.json --baseline cache/baseline.json

# Market-relative research (only versioned raw forecasts qualify)
python -m v2.market_model.residual --start 2026-03-26 --end 2026-07-21 --market ml
python -m v2.market_model.features --start 2026-03-26 --end 2026-07-21

# Lean production-critical suite
pytest tests/

# Lint
ruff check .
```

Sampler pins are load-bearing: `numpyro==0.20.1` + `jax==0.7.2` +
`jaxlib==0.7.2`. Newer JAX dropped `xla_pmap_p`, which numpyro still uses, and
sampling fails silently if those drift. Pins are in `v2/requirements.txt`.

The first Statcast fetch for a prior season takes ~30 min. After that runs
read from `cache/` and finish in seconds. The cache is gitignored and reused
across CI runs via `actions/cache`.

## Environment

Root `.env`:
```
DATABASE_URL=postgresql://...   # Supabase session pooler URL.
ODDS_API_KEY=...                # the-odds-api.com key.
```

`DATABASE_URL` points at the live production database, so `backend/db.py`
refuses write statements unless `GITHUB_ACTIONS=true` (set by CI) or
`MLBMODEL_DB_WRITES=1` is set explicitly. Read-only local runs need neither.
Schema changes are applied directly in Supabase, not through migration files
in this repo.

## Database

All tables live in the `mlb` schema and key off `game_pk`, the integer ID
from the MLB Stats API, so joins stay clean even when data sources disagree on
team naming.

| Table | Holds |
|---|---|
| `games` | Schedule, scores, status, venue. |
| `probable_starters` | Day-of starters per team. |
| `pitcher_stats`, `bullpen_stats` | Statcast-derived pitching aggregates. |
| `bullpen_daily` | Per-team reliever outs per day (opener-aware). |
| `pitcher_workload` | Per-pitcher outs per day, for live rest-aware bullpens. |
| `team_batting`, `park_factors` | Legacy v1 inputs; retained for archive grading. |
| `odds` | ML, RL, totals from The Odds API. |
| `model_outputs`, `model_outputs_season` | v2 daily and rolling per-team predictions. |
| `model_outputs_v1_archive`, `model_outputs_season_v1_archive` | Frozen v1 history pre-cutover. |
| `model_evaluation`, `model_calibration`, `model_edge_buckets` | Running accuracy across the full season (v1 + v2 stitched). |
| `posterior_skills`, `posterior_sigmas` | Top-N xwOBA leaderboard and per-outcome σ rows, written after each refit. |

Row-level security is on for every table with one policy, `public_read`,
granting SELECT to `anon` and `authenticated`. The site reads through that
policy; this pipeline writes through `DATABASE_URL` as `postgres`.

## Evaluation

`model_evaluation` holds running tallies keyed on `(date, eval_window)`. The
morning `daily-pipeline-v2.yml` and midnight `nightly-eval.yml` runs execute
`backend/evaluate_model.py` and upsert every window. The site also grades a
game the moment it goes final so the dashboard updates within a minute; the
nightly run is the source of truth and reconciles anything the live path
missed.

## Schedule

| Workflow | Cron (UTC) | Purpose |
|---|---|---|
| `train-v2.yml` | `7 11 * * *` (~4 AM PT) | Nightly NUTS refit of all three Bayesian models, then `write_posterior_summaries` populates the diagnostics tables. |
| `daily-pipeline-v2.yml` | `workflow_run` on train-v2 success | Schedule → bullpen → odds → score → verify. Chained off train to guarantee fresh posteriors. |
| `refresh-lineups-v2.yml` | `repository_dispatch` from Supabase pg_cron, every 20 min ~5 AM-7:40 PM PT | Re-scores games whose posted lineup hash changed. |
| `nightly-eval.yml` | `repository_dispatch` at 12:01 AM PT, `37 8 * * *` fallback | Eval yesterday + write tomorrow's predictions. |
GitHub's scheduler drops or delays cron fires often enough that the two
intraday workflows are dispatched from Supabase pg_cron instead; the GitHub
cron on `nightly-eval.yml` is only a fallback.

## Known limits

- **Statcast availability.** Baseball Savant lags by 24-48 hours at the start
  of a season. Unknown actors (call-ups not yet in the training pool) fall
  back to league-mean offsets via a sentinel row in the posterior loader.
- **FanGraphs is unreachable.** All advanced pitching stats are computed
  directly from Statcast pitch data because FanGraphs blocks automated
  requests at the Cloudflare layer.
- **Variance underdispersion.** v2's simulated runs/team-game variance lands
  about 6% low vs actual, even with a calibrated form-noise term and
  ground-ball-stratified out subtypes.
- **Cold-start cost.** First Statcast fetch is ~30 minutes for a prior
  season. After that the cache makes runs cheap.
- **No pipeline failure alerts yet.** A failing GitHub Action surfaces only
  as a red badge in the Actions tab. Email or Slack notification on failure
  is the next operational item.
- **No umpire, travel, or batter-pitcher interaction terms.** Weather enters
  as wind and temperature shifts on the outcome logits; nothing else does.

## Tests

```
pytest tests/
```

Simulator acceptance tests reproduce runner conservation, pregame provenance rejection, and Monte Carlo uncertainty separation.
The former 2025 replay using current posteriors and actual relief appearances is no longer an accuracy gate.
`v2.market_model.acceptance` compares paired frozen forecasts on run MAE, run and margin CRPS, run-line calibration, scoring tails, and win probabilities, including a separate opener segment.
It refuses hindsight training cutoffs, missing provenance, mixed model versions, and insufficient paired evidence.
Legacy forecasts have no recoverable raw probability or input snapshot and are excluded from raw-model research.
No historical accuracy improvement is implied by passing the regression tests.
