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

1. **Hierarchical Bayesian skill layer.** Batter and pitcher skill are
   hierarchical multinomial-logit models over the eight plate-appearance
   outcomes (K, BB, HBP, 1B, 2B, 3B, HR, OUT): each actor carries a vector of
   seven additive log-odds offsets against OUT as the reference category,
   partially pooled through a non-centered Normal hierarchy, so an actor's
   outcome probabilities are logistic-normal rather than Dirichlet. Batters
   split by platoon (`vs_LHP`); pitchers shrink toward role-specific spreads
   (`SP`/`RP`). Park is a separate Gaussian model of a per-venue log park
   factor fit to residual wOBA after batter and pitcher effects. All three are
   fit with NUTS via numpyro/JAX on aggregated per-actor outcome counts
   (Multinomial likelihood), 4 chains × 2000 draws. R-hat 1.00 and min ESS >
   400 on all three fits. Trained on 401,826 PAs across 2024 + 2025 +
   2026-YTD, refit nightly (~12 min on M-series, ~30 min on a GitHub Actions
   runner).
2. **Per-PA Monte Carlo simulator.** K=30 random posterior draws × N
   inning-level simulations per draw. N is configurable via `--n-sims`; the
   production scoring default is 10,000 total sims (~333 per draw) and the
   acceptance-gate test runs 990 (33 per draw). The K-draw outer loop
   propagates parameter uncertainty; the inner loop samples PAs vectorized in
   NumPy. Baserunner advancement uses an empirical
   `P(new_state, runs, outs_added | state, outs, outcome, subtype)` table
   built from 365k PAs of Statcast data, with linear shrinkage toward the
   outcome-conditional marginal on cells with fewer than 100 observations and
   deterministic forced advances for HR/BB/HBP. Bullpens are rest-aware:
   relievers with ≥ 6 outs in the last 1 day or ≥ 9 outs in the last 2 days
   are skipped.

Win, total, and run-line probabilities are computed empirically from the
simulated run distributions per matchup. The p10/p90 win-probability band is
taken across the K posterior draws (parameter uncertainty), not across the
inner sims (run-scoring noise). A play is flagged when modeled probability
exceeds the sportsbook's de-vigged implied probability by more than 4.5%
(ML/RL) or 6.5% (totals); sizing is quarter-Kelly.

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
  tools/                One-off calibration scripts (form sigma, weather coefficients).
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
python -m v2.pipeline.score_games --date 2026-05-14 --n-sims 10000

# Intraday lineup refresh (re-scores games whose posted lineup changed)
python -m v2.pipeline.refresh_lineups

# Market-relative research
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

The suite includes a slow acceptance gate
(`tests/test_simulator_acceptance.py::test_runs_per_game_within_5pct`) that simulates
200 stratified 2025 games × 990 sims (= 396k team-game samples) and checks
mean and variance against actuals. It takes ~2 min. `pytest tests/` runs it by
default; pass `--ignore=tests/test_simulator_acceptance.py` for quick iteration.
