# MLB Expected Runs Model

A daily pipeline that predicts a per-team run distribution for every MLB game,
derives win, run-line, and totals probabilities by Monte Carlo simulation, and
publishes the output to a public dashboard each morning of the season. The
betting markets function as a calibration benchmark, not as a gambling
application: sharp participants push lines toward true probabilities quickly,
which makes them a higher-quality signal than most independently constructed
models.

The site is at [renenunez.dev](https://renenunez.dev). The methodology page
describes the model in detail; this README covers what's in the repository and
how to run it.

## What's in the model

The current model (v2, live since 2026-05-12) is a two-layer system:

1. **Hierarchical Bayesian skill layer.** Three Dirichlet-Multinomial models
   (batter, pitcher, park) over the eight plate-appearance outcomes (K, BB,
   HBP, 1B, 2B, 3B, HR, OUT), fit with NUTS via numpyro/JAX. Batters split by
   platoon (`vs_LHP`), pitchers split by role (`SP`/`RP`), park applies as a
   per-venue log-PF on residual wOBA. Non-centered parameterization, aggregated
   Multinomial likelihood, 4 chains × 2000 draws. R-hat 1.00 and min ESS > 400
   on all three fits. Trained on 401,826 PAs across 2024 + 2025 + 2026-YTD,
   refit nightly (~12 min on M-series, ~30 min on a GitHub Actions runner).
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

v2 replaced a prior XGBoost regressor (v1) after a 542-game head-to-head
backtest: Brier −6.9%, log-loss −7.3%, max calibration gap from 41.9% to 3.2%,
ROI improved on every market. v1 code is preserved at SHA `a84b4dd` in
`v2/evaluation/baseline_v1/`; its predictions before 2026-05-12 are still
served from `model_outputs_v1_archive` and `model_outputs_season_v1_archive`.

## Repository layout

```
pipeline.py             Legacy v1 daily orchestrator. Writes to *_v1_archive.
verify_pipeline.py      v1 sanity checks.
backtest.py             v1 walk-forward backtest.
backend/
  data/                 Fetchers: MLB Stats API, Statcast (pybaseball), Savant,
                        The Odds API, per-pitcher workload from boxscores.
  db.py                 SQLAlchemy engine pointed at Supabase via DATABASE_URL.
  team_mappings.py      3-letter codes + MLB team-id lookup table.
  kelly.py, simulation.py, metrics.py, strategy.py
                        Shared math: Kelly, american/implied/odds conversions,
                        Brier and log-loss, EV thresholds.
v2/
  bayesian/             Three D-M models + fit_all orchestrator. Posteriors
                        saved to v2/bayesian/posteriors/*.nc (gitignored).
  simulator/            posteriors loader, vectorized PA sampler, empirical
                        baserunner table, rest-aware bullpen, game loop.
  markets/              Empirical market probs, EV flags, Kelly, writer to
                        Supabase model_outputs.
  pipeline/             daily_run, train, score_games, refresh_lineups,
                        verify, write_posterior_summaries.
  evaluation/           Frozen v1 baseline + head-to-head backtester.
  data/                 Multi-year Statcast cache builder + per-PA dataset.
frontend/               Next.js 16 app (App Router, TypeScript, Tailwind, shadcn/ui).
  src/app/
    page.tsx                Methodology page (long-form model documentation).
    games/page.tsx          Today's games with +EV flags and live scores.
    history/page.tsx        Per-game prediction log with filters.
    performance/page.tsx    Accuracy charts, calibration, KPIs, posterior
                            leaderboard, variance decomposition.
    about/page.tsx          Contact, blog posts.
    api/live-scores/        Cached MLB Stats API proxy for in-game scores.
    api/eval-game/          Per-game evaluation endpoint, called by the games
                            page when a game finalizes.
  src/components/         Game cards, charts, filters, V2 badge, theme toggle.
  src/lib/
    supabase.ts             Browser client (anon key).
    eval.ts                 TypeScript port of the per-game eval math.
    constants.ts            V2_CUTOVER_DATE (used by chart reference lines).
    types.ts                Database row types.
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

# Head-to-head backtest vs frozen v1
python -m v2.evaluation.replay --start 2026-03-26 --end 2026-05-09 --n-sims 2000 --resume
python -m v2.evaluation.backtester --start 2026-03-26 --end 2026-05-09

# Tests
pytest                       # v1 unit tests (kelly, metrics, win_prob, splits)
pytest v2/                   # v2 tests (Bayesian, simulator, markets, eval)
```

```
# Frontend
cd frontend
npm install
npm run dev                  # http://localhost:3000
npm run build && npm start
```

Pinning matters for the sampler: `numpyro==0.20.1` + `jax==0.7.2` +
`jaxlib==0.7.2`. Newer JAX removed an internal primitive (`xla_pmap_p`) that
numpyro depends on, so sampling fails silently otherwise. The pinned versions
are in `v2/requirements.txt`.

First Statcast fetch is slow (~30 min for a prior season's pitch data).
Subsequent runs read from `cache/` and finish in seconds. The cache is
gitignored and persisted across CI runs via `actions/cache`.

## Environment

Backend, root `.env`:
```
DATABASE_URL=postgresql://...   # Supabase session pooler URL.
ODDS_API_KEY=...                # the-odds-api.com key.
```

Frontend, `frontend/.env.local`:
```
NEXT_PUBLIC_SUPABASE_URL=https://<project>.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=<anon key>
SUPABASE_SERVICE_ROLE_KEY=<service role key>   # server-only; /api/eval-game.
```

The service-role key bypasses RLS, so it must never be exposed in any
`NEXT_PUBLIC_*` variable or imported outside `src/app/api/eval-game/`. In Vercel
it should be set as a regular (non-public) environment variable.

The Supabase project for this repo is `zgirspbdvzikzaeqytvf`. Schema changes
are managed through the Supabase MCP tools, not loose migration files.

## Database

All tables are keyed off `game_pk`, the integer ID assigned by the MLB Stats
API. That makes joins trivial across data sources that otherwise disagree on
team naming conventions.

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
| `experiment_runs` | Hyperparameters and CV scores per training run (legacy v1). |

Row Level Security is enabled on every public table. The only policy is
`public_read`, granting SELECT to the `anon` and `authenticated` roles. The
browser sees only what the anon key plus that policy allows. Writes happen
through `DATABASE_URL` as the `postgres` role (Python pipeline, bypasses RLS)
or via the service-role key from server-only Next.js routes like
`/api/eval-game`.

## Live per-game evaluation

`model_evaluation` rows are running tallies keyed on `(date, eval_window)`.
They are written by two redundant paths:

1. The morning `daily-pipeline-v2.yml` and the midnight `nightly-eval.yml`
   crons run the full Python evaluator and upsert all rows.
2. While the games page is open, `frontend/src/components/games-live.tsx`
   polls the MLB Stats API every sixty seconds. The first time a game's
   status flips to `Final`, the page POSTs to `/api/eval-game`, which writes
   the score back to `games`, recomputes today's window rows, and upserts
   them. The History tab fills in automatically because the W/L badges are
   derived in JSX from `games.status` and the score columns.

The live path makes the dashboard update within a minute of a game ending; the
cron paths are the canonical reconciliation. The math lives in two places
(`backend/evaluate_model.py` and `frontend/src/lib/eval.ts`); a fixture-based
test covers the TypeScript port to catch drift.

## Schedule

| Workflow | Cron (UTC) | Purpose |
|---|---|---|
| `train-v2.yml` | `0 11 * * *` (~4 AM PT) | Nightly NUTS refit of all three Bayesian models, then `write_posterior_summaries` populates the diagnostics tables. |
| `daily-pipeline-v2.yml` | `workflow_run` on train-v2 success | Schedule → bullpen → odds → score → verify. Chained off train to guarantee fresh posteriors. |
| `refresh-lineups-v2.yml` | `0/30 14-23 * * *` (every 30 min, 7 AM-4 PM PT) | Re-scores games whose posted lineup hash changed. |
| `nightly-eval.yml` | `0 7 * * *` (midnight PT) | Eval yesterday + write tomorrow's predictions. |
| `daily-pipeline.yml` (v1) | disabled | Cron removed; `workflow_dispatch` retained for emergencies. v1 writes go to `_v1_archive`. |

GitHub-hosted runners typically add thirty to sixty minutes of queue delay to
scheduled workflows. Real start times therefore drift around the nominal cron.

## Known limits

- **Statcast availability.** Baseball Savant lags by 24-48 hours at the start
  of a season. Unknown actors (call-ups not yet in the training pool) fall
  back to league-mean offsets via a sentinel row in the posterior loader.
- **FanGraphs is unreachable.** All advanced pitching stats are computed
  directly from Statcast pitch data because FanGraphs blocks automated
  requests at the Cloudflare layer.
- **Variance underdispersion.** v2's simulated runs/team-game variance lands
  about 6% low vs actual, even with a calibrated form-noise term. Closing
  that gap is a v2.1 item (out-subtype conditioning on batter/pitcher GB%).
- **Cold-start cost.** First Statcast fetch is ~30 minutes for a prior
  season. After that the cache makes runs cheap.
- **No pipeline failure alerts yet.** A failing GitHub Action surfaces only
  as a red badge in the Actions tab. Email or Slack notification on failure
  is the next operational item.
- **No weather, umpire, travel, or batter-pitcher interaction terms.** Listed
  as deferred features in CLAUDE.md.

## Tests

```
pytest                       # v1 tests: kelly, metrics, win_prob, splits.
pytest v2/                   # v2 tests: Bayesian, simulator, markets, eval.
```

The v2 suite includes a slow acceptance gate
(`v2/tests/test_game_sim.py::test_runs_per_game_within_5pct`) that simulates
200 stratified 2025 games × 990 sims (= 396k team-game samples) and checks
mean and variance against actual. It takes ~2 minutes; the default `pytest v2/` invocation excludes
nothing, so use `--ignore=v2/tests/test_game_sim.py` for quick iteration.
