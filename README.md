# MLB Expected Runs Model

A daily pipeline that forecasts a run distribution for each team in every MLB game, derives win, run-line, and totals probabilities from it, and publishes the results each morning of the season.
The same repository also grades players and prospects and publishes the site's other MLB research pages.

The site is [renenunez.dev](https://renenunez.dev), in its own repository, [momentumweb](https://github.com/renenunezg/momentumweb).
Supabase is the only interface between the two: this repository writes the `mlb` schema and the site reads it.
The site's methodology page explains the model in detail.

## The model

1. Hierarchical Bayesian models estimate batter, pitcher, and park effects on plate-appearance outcomes, partially pooled so thin samples shrink toward the league.
2. A plate-appearance Monte Carlo simulator plays each game many times across posterior draws, with posted lineups, starters, and a rest-aware bullpen.
3. Simulated probabilities are reconciled with the de-vigged betting consensus before publication.
   The market is the calibration benchmark: sharp bettors move lines toward true probabilities quickly.
4. Recommendations follow the versioned pick policy in `backend/strategy.py` and need known starters, posted lineups, and a paired market price.
   Totals recommendations are disabled.
5. A nightly job grades every frozen forecast and writes running accuracy, calibration, and betting results.

Other packages under `v2/` produce player and prospect grades, award boards, playoff odds, live win probability, and descriptive stats such as pitch and batting run value.
Each writes its own tables in the `mlb` schema.

## Repository layout

```
pipeline.py      Schedule and nightly evaluation orchestrator
backend/         Data fetchers, database access, evaluation, odds and staking math, SQL migrations
v2/bayesian/     Skill-layer models and the fitting orchestrator
v2/simulator/    Plate-appearance simulator and bullpen logic
v2/markets/      Market probabilities, pick selection, and the forecast writer
v2/pipeline/     Training, daily scoring, lineup refresh, and verification entry points
v2/live/         In-game win probability and closing-line capture
v2/players/      Player grades and trade values
v2/prospects/    Prospect grades and pitch-level stuff
v2/              Other research and publishing packages, one per site page
tests/           Production-critical tests
```

## Setup

Python 3.13.

```bash
pip install -c constraints.txt -r requirements.txt -r v2/requirements.txt
pip install --no-deps -e .
```

Put these in a root `.env`:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Supabase session pooler URL for the production database |
| `ODDS_API_KEY` | The Odds API key for market prices |
| `MLBMODEL_DB_WRITES` | Set to `1` to allow writes from a local run |

The sampler pins in `v2/requirements.txt` (`numpyro`, `jax`, `jaxlib`) are load-bearing: newer JAX breaks numpyro and sampling can fail silently.
The Statcast cache under `cache/` is gitignored; the first fetch of a season is slow, and later runs read from the cache.

Core commands:

```bash
python -m v2.bayesian.fit_all --save-traces       # refit the skill layer
python -m v2.pipeline.daily_run                   # daily scoring run
python -m v2.pipeline.score_games --date YYYY-MM-DD --no-write
python -m backend.evaluate_model --dry-run        # nightly evaluation without writing
pytest tests/
ruff check .
```

Each entry point takes `--help`.
`pytest` skips the `live_read` and `expensive` markers by default; run them only on purpose.

## Production

GitHub Actions runs production, and every production workflow checks out the `release` branch, never `main`.
Promote a verified commit with `git push origin main:release`.
A release that changes `v2/players/` must first revalidate and republish the player model row, or the scorer refuses the stale row.

| Workflow | Trigger | Job |
|---|---|---|
| `train-v2.yml` | Daily cron in season, manual in winter | Refit the skill layer |
| `daily-pipeline-v2.yml` | After `train-v2.yml` succeeds | Schedule, bullpen, odds, score, verify |
| `refresh-lineups-v2.yml` | Supabase pg_cron dispatch through the day | Re-score games whose lineups changed |
| `nightly-eval.yml` | Supabase pg_cron dispatch after midnight Pacific, with a GitHub cron fallback | Grade yesterday and score tomorrow |
| `live-win-probability.yml` | Supabase pg_cron dispatch around live games | In-game probability and closing lines |
| `players.yml` | Cron, twice a week | Player and prospect grades, contracts, season stats |
| `playoff-forecast.yml` | Manual | Playoff bracket odds |
| `ci.yml` | Push to `main` and pull requests | Tests |

Intraday jobs are dispatched from Supabase pg_cron because GitHub's scheduler drops and delays cron fires.
In the offseason the season workflows and their pg_cron jobs are switched off; `players.yml` keeps running.

`backend/db.py` refuses write statements unless `GITHUB_ACTIONS=true` or `MLBMODEL_DB_WRITES=1`, because `DATABASE_URL` is the live database.
A forecast for a game that has started is frozen: rescoring never overwrites it, and the pick shown on the site is the pick that gets graded.
The site grades games provisionally as they finish; the nightly evaluation is the source of truth.

## Database

All tables live in the `mlb` schema and key games on `game_pk`, the MLB Stats API game ID.
Row-level security grants read-only access to the site; this pipeline writes as `postgres`.
Migrations live in `backend/sql/`, and production schema changes need separate approval.

To rebuild an empty project, install momentumweb's `sql/001_site_revalidate.sql`, restore the schema snapshot in `backend/sql/baseline/`, then apply the migrations dated after that snapshot.
The shared callback needs pg_net, Vault, and the `site_revalidate_secret` matched to the site's `REVALIDATE_SECRET`.
