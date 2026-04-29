# MLB Expected Runs Model

A daily pipeline that predicts expected runs per team per MLB game, compares the
resulting win and total probabilities to sportsbook lines, and publishes the
output to a public dashboard each morning of the season. The betting markets
function here as a calibration benchmark, not as a gambling application: sharp
participants push lines toward true probabilities quickly, which makes them a
higher-quality signal than most independently constructed models.

The site is at [renenunez.dev](https://renenunez.dev). The methodology page
describes the model in detail; this README covers what's in the repository and
how to run it.

## What's in the model

Twelve features per team per game, all derived from pitch-level Statcast data
plus a handful of contextual fields. Pitching covers starter and bullpen quality
(xFIP, WHIP, K/9, IP-weighted across relievers). Batting uses a dynamic blend of
splits versus the opposing starter's hand and the opposing bullpen's actual
right-handed IP share, weighted by the starter's trailing average innings per
start. Rolling fields cover the team's last five and ten games, including
volatility. Two contextual features cover park factor and home-field advantage.

Predictions come from an XGBoost regressor trained with `TimeSeriesSplit`
cross-validation and `GridSearchCV`, retrained daily on all completed games of
the season. Win probabilities are computed from a negative binomial distribution
with dispersion `r = 6`, fit to historical run distributions. Earlier versions
used Poisson, which assumes variance equals the mean; that assumption breaks for
MLB run scoring, so the model was systematically overconfident in the tails.

Predictions are graded against sportsbook lines on three markets (moneyline, run
line, totals) using a quarter-Kelly sizing rule. A play is flagged `+EV` when
the model's probability of an outcome exceeds the book's implied probability by
more than 4.5% on moneylines and run lines, or 6.5% on totals (totals carry a
stricter bar because the market is noisier).

## Repository layout

```
pipeline.py             Daily orchestration entry point. `nightly` mode is lighter.
verify_pipeline.py      14 sanity checks. Exits non-zero on any failure.
backtest.py             Walk-forward backtest on historical seasons.
backend/
  data/                 Fetchers: MLB Stats API, Statcast (pybaseball),
                        Baseball Savant, The Odds API.
  db.py                 SQLAlchemy engine pointed at Supabase via DATABASE_URL.
  team_mappings.py      3-letter code normalization across data sources.
  model.py              XGBoost training + negative binomial win probabilities.
  evaluate_model.py     Post-game grading: regression metrics, calibration,
                        equity, edge buckets.
frontend/               Next.js 16 app (App Router, TypeScript, Tailwind, shadcn/ui).
  src/app/
    page.tsx                Methodology page (the long-form model documentation).
    games/page.tsx          Today's games with +EV flags and live scores.
    history/page.tsx        Per-game prediction log with filters.
    performance/page.tsx    Accuracy charts, calibration, KPIs.
    about/page.tsx          Contact, blog posts.
    api/live-scores/        Cached MLB Stats API proxy for in-game scores.
    api/eval-game/          Per-game evaluation endpoint, called by the games
                            page when a game finalizes.
  src/components/         Game cards, charts, filters, theme toggle.
  src/lib/
    supabase.ts             Browser client (anon key).
    eval.ts                 TypeScript port of the per-game eval math.
    types.ts                Database row types.
```

## Running locally

```
# Backend
pip install -r requirements.txt

python pipeline.py            # Full daily run: ingest, train, predict, grade.
python pipeline.py nightly    # Light run: grade yesterday + predict tomorrow.
python verify_pipeline.py     # 14 post-pipeline checks.
python backtest.py --season 2025

pytest                        # 44 tests across kelly, metrics, win_prob, splits.
```

```
# Frontend
cd frontend
npm install
npm run dev                   # http://localhost:3000
npm run build && npm start
```

The first full pipeline run is slow. Statcast fetches the previous season's
pitch data on cold start to compute prior-year pitcher baselines, which takes
roughly thirty minutes. Subsequent runs read from `cache/` and finish in two to
three minutes. The cache is gitignored and is also persisted across CI runs via
the `actions/cache` GitHub Action.

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
SUPABASE_SERVICE_ROLE_KEY=<service role key>   # server-only; used by /api/eval-game.
```

The service-role key bypasses RLS, so it must never be exposed in any
`NEXT_PUBLIC_*` variable or imported outside `src/app/api/eval-game/`. In Vercel
it should be set as a regular (non-public) environment variable.

The Supabase project for this repo is `zgirspbdvzikzaeqytvf`. Schema changes are
managed through the Supabase MCP tools, not loose migration files.

## Database

All tables are keyed off `game_pk`, the integer ID assigned by the MLB Stats
API. That makes joins trivial across data sources that otherwise disagree on
team naming conventions.

| Table | Holds |
|---|---|
| `games` | Schedule, scores, status, venue. |
| `probable_starters` | Day-of starters per team. |
| `pitcher_stats`, `bullpen_stats` | Statcast-derived pitching features. |
| `team_batting` | Pre-blended batting splits per team. |
| `park_factors` | Run-scoring environment per venue. |
| `odds` | Moneyline, run line, totals from The Odds API. |
| `model_outputs`, `model_outputs_season` | Daily and rolling per-team predictions. |
| `model_evaluation` | Running accuracy and regression metrics per `(date, eval_window)`. |
| `model_calibration` | Decile probability bins versus observed rates. |
| `model_feature_importance` | XGBoost gain importances per training run. |
| `model_edge_buckets` | Hit rate and ROI bucketed by edge size. |
| `experiment_runs` | Hyperparameters, CV scores, feature list per training run. |

Row Level Security is enabled on every public table. The only policy is
`public_read`, granting SELECT to the `anon` and `authenticated` roles. The
browser sees only what the anon key plus that policy allows, which is reads.
Writes happen one of two ways: the Python pipeline connects through
`DATABASE_URL` as the `postgres` role and bypasses RLS, and the Next.js
`/api/eval-game` endpoint uses the service-role key (server-side only) to
write the live game state and evaluation rows.

## Live per-game evaluation

`model_evaluation` rows are running tallies keyed on `(date, eval_window)`. They
are written by two paths:

1. The morning `daily-pipeline.yml` cron and the midnight `nightly-eval.yml`
   cron run the full Python evaluator and upsert all rows.
2. While the games page is open, `frontend/src/components/games-live.tsx` polls
   the MLB Stats API every sixty seconds. The first time a game's status flips
   to `Final`, the page POSTs to `/api/eval-game`, which writes the score back
   to `games`, recomputes today's four window rows in `model_evaluation`, and
   upserts them. The History tab fills in automatically because the W/L badges
   are derived in JSX from `games.status` and the score columns.

The two paths are deliberately redundant. The live path makes the dashboard
update within a minute of a game ending. The cron paths are the canonical
reconciliation: anything the live path got wrong gets overwritten the next
morning. The math lives in two places (`backend/evaluate_model.py` and
`frontend/src/lib/eval.ts`); a fixture-based test covers the TypeScript port
to catch drift.

## Schedule

| Workflow | Cron (UTC) | Purpose |
|---|---|---|
| `daily-pipeline.yml` | `0 12 * * *` (~5 AM PT) | Full pipeline. |
| `daily-pipeline-backup.yml` | `30 12 * * *` (~5:30 AM PT) | Reruns full pipeline only if the primary did not complete. |
| `nightly-eval.yml` | `0 7 * * *` (midnight PT) | Eval, starters refresh, predictions for the new day. |

GitHub-hosted runners typically add thirty to sixty minutes of queue delay to
scheduled workflows. Real start times therefore drift around the nominal cron.

## Known limits

- **Statcast availability.** Baseball Savant lags by 24-48 hours at the start
  of a season. The pipeline falls back to league averages when tables are sparse.
- **FanGraphs is unreachable.** All advanced pitching stats are computed
  directly from Statcast pitch data because FanGraphs blocks automated
  requests at the Cloudflare layer. This was the original data source and
  removing it forced the team-batting split logic to be rebuilt from scratch
  against `pybaseball`. The current setup is more dependency-light as a
  consequence.
- **Cold-start cost.** First Statcast fetch is ~30 minutes for the prior
  season. After that the cache makes runs cheap.
- **No pipeline failure alerts yet.** A failing GitHub Action surfaces only as a red badge in the Actions tab. Email or Slack notification on failure is the next operational item.
- **Single-frame predictions.** The model has no awareness of weather, umpire,
  rest days, or travel. Adding these is on the list.

## Tests

```
pytest
```

Forty-four tests across `tests/test_kelly.py`, `tests/test_metrics.py`,
`tests/test_win_prob.py`, and `tests/test_pitcher_split.py`. The split tests are
the most useful; they cover the dynamic starter/bullpen blend, the IP clamp, and
the early-season fallback behavior.
