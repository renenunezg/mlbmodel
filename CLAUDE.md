# MLB Expected Runs Model

## Project Goal
Predict expected runs per team per MLB game, compare to sportsbook odds, and identify +EV betting opportunities. Output daily predictions to a web dashboard.

## Tech Stack
- **Database**: Supabase (PostgreSQL) — project ID: `zgirspbdvzikzaeqytvf`
- **Data Sources**:
  - MLB Stats API (`MLB-StatsAPI` package) — schedule, scores, probable starters
  - Baseball Savant / Statcast (`pybaseball`) — pitch-level data for computing xFIP, WHIP, K/9, team batting splits
  - Baseball Savant — park factors (static fallback if scraping fails)
  - The Odds API (`the-odds-api.com`) — betting odds
- **Model**: XGBoost regressor (12 features) with negative binomial win probabilities (r=6)
- **Frontend**: Next.js 16 (App Router) + Tailwind CSS + shadcn/ui + Supabase JS client
- **Language**: Python 3.13 (backend), TypeScript (frontend)

## Project Structure
```
pipeline.py             # Daily pipeline: fetch data, update DB, train model, evaluate
verify_pipeline.py      # Post-pipeline sanity checks (14 checks, exits 0/1)
backtest.py             # Walk-forward backtesting on historical seasons
requirements.txt
backend/
  data/                 # Data fetchers
    mlb_api.py          #   Schedule, scores, starters via MLB Stats API
    fangraphs.py        #   Pitcher stats + batting splits computed from Statcast pitch data
    savant.py           #   Park factors via Baseball Savant (static fallback)
    odds_api.py         #   Betting lines via The Odds API
  db.py                 # SQLAlchemy engine (Supabase connection via DATABASE_URL)
  team_mappings.py      # Team name normalization (3-letter codes)
  model.py              # XGBoost model: 12 features, TimeSeriesSplit CV, NB win probs
  evaluate_model.py     # Post-game evaluation metrics
frontend/                 # Next.js app (TypeScript, Tailwind, shadcn/ui)
  src/
    app/
      page.tsx            #   Methodology — model docs, changelog, feature engineering
      games/page.tsx      #   Today's Games — game cards with +EV flags
      history/page.tsx    #   Season History — filtered table with pagination
      performance/page.tsx #  Model Performance — accuracy charts + KPIs
      about/page.tsx      #   About — contact info, research/blog posts
      about/posts.ts      #   Blog post entries (add new posts here)
      layout.tsx          #   Root layout with dark theme + nav
    components/           #   Game cards, badges, filters, charts
    lib/
      supabase.ts         #   Supabase client (reads NEXT_PUBLIC_SUPABASE_*)
      types.ts            #   TypeScript interfaces for DB rows
      utils.ts            #   Formatting helpers
```

## Model Features (12)
| # | Feature | Source | Description |
|---|---------|--------|-------------|
| 1 | `xfip` | Statcast (computed) | Starter xFIP — park-independent pitching quality |
| 2 | `xfip_bullpen` | Statcast (computed) | Bullpen xFIP (IP-weighted team avg) |
| 3 | `starter_whip` | Statcast (computed) | Starter WHIP |
| 4 | `bullpen_k_9` | Statcast (computed) | Bullpen K/9 (IP-weighted) |
| 5 | `batting_ops` | Statcast (computed) | Team OPS — dynamic starter/bullpen handedness blend |
| 6 | `batting_iso` | Statcast (computed) | Team ISO — dynamic starter/bullpen handedness blend |
| 7 | `batting_k_pct` | Statcast (computed) | Team K% — dynamic starter/bullpen handedness blend |
| 8 | `avg_last5` | games table | Rolling 5-game scoring average |
| 9 | `avg_last10` | games table | Rolling 10-game scoring average |
| 10 | `std_last5` | games table | Rolling 5-game scoring std dev (volatility signal) |
| 11 | `park_factor` | Baseball Savant | Venue run-scoring factor |
| 12 | `is_home` | probable_starters | Home field advantage (0/1) |

## Batting Split Blend (features 5-7)
Each batting feature blends the team's splits vs the opposing starter's handedness (known pre-game) and the opposing bullpen's handedness distribution:

```
blended = starter_share * split_vs_starter_hand + (1 - starter_share) * bullpen_blend
bullpen_blend = rhp_ip_share * vs_r + (1 - rhp_ip_share) * vs_l
```

- `starter_share` = opposing starter's trailing avg IP/start / 9, clamped [0.35, 0.78]. Falls back to league mean (~0.578) when fewer than 3 starts available.
- `rhp_ip_share` = opposing team's actual reliever RHP IP share from Statcast. Falls back to 0.6 when unavailable.
- IP is always computed from raw out counts (never baseball's .1/.2 notation).

## +EV Thresholds (`EV_THRESHOLDS` in `backend/model.py`)
```python
EV_THRESHOLDS = {"ml": 0.045, "rl": 0.045, "totals": 0.065}
```
Single source of truth. Totals bar is higher because total-runs markets are noisier.

## Key Conventions
- **Universal game ID**: MLB `game_pk` (integer) — used as foreign key across all tables
- **Team abbreviations**: 3-letter codes (LAD, NYY, etc.) — normalized via `team_mappings.py`
- All numeric stats stored as proper NUMERIC types, never strings
- Environment variables in `.env` — never commit secrets
- FanGraphs is blocked by Cloudflare — all advanced stats computed from Statcast pitch data via `pybaseball`
- Early-season: model falls back to league averages when tables are empty or data is sparse
- Start times stored as UTC ISO timestamps; displayed as PT in the frontend

## Running
```bash
# Install dependencies
pip install -r requirements.txt

# Run full daily pipeline (fetches data, updates scores, trains model, evaluates)
python pipeline.py

# Run lightweight nightly refresh (eval + starters + predict, no Statcast/odds)
python pipeline.py nightly

# Verify pipeline output
python verify_pipeline.py

# Run backtest on historical season
python backtest.py --season 2025

# Run frontend
cd frontend && npm run dev
```

## Database
All tables in Supabase use `game_pk` as the universal join key. Schema managed via Supabase MCP tools.

**Tables:** `games`, `probable_starters`, `pitcher_stats`, `bullpen_stats`, `team_batting`, `park_factors`, `odds`, `model_outputs`, `model_outputs_season`, `model_evaluation`, `model_calibration`, `model_feature_importance`, `model_edge_buckets`, `experiment_runs`

**Recent migrations:**
- `bullpen_stats.rhp_ip_share NUMERIC` — reliever RHP IP share per team
- `pitcher_stats.avg_ip_per_start NUMERIC` — starter avg innings per start

## Pipeline Modes

### Full pipeline (`python pipeline.py`)
1. **Schedule & scores** — Fetch last 3 days + today + tomorrow, upsert games, finalize scores, refresh probable starters
2. **Statcast stats** — Compute pitcher xFIP/WHIP/K9, bullpen stats (incl. `rhp_ip_share`), batting splits from Statcast pitch data
3. **Park factors** — Load from Baseball Savant if not already cached (no-op once populated)
4. **Odds** — Fetch today's lines from The Odds API, match to `game_pk` by team + nearest start time
5. **Model** — Train XGBoost (TimeSeriesSplit CV + GridSearchCV), predict xR, NB win probs, write to `model_outputs` + `model_outputs_season`
6. **Evaluation** — Compare predictions to actual results, write to `model_evaluation`, `model_calibration`, `model_feature_importance`, `model_edge_buckets`

### Nightly refresh (`python pipeline.py nightly`)
1. **Evaluation** — Score yesterday's completed games
2. **Schedule & scores** — Refresh starters for new day
3. **Model** — Retrain on existing DB data, predict new day's games

## GitHub Actions Schedule
| Workflow | Cron | PT time | Purpose |
|---|---|---|---|
| `daily-pipeline.yml` | `0 12 * * *` | ~5 AM PT (primary) | Full pipeline |
| `daily-pipeline-backup.yml` | `30 12 * * *` | ~5:30 AM PT | Runs only if primary did not succeed today (checked via `gh run list`) |
| `nightly-eval.yml` | `0 7 * * *` | midnight PT | Eval + starters + odds + predictions for new day |

GitHub runners typically add 30-60 min delay to scheduled workflows.

## Frontend Pages
| Page | Route | Data source |
|---|---|---|
| Methodology | `/` | Static (methodology-content.tsx) |
| Games | `/games` | `games` + `model_outputs` + live-scores API |
| History | `/history` | `model_outputs_season` + `games` |
| Performance | `/performance` | `model_evaluation`, `model_calibration`, `model_feature_importance` |
| About | `/about` | Static (about/posts.ts for blog entries) |

**Adding a blog post:** edit `frontend/src/app/about/posts.ts`, add an entry at the top of the array, push.

## Tests
```bash
pytest          # 44 tests across kelly, metrics, win_prob, pitcher_split
```
Key test file: `tests/test_pitcher_split.py` — covers blend math, fallback, clamp behavior.

## Known Issues
- **Supabase RLS**: Before public deploy, enable Row Level Security on frontend-facing tables with SELECT-only policies for the `anon` role.
- **First-start pitcher cache**: Prior-season Statcast fetch takes ~30 min on first run. Cached to `cache/` directory after that.
- **Statcast availability**: Baseball Savant data may be delayed 1-2 days at season start. Pipeline handles empty data gracefully.
- **GitHub Actions delay**: Scheduled workflows run 30-60 min late under load. Morning pipeline scheduled at 12:00 UTC typically runs ~6 AM PT.

## Next Steps
- **Production hardening**: Enable Supabase RLS, add error alerting (email/Slack on pipeline failure)
- **Model iteration**: Add features (weather, umpire, rest days), tune hyperparameters based on backtest results, consider ensemble methods
