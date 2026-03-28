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
- **Model**: XGBoost regressor (11 features) with Poisson-based win probabilities
- **Frontend**: Streamlit (to be replaced with Next.js + Supabase REST API)
- **Language**: Python 3.13

## Project Structure
```
pipeline.py             # Daily pipeline: fetch data, update DB, train model, evaluate
requirements.txt
backend/
  data/                 # Data fetchers
    mlb_api.py          #   Schedule, scores, starters via MLB Stats API
    fangraphs.py        #   Pitcher stats + batting splits computed from Statcast pitch data
    savant.py           #   Park factors via Baseball Savant (static fallback)
    odds_api.py         #   Betting lines via The Odds API
  db.py                 # SQLAlchemy engine (Supabase connection via DATABASE_URL)
  team_mappings.py      # Team name normalization (3-letter codes)
  model.py              # XGBoost model: 11 features, TimeSeriesSplit CV, Poisson win probs
  evaluate_model.py     # Post-game evaluation metrics
frontend/
  app.py                # Streamlit dashboard (to be replaced)
```

## Model Features (11)
| # | Feature | Source | Description |
|---|---------|--------|-------------|
| 1 | `xfip` | Statcast (computed) | Starter xFIP — park-independent pitching quality |
| 2 | `xfip_bullpen` | Statcast (computed) | Bullpen xFIP (IP-weighted team avg) |
| 3 | `starter_whip` | Statcast (computed) | Starter WHIP |
| 4 | `bullpen_k_9` | Statcast (computed) | Bullpen K/9 (IP-weighted) |
| 5 | `batting_ops` | Statcast (computed) | Team OPS vs opponent pitcher handedness |
| 6 | `batting_iso` | Statcast (computed) | Team ISO vs opponent pitcher handedness |
| 7 | `batting_k_pct` | Statcast (computed) | Team K% vs opponent pitcher handedness |
| 8 | `avg_last5` | games table | Rolling 5-game scoring average |
| 9 | `avg_last10` | games table | Rolling 10-game scoring average |
| 10 | `park_factor` | Baseball Savant | Venue run-scoring factor |
| 11 | `is_home` | probable_starters | Home field advantage (0/1) |

## Key Conventions
- **Universal game ID**: MLB `game_pk` (integer) — used as foreign key across all tables
- **Team abbreviations**: 3-letter codes (LAD, NYY, etc.) — normalized via `team_mappings.py`
- All numeric stats stored as proper NUMERIC types, never strings
- Environment variables in `.env` — never commit secrets
- FanGraphs is blocked by Cloudflare — all advanced stats computed from Statcast pitch data via `pybaseball`
- Early-season: model falls back to league averages when tables are empty or data is sparse

## Running
```bash
# Install dependencies
pip install -r requirements.txt

# Run daily pipeline (fetches data, updates scores, trains model, evaluates)
python pipeline.py

# Run frontend
streamlit run frontend/app.py
```

## Database
All tables in Supabase use `game_pk` as the universal join key. Schema managed via Supabase MCP tools.

**Tables:** `games`, `probable_starters`, `pitcher_stats`, `bullpen_stats`, `team_batting`, `park_factors`, `odds`, `model_outputs`, `model_outputs_season`, `model_evaluation`

## Daily Pipeline Flow (`pipeline.py`)
1. **Schedule & scores** — Fetch last 3 days + today + tomorrow schedules, upsert games, finalize scores, refresh probable starters
2. **Statcast stats** — Compute pitcher xFIP/WHIP/K9, bullpen stats, and team batting splits from Statcast pitch data (single cached fetch)
3. **Park factors** — Load from Baseball Savant if not already cached in DB
4. **Odds** — Fetch today's lines from The Odds API, match to `game_pk` by team+date, upsert
5. **Model** — Train XGBoost (TimeSeriesSplit CV), predict expected runs, Poisson win probs, write to `model_outputs` + `model_outputs_season`
6. **Evaluation** — Compare predictions to actual results, write accuracy metrics to `model_evaluation`

## Progress (completed)
- **Phase 1 — Data fetching**: Replaced 9 fragile scrapers with 4 clean API/fetcher modules. Selenium eliminated.
- **Phase 2 — Database**: Supabase schema with 10 tables, all joined on `game_pk`.
- **Phase 3 — Model improvements**: Expanded to 11 features, TimeSeriesSplit CV with GridSearchCV, Poisson-based win probabilities, early-season fallbacks. Replaced FanGraphs scraping with Statcast-computed stats (FanGraphs blocked by Cloudflare).
- **Phase 4 — Pipeline**: Consolidated 9-step daily_runner.py into 6-step pipeline.py with timing, batch DB ops, and proper error handling.

## Known Issues
- **Doubleheader odds matching**: The Odds API doesn't return `game_pk`, so odds are matched by team name. For doubleheaders, this can't distinguish Game 1 vs Game 2 — needs start time matching or Odds API event ID correlation.

## Next Steps
- **Phase 5 — Frontend**: Migrate from Streamlit to Next.js + Supabase REST API
- **Phase 6 — Validation**: Backtest with 2024-2025 data, dry run, monitor opening week
