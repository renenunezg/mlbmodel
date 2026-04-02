# MLB Expected Runs Model

Predict expected runs per team per MLB game, compare to sportsbook odds, and identify +EV betting opportunities. Outputs daily predictions to a web dashboard.

## Setup

### Prerequisites
- Python 3.13+
- Node.js 20+
- Supabase project (for database)
- API keys: The Odds API

### Environment Variables

**Root `.env`** (for Python pipeline):
```
DATABASE_URL=postgresql://...
ODDS_API_KEY=your_key
```

**`frontend/.env.local`** (for Next.js frontend):
```
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key
```

## Running

### Backend Pipeline (Python)

```bash
# Install dependencies
pip install -r requirements.txt

# Run daily pipeline (fetches data, trains model, generates predictions)
python pipeline.py

# Verify pipeline output (10 sanity checks)
python verify_pipeline.py

# Run walk-forward backtest on a historical season
python backtest.py --season 2025
```

### Frontend (Next.js)

```bash
# Install dependencies
cd frontend
npm install

# Run dev server (http://localhost:3000)
npm run dev

# Production build + start
npm run build
npm start
```

## Tech Stack

- **Database**: Supabase (PostgreSQL)
- **Data Sources**: MLB Stats API, Baseball Savant / Statcast, The Odds API
- **Model**: XGBoost regressor (11 features) with Poisson-based win probabilities
- **Frontend**: Next.js 16 (App Router) + Tailwind CSS 4 + shadcn/ui
- **Languages**: Python 3.13 (backend), TypeScript (frontend)
