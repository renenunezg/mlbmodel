BEGIN;

-- Closing lines: the last pregame market quote held when each game started, one row per game, written once.
-- This is the shared Momentum contract (the cfb, nfl and nhl schemas carry the same table shape); the live
-- worker (v2/live/closing.py) writes it at the first poll where a game is live or past its scheduled start.
-- Sources: 'odds_api:median' (one The Odds API request per first-pitch cluster inside a credit budget; the run
-- line and total are the median point across bookmakers, prices from DraftKings at that point or else the most
-- recently updated bookmaker, observed_at that bookmaker's last_update) and 'espn:draftkings' (ESPN's free
-- scoreboard, observed_at = fetched_at) when the request was skipped. Nothing here is written to mlb.odds, so
-- odds_quotes stays the history of the morning pull alone. Evaluation reads the moneylines; the rest is kept so
-- a run-line or totals benchmark needs no new capture.
CREATE TABLE mlb.closing_lines (
  game_pk integer PRIMARY KEY CHECK (game_pk > 0),
  season integer NOT NULL,
  start_time timestamptz NOT NULL,
  source text NOT NULL,
  observed_at timestamptz NOT NULL,
  fetched_at timestamptz NOT NULL,
  home_spread double precision,
  spread_home_price double precision,
  spread_away_price double precision,
  total double precision,
  over_price double precision,
  under_price double precision,
  home_moneyline double precision,
  away_moneyline double precision,
  n_offers integer CHECK (n_offers >= 0),
  price_book text,
  -- A moneyline counts only when both sides were quoted together.
  CHECK ((home_moneyline IS NULL) = (away_moneyline IS NULL))
);
ALTER TABLE mlb.closing_lines ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON mlb.closing_lines FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON mlb.closing_lines TO anon, authenticated;
GRANT SELECT, INSERT ON mlb.closing_lines TO service_role;
COMMENT ON TABLE mlb.closing_lines IS
  'One closing quote per game, written once by the live worker when the game starts. home_spread is the home run line (negative when home is favored); prices are American. source names the feed; n_offers and price_book describe the Odds API median row.';
-- No site_revalidate trigger: a closing line is not shown on a page, and a revalidation per first pitch would
-- rebuild the site about fifteen times a night for nothing.

-- Write-once, pregame only, and on time: a row can only be inserted, its quote must predate the scheduled first
-- pitch, and the insert must arrive within ten minutes of that first pitch. A late worker therefore records no
-- close rather than a stale one. The row-level trigger runs before ON CONFLICT is checked, so a duplicate insert
-- after the ten minutes is refused too; the worker only attempts a write inside that window.
CREATE FUNCTION mlb.refuse_closing_line_change() RETURNS trigger
LANGUAGE plpgsql SET search_path = '' AS $$
BEGIN
  IF TG_OP <> 'INSERT' THEN
    RAISE EXCEPTION 'closing_lines is write-once (%)', TG_OP;
  END IF;
  IF NEW.fetched_at >= NEW.start_time THEN
    RAISE EXCEPTION 'Closing line for game % was fetched at %, not before its start %',
      NEW.game_pk, NEW.fetched_at, NEW.start_time;
  END IF;
  IF clock_timestamp() > NEW.start_time + interval '10 minutes' THEN
    RAISE EXCEPTION 'Closing line for game % arrived more than 10 minutes after its start %',
      NEW.game_pk, NEW.start_time;
  END IF;
  RETURN NEW;
END $$;

CREATE TRIGGER refuse_closing_line_change BEFORE INSERT OR UPDATE OR DELETE ON mlb.closing_lines
  FOR EACH ROW EXECUTE FUNCTION mlb.refuse_closing_line_change();
CREATE TRIGGER refuse_truncate BEFORE TRUNCATE ON mlb.closing_lines
  FOR EACH STATEMENT EXECUTE FUNCTION mlb.refuse_truncate();

-- The closing benchmark beside the morning one. paired_games and market_brier keep their definition (the frozen
-- forecast's own stored prices); the closing columns score the same forecasts against closing_lines on the games
-- that have both, and closing_games_by_source counts those games per feed so the two feeds are never pooled
-- silently.
ALTER TABLE mlb.model_evaluation
  ADD COLUMN closing_paired_games integer CHECK (closing_paired_games >= 0),
  ADD COLUMN closing_paired_brier numeric,
  ADD COLUMN closing_paired_log_loss numeric,
  ADD COLUMN closing_market_brier numeric,
  ADD COLUMN closing_market_log_loss numeric,
  ADD COLUMN brier_vs_closing_ci95_low numeric,
  ADD COLUMN brier_vs_closing_ci95_high numeric,
  ADD COLUMN closing_games_by_source jsonb;

NOTIFY pgrst, 'reload schema';

COMMIT;
