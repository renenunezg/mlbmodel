BEGIN;

-- Make the first-pitch freeze checkable by an outsider and enforced by the database, not only by the writer.
-- Independent of any code release: the triggers capture every write path, so this can be applied at any time.

-- 1. A graded forecast cannot change once its game has started. v2/markets/writer.py already refuses such
--    writes; this refuses them for every other path too (manual SQL, backfills, future jobs). An unknown
--    first pitch is refused as well, matching the writer, which never publishes a game without one. An
--    intentional correction must disable this trigger explicitly.
CREATE FUNCTION mlb.refuse_started_forecast_write() RETURNS trigger
LANGUAGE plpgsql SET search_path = '' AS $$
DECLARE
  target bigint := CASE WHEN TG_OP = 'DELETE' THEN OLD.game_pk ELSE NEW.game_pk END;
  game record;
BEGIN
  SELECT start_time, status INTO game FROM mlb.games WHERE game_pk = target;
  IF NOT FOUND OR game.start_time IS NULL OR game.start_time <= clock_timestamp()
     OR game.status IN ('Live', 'In Progress', 'Final') THEN
    RAISE EXCEPTION 'Forecast for game % is frozen: first pitch passed or unknown (%)', target, TG_OP;
  END IF;
  -- An UPDATE that moves a row onto another game must also leave the old game's frozen row alone.
  IF TG_OP = 'UPDATE' AND OLD.game_pk IS DISTINCT FROM NEW.game_pk THEN
    SELECT start_time, status INTO game FROM mlb.games WHERE game_pk = OLD.game_pk;
    IF NOT FOUND OR game.start_time IS NULL OR game.start_time <= clock_timestamp()
       OR game.status IN ('Live', 'In Progress', 'Final') THEN
      RAISE EXCEPTION 'Forecast for game % is frozen: first pitch passed or unknown (%)', OLD.game_pk, TG_OP;
    END IF;
  END IF;
  RETURN CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END;
END $$;

CREATE TRIGGER refuse_started_forecast_write
  BEFORE INSERT OR UPDATE OR DELETE ON mlb.model_outputs_season
  FOR EACH ROW EXECUTE FUNCTION mlb.refuse_started_forecast_write();

-- Statement-level guard: TRUNCATE bypasses row triggers.
CREATE FUNCTION mlb.refuse_truncate() RETURNS trigger
LANGUAGE plpgsql SET search_path = '' AS $$
BEGIN
  RAISE EXCEPTION '% is append-only or frozen; TRUNCATE is refused', TG_TABLE_NAME;
END $$;

CREATE TRIGGER refuse_truncate BEFORE TRUNCATE ON mlb.model_outputs_season
  FOR EACH STATEMENT EXECUTE FUNCTION mlb.refuse_truncate();

-- 2. Append-only history of every published forecast. The writer replaces a game's rows on each pregame
--    re-score, so the history keeps each version with its publish time. Compact on purpose (about 200 bytes
--    a row): the graded fields plus an md5 of the full published row, which an outsider can compare with
--    the final row in model_outputs_season.
CREATE TABLE mlb.forecast_history (
  game_pk bigint NOT NULL,
  team text NOT NULL,
  date date NOT NULL,
  published_at timestamptz NOT NULL,
  recorded_at timestamptz NOT NULL DEFAULT clock_timestamp(),
  start_time timestamptz,
  win_prob double precision,
  expected_runs double precision,
  moneyline double precision,
  spread double precision,
  spread_odds double precision,
  total double precision,
  total_over_odds numeric,
  total_under_odds numeric,
  p_cover numeric,
  p_over numeric,
  p_under numeric,
  ev_flag text,
  run_line_ev_flag text,
  total_play text,
  kelly_quarter_ml numeric,
  kelly_quarter_rl numeric,
  kelly_quarter_total numeric,
  lineup_hash text,
  row_md5 text NOT NULL,
  PRIMARY KEY (game_pk, team, published_at)
);
ALTER TABLE mlb.forecast_history ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON mlb.forecast_history FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON mlb.forecast_history TO anon, authenticated;
GRANT SELECT, INSERT ON mlb.forecast_history TO service_role;
COMMENT ON TABLE mlb.forecast_history IS
  'Every forecast row published to model_outputs_season, append-only. published_at is the row''s prediction_updated_at; row_md5 is md5(to_jsonb(row)::text) of the full published row.';

CREATE FUNCTION mlb.record_forecast_history() RETURNS trigger
LANGUAGE plpgsql SET search_path = '' AS $$
BEGIN
  INSERT INTO mlb.forecast_history (
    game_pk, team, date, published_at, start_time, win_prob, expected_runs, moneyline, spread, spread_odds,
    total, total_over_odds, total_under_odds, p_cover, p_over, p_under, ev_flag, run_line_ev_flag, total_play,
    kelly_quarter_ml, kelly_quarter_rl, kelly_quarter_total, lineup_hash, row_md5)
  VALUES (
    NEW.game_pk, NEW.team, NEW.date::date, coalesce(NEW.prediction_updated_at, clock_timestamp()), NEW.start_time,
    NEW.win_prob, NEW.expected_runs, NEW.moneyline, NEW.spread, NEW.spread_odds, NEW.total, NEW.total_over_odds,
    NEW.total_under_odds, NEW.p_cover, NEW.p_over, NEW.p_under, NEW.ev_flag, NEW.run_line_ev_flag, NEW.total_play,
    NEW.kelly_quarter_ml, NEW.kelly_quarter_rl, NEW.kelly_quarter_total, NEW.lineup_hash,
    md5(to_jsonb(NEW)::text));
  RETURN NULL;
END $$;

CREATE TRIGGER record_forecast_history AFTER INSERT ON mlb.model_outputs_season
  FOR EACH ROW EXECUTE FUNCTION mlb.record_forecast_history();

CREATE FUNCTION mlb.refuse_history_change() RETURNS trigger
LANGUAGE plpgsql SET search_path = '' AS $$
BEGIN
  RAISE EXCEPTION '% is append-only (%)', TG_TABLE_NAME, TG_OP;
END $$;

CREATE TRIGGER refuse_history_change BEFORE UPDATE OR DELETE ON mlb.forecast_history
  FOR EACH ROW EXECUTE FUNCTION mlb.refuse_history_change();
CREATE TRIGGER refuse_truncate BEFORE TRUNCATE ON mlb.forecast_history
  FOR EACH STATEMENT EXECUTE FUNCTION mlb.refuse_truncate();

-- 3. Every odds quote with its scrape time. mlb.odds keeps only the latest quote per game, team and book (the
--    pipeline replaces it), so the price a forecast used could not be checked later. About 90 rows a day.
CREATE TABLE mlb.odds_quotes (
  game_pk integer NOT NULL,
  team varchar(3) NOT NULL,
  book varchar(30) NOT NULL,
  scraped_at timestamptz NOT NULL,
  moneyline integer,
  spread numeric(3,1),
  spread_odds integer,
  total numeric(4,1),
  total_over_odds integer,
  total_under_odds integer,
  PRIMARY KEY (game_pk, team, book, scraped_at)
);
ALTER TABLE mlb.odds_quotes ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON mlb.odds_quotes FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON mlb.odds_quotes TO anon, authenticated;
GRANT SELECT, INSERT ON mlb.odds_quotes TO service_role;
COMMENT ON TABLE mlb.odds_quotes IS 'Every odds quote written to mlb.odds, append-only, keyed by its scrape time.';

CREATE FUNCTION mlb.record_odds_quote() RETURNS trigger
LANGUAGE plpgsql SET search_path = '' AS $$
BEGIN
  IF NEW.game_pk IS NULL THEN
    RETURN NULL;
  END IF;
  INSERT INTO mlb.odds_quotes (game_pk, team, book, scraped_at, moneyline, spread, spread_odds, total,
                               total_over_odds, total_under_odds)
  VALUES (NEW.game_pk, NEW.team, NEW.book, coalesce(NEW.scraped_at, clock_timestamp()), NEW.moneyline, NEW.spread,
          NEW.spread_odds, NEW.total, NEW.total_over_odds, NEW.total_under_odds)
  ON CONFLICT DO NOTHING;
  RETURN NULL;
END $$;

CREATE TRIGGER record_odds_quote AFTER INSERT ON mlb.odds
  FOR EACH ROW EXECUTE FUNCTION mlb.record_odds_quote();

CREATE TRIGGER refuse_history_change BEFORE UPDATE OR DELETE ON mlb.odds_quotes
  FOR EACH ROW EXECUTE FUNCTION mlb.refuse_history_change();
CREATE TRIGGER refuse_truncate BEFORE TRUNCATE ON mlb.odds_quotes
  FOR EACH STATEMENT EXECUTE FUNCTION mlb.refuse_truncate();

-- Seed the history with what exists today so the record starts complete. Existing forecast rows enter once
-- with their stored publish time; odds rows with their scrape time.
INSERT INTO mlb.forecast_history (
  game_pk, team, date, published_at, start_time, win_prob, expected_runs, moneyline, spread, spread_odds,
  total, total_over_odds, total_under_odds, p_cover, p_over, p_under, ev_flag, run_line_ev_flag, total_play,
  kelly_quarter_ml, kelly_quarter_rl, kelly_quarter_total, lineup_hash, row_md5)
SELECT game_pk, team, date::date, coalesce(prediction_updated_at, created_at, '-infinity'), start_time, win_prob,
  expected_runs, moneyline, spread, spread_odds, total, total_over_odds, total_under_odds, p_cover, p_over,
  p_under, ev_flag, run_line_ev_flag, total_play, kelly_quarter_ml, kelly_quarter_rl, kelly_quarter_total,
  lineup_hash, md5(to_jsonb(s)::text)
FROM mlb.model_outputs_season s
WHERE game_pk IS NOT NULL AND team IS NOT NULL AND date IS NOT NULL
ON CONFLICT DO NOTHING;

INSERT INTO mlb.odds_quotes (game_pk, team, book, scraped_at, moneyline, spread, spread_odds, total,
                             total_over_odds, total_under_odds)
SELECT game_pk, team, book, coalesce(scraped_at, '-infinity'), moneyline, spread, spread_odds, total,
       total_over_odds, total_under_odds
FROM mlb.odds
WHERE game_pk IS NOT NULL
ON CONFLICT DO NOTHING;

NOTIFY pgrst, 'reload schema';

COMMIT;
