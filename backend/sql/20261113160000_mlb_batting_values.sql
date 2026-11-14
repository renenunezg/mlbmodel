BEGIN;

-- Batting run value (v2.batting_value): every pitch a hitter sees graded by the change in expected runs, with the
-- difficulty of the pitch taken out and the rest split into the decision to swing or take, the contact made and
-- the quality of that contact. Runs are the batting team's, so positive is good for the hitter.
-- Written by v2.batting_value.publish.

-- One jsonb row per edition: the seasons covered, the pitch count, and what the page quotes (correlation of one
-- season's value with the next season's batting runs beside wOBA, xWOBA and the hitter grade, how well each part
-- and each split repeats, with intervals).
CREATE TABLE mlb.batting_value_study (
  season integer PRIMARY KEY,
  model_version text NOT NULL,
  payload jsonb NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE mlb.batting_value_study ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON mlb.batting_value_study FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON mlb.batting_value_study TO anon, authenticated;
GRANT ALL ON mlb.batting_value_study TO service_role;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.batting_value_study
  FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();

-- One row per hitter-season with enough plate appearances.
-- decision_runs, contact_runs, quality_runs: the three credited parts; the headline total is their sum.
-- call_runs: called strikes and hit batters on his takes against what the league model expected (not credited).
-- faced_runs: the league-average value of the pitches he saw (not credited; positive for hitters pitched around).
-- raw_runs: the runs as they happened, in the bases and outs they happened in.
-- The site derives the total, runs per 600 plate appearances and runs per 100 pitches.
CREATE TABLE mlb.batting_values (
  player_id bigint NOT NULL,
  season smallint NOT NULL,
  player_name text NOT NULL,
  team text NOT NULL,
  pa smallint NOT NULL,
  pitches smallint NOT NULL,
  decision_runs real NOT NULL,
  contact_runs real NOT NULL,
  quality_runs real NOT NULL,
  call_runs real NOT NULL,
  faced_runs real NOT NULL,
  raw_runs real NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (player_id, season)
);
ALTER TABLE mlb.batting_values ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON mlb.batting_values FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON mlb.batting_values TO anon, authenticated;
GRANT ALL ON mlb.batting_values TO service_role;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.batting_values
  FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();

-- One row per hitter-season and split, for hitters in batting_values: the same three parts on part of his pitches.
-- split is one code from one of these families. The first four never mix; the directions were added after the table
-- was created, and a pitch can be in one side and one height at once (up and in):
--   count situations: two_strikes, ahead (more balls than strikes, under two strikes), first_pitch;
--   pitch groups: fastball, breaking, offspeed;
--   pitch types (Statcast codes, with enough pitches of the type): FF, SI, SL, CH;
--   attack zones: heart, shadow, chase, waste;
--   sides (outer thirds of the plate, from the hitter's side of it): inside, away;
--   heights (outer thirds of his strike zone): up, down.
CREATE TABLE mlb.batting_split_values (
  player_id bigint NOT NULL,
  season smallint NOT NULL,
  split text NOT NULL,
  pitches smallint NOT NULL,
  decision_runs real NOT NULL,
  contact_runs real NOT NULL,
  quality_runs real NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (player_id, season, split),
  FOREIGN KEY (player_id, season) REFERENCES mlb.batting_values (player_id, season) ON DELETE CASCADE
);
ALTER TABLE mlb.batting_split_values ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON mlb.batting_split_values FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON mlb.batting_split_values TO anon, authenticated;
GRANT ALL ON mlb.batting_split_values TO service_role;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.batting_split_values
  FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();

COMMIT;
