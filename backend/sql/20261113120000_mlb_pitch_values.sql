BEGIN;

-- Pitch run value (v2.pitch_value): every pitch graded by the change in expected runs it causes, summed per pitcher.
-- Runs are the batting team's, so negative is good for the pitcher. Written by v2.pitch_value.publish.

-- One jsonb row per edition: the seasons covered, the pitch count, and the validation the page quotes (correlation of
-- one season's value with the next season's runs allowed per nine, beside ERA, FIP and xFIP, with intervals).
CREATE TABLE mlb.pitch_value_study (
  season integer PRIMARY KEY,
  model_version text NOT NULL,
  payload jsonb NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE mlb.pitch_value_study ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON mlb.pitch_value_study FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON mlb.pitch_value_study TO anon, authenticated;
GRANT ALL ON mlb.pitch_value_study TO service_role;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.pitch_value_study
  FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();

-- One row per pitcher-season with enough pitches.
-- raw_runs: the runs as they happened. context_runs: expected contact with park, hitter and season removed, each pitch
-- valued in the bases and outs it was thrown in. runs: the same with each pitch valued at its average over bases and
-- outs (the headline). The site derives runs per 100 pitches.
CREATE TABLE mlb.pitch_values (
  player_id bigint NOT NULL,
  season smallint NOT NULL,
  player_name text NOT NULL,
  team text NOT NULL,
  pitches smallint NOT NULL,
  raw_runs real NOT NULL,
  context_runs real NOT NULL,
  runs real NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (player_id, season)
);
ALTER TABLE mlb.pitch_values ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON mlb.pitch_values FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON mlb.pitch_values TO anon, authenticated;
GRANT ALL ON mlb.pitch_values TO service_role;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.pitch_values
  FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();

-- One row per pitcher-season and pitch type with enough pitches of that type, for pitchers in pitch_values.
-- usage: the type's share of his pitches that season.
CREATE TABLE mlb.pitch_type_values (
  player_id bigint NOT NULL,
  season smallint NOT NULL,
  pitch_type text NOT NULL,
  pitches smallint NOT NULL,
  usage real NOT NULL,
  runs real NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (player_id, season, pitch_type),
  FOREIGN KEY (player_id, season) REFERENCES mlb.pitch_values (player_id, season) ON DELETE CASCADE
);
ALTER TABLE mlb.pitch_type_values ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON mlb.pitch_type_values FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON mlb.pitch_type_values TO anon, authenticated;
GRANT ALL ON mlb.pitch_type_values TO service_role;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.pitch_type_values
  FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();

COMMIT;
