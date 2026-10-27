BEGIN;

-- One jsonb row per season edition of the carry study: the aggregates behind each section of the site's carry page
-- (puzzle balls, repeatability, air and park effects, spin curves, collision summary, explained shares, swing types,
-- swing changers, lab coefficients and the generic hitter's lab grid). Rebuilt each offseason by v2.carry.build.
CREATE TABLE mlb.carry_study (
  season integer PRIMARY KEY,
  model_version text NOT NULL,
  payload jsonb NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE mlb.carry_study ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON mlb.carry_study FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON mlb.carry_study TO anon, authenticated;
GRANT ALL ON mlb.carry_study TO service_role;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.carry_study
  FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();

-- One row per hitter-season with enough balls in the air. Counting stats are raw so the site derives rates.
-- carry: feet beyond league distance at the same exit velocity, launch angle, spray, park-season and month.
-- carry_raw: the same against exit velocity and launch angle only. physics: the part of carry the swing physics explains.
-- backspin: rpm relative to the league at the same launch angle (relative only; absolute rpm is not identified).
-- tilt and attack: swing path tilt and attack angle at a belt-high pitch (2.5 ft). lab_carry: tenths of a foot of carry
-- change on a 7 x 7 grid of tilt change (outer) by attack-angle change (inner), each -6..6 degrees in steps of 2;
-- only for the study season's hitters with tracked swings.
CREATE TABLE mlb.carry_hitters (
  player_id bigint NOT NULL,
  season smallint NOT NULL,
  player_name text NOT NULL,
  team text,
  bats text,
  pa smallint NOT NULL,
  ab smallint NOT NULL,
  h smallint NOT NULL,
  tb smallint NOT NULL,
  hr smallint NOT NULL,
  bb smallint NOT NULL,
  hbp smallint NOT NULL,
  sf smallint NOT NULL,
  so smallint NOT NULL,
  swings smallint,
  whiffs smallint,
  n_air smallint NOT NULL,
  carry real NOT NULL,
  carry_raw real NOT NULL,
  physics real,
  backspin smallint,
  sidespin smallint,
  tilt real,
  attack real,
  bat_speed real,
  lab_carry smallint[],
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (player_id, season)
);
ALTER TABLE mlb.carry_hitters ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON mlb.carry_hitters FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON mlb.carry_hitters TO anon, authenticated;
GRANT ALL ON mlb.carry_hitters TO service_role;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.carry_hitters
  FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();

-- Triple-A ('aaa') and Florida State League ('fsl') hitter-seasons, each measured against its own league and ballparks.
-- No bat tracking exists at these levels, so there are no swing columns.
CREATE TABLE mlb.carry_minors (
  player_id bigint NOT NULL,
  season smallint NOT NULL,
  level text NOT NULL CHECK (level IN ('aaa', 'fsl')),
  player_name text NOT NULL,
  team text,
  pa smallint NOT NULL,
  hr smallint NOT NULL,
  n_air smallint NOT NULL,
  carry real NOT NULL,
  carry_raw real NOT NULL,
  exit_velo real NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (player_id, season, level)
);
ALTER TABLE mlb.carry_minors ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON mlb.carry_minors FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON mlb.carry_minors TO anon, authenticated;
GRANT ALL ON mlb.carry_minors TO service_role;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.carry_minors
  FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();

COMMIT;
