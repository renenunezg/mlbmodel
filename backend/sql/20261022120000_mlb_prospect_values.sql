BEGIN;

-- The fitted prospect model, one jsonb row per fit (boosters gzipped and base64). Scoring in CI loads the newest.
CREATE TABLE mlb.prospect_models (
  model_version text NOT NULL,
  last_dev_season integer NOT NULL,
  payload jsonb NOT NULL,
  fitted_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (model_version, last_dev_season)
);
ALTER TABLE mlb.prospect_models ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON mlb.prospect_models FROM PUBLIC, anon, authenticated;
GRANT ALL ON mlb.prospect_models TO service_role;

-- One row per rookie-eligible player in affiliated ball: peak grade, probability of a regular, trade value, ETA,
-- trait grades and percentiles as arrays in the pool's fixed order (hitters: contact, discipline, power, batted_ball, speed,
-- defense, age_level, track_record, pedigree; pitchers: bat_missing, control, ground_balls, contact_quality, stuff, role,
-- age_level, track_record, pedigree; null where untracked), and the tip: the four inputs that moved the grade most, as
-- [input, contribution] pairs. The control-years line is derived on the site from expected_control_war, p_mlb and eta_season;
-- the arrival-year band is the model row's eta_band per pool. Grade bands widen with a short window (10 * sqrt(floor / (n + floor))).
CREATE TABLE mlb.prospect_values (
  player_id bigint PRIMARY KEY,
  player_name text,
  org text,
  org_id integer,
  team text,
  team_id integer,
  level text,
  position_group text NOT NULL,
  age numeric,
  bats text,
  throws text,
  headshot_url text,
  as_of date NOT NULL,
  model_version text NOT NULL,
  season integer NOT NULL,
  draft_year integer,
  draft_round text,
  draft_pick integer,
  signing_bonus_musd numeric,
  acquired text,
  peak_grade numeric,
  peak_grade_low numeric,
  peak_grade_high numeric,
  peak_grade_percentile numeric,
  grade_reason text,
  p_regular numeric,
  p_mlb numeric,
  eta_season integer,
  trade_value_musd numeric,
  trade_value_low numeric,
  trade_value_high numeric,
  market_value_musd numeric,
  expected_control_war numeric,
  peak_war numeric,
  risk text,
  on_40_man boolean NOT NULL DEFAULT false,
  mlb_ab_or_ip numeric,
  service_days integer,
  season_sample numeric,
  sample numeric,
  tracked_share numeric,
  trait_grades smallint[] NOT NULL DEFAULT '{}',
  trait_pcts smallint[] NOT NULL DEFAULT '{}',
  tip jsonb NOT NULL DEFAULT '[]'::jsonb,
  updated_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE mlb.prospect_values ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON mlb.prospect_values FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON mlb.prospect_values TO anon, authenticated;
GRANT ALL ON mlb.prospect_values TO service_role;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.prospect_values
  FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();

COMMIT;
