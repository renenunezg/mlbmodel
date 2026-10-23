BEGIN;

-- The fitted prospect model, one jsonb row per fit (boosters as base64). Scoring in CI loads the newest.
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

-- One row per rookie-eligible player in affiliated ball: peak grade, probability of a regular, trade value, ETA.
CREATE TABLE mlb.prospect_values (
  player_id bigint PRIMARY KEY,
  player_name text,
  org text,
  org_id integer,
  team text,
  team_id integer,
  level text,
  source text NOT NULL DEFAULT 'milb',
  position_group text NOT NULL,
  secondary_positions text[] NOT NULL DEFAULT '{}',
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
  eta_low integer,
  eta_high integer,
  trade_value_musd numeric,
  trade_value_low numeric,
  trade_value_high numeric,
  market_value_musd numeric,
  expected_control_war numeric,
  peak_war numeric,
  risk text,
  rookie_eligible boolean NOT NULL DEFAULT true,
  on_40_man boolean NOT NULL DEFAULT false,
  mlb_pa_or_ip numeric,
  service_days integer,
  injured boolean NOT NULL DEFAULT false,
  season_sample numeric,
  sample numeric,
  tracked_share numeric,
  updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX prospect_values_grade_idx ON mlb.prospect_values (peak_grade DESC);
CREATE INDEX prospect_values_org_idx ON mlb.prospect_values (org_id, peak_grade DESC);
ALTER TABLE mlb.prospect_values ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON mlb.prospect_values FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON mlb.prospect_values TO anon, authenticated;
GRANT ALL ON mlb.prospect_values TO service_role;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.prospect_values
  FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();

-- Per player: the season-by-season outlook and the tip content the site renders without model code.
CREATE TABLE mlb.prospect_details (
  player_id bigint PRIMARY KEY,
  outlook jsonb NOT NULL DEFAULT '[]'::jsonb,
  tip jsonb NOT NULL DEFAULT '{}'::jsonb,
  updated_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE mlb.prospect_details ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON mlb.prospect_details FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON mlb.prospect_details TO anon, authenticated;
GRANT ALL ON mlb.prospect_details TO service_role;

-- One row per player per trait, same shape as player_traits.
CREATE TABLE mlb.prospect_traits (
  player_id bigint NOT NULL,
  trait text NOT NULL,
  panel text NOT NULL,
  grade numeric,
  grade_low numeric,
  grade_high numeric,
  percentile numeric,
  value numeric,
  in_grade boolean NOT NULL,
  kind text NOT NULL,
  sample integer NOT NULL DEFAULT 0,
  inputs jsonb NOT NULL DEFAULT '[]'::jsonb,
  as_of date NOT NULL,
  model_version text NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (player_id, trait)
);
CREATE INDEX prospect_traits_panel_trait_idx ON mlb.prospect_traits (panel, trait);
ALTER TABLE mlb.prospect_traits ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON mlb.prospect_traits FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON mlb.prospect_traits TO anon, authenticated;
GRANT ALL ON mlb.prospect_traits TO service_role;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.prospect_traits
  FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();

COMMIT;
