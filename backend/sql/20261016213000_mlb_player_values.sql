BEGIN;

-- The fitted valuation model, one jsonb row per fit. Scoring in CI loads the newest.
CREATE TABLE mlb.player_value_models (
  model_version text NOT NULL,
  last_dev_season integer NOT NULL,
  payload jsonb NOT NULL,
  fitted_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (model_version, last_dev_season)
);
ALTER TABLE mlb.player_value_models ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON mlb.player_value_models FROM PUBLIC, anon, authenticated;
GRANT ALL ON mlb.player_value_models TO service_role;

-- One row per player: Grade, Trade value, Impact, contract summary, and the tip components.
CREATE TABLE mlb.player_values (
  player_id bigint PRIMARY KEY,
  player_name text,
  team text,
  team_id integer,
  level text NOT NULL DEFAULT 'mlb',
  position_group text NOT NULL,
  secondary_positions text[] NOT NULL DEFAULT '{}',
  age integer,
  bats text,
  throws text,
  status text NOT NULL,
  as_of date NOT NULL,
  model_version text NOT NULL,
  season integer NOT NULL,
  grade numeric,
  grade_low numeric,
  grade_high numeric,
  grade_percentile numeric,
  grade_reason text,
  projected_war numeric,
  trade_value_musd numeric,
  trade_value_low numeric,
  trade_value_high numeric,
  market_value_musd numeric,
  impact_war numeric,
  impact_runs numeric,
  playing_time numeric,
  service_time numeric,
  contract_status text NOT NULL,
  fa_year integer,
  control_years integer NOT NULL DEFAULT 0,
  salary_current_musd numeric,
  owed_through_control_musd numeric,
  outlook jsonb NOT NULL DEFAULT '[]'::jsonb,
  tip jsonb NOT NULL DEFAULT '{}'::jsonb,
  headshot_url text,
  updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX player_values_position_grade_idx ON mlb.player_values (position_group, grade DESC);
ALTER TABLE mlb.player_values ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON mlb.player_values FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON mlb.player_values TO anon, authenticated;
GRANT ALL ON mlb.player_values TO service_role;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.player_values
  FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();

-- One row per player per trait, with the inputs behind the tip as numbers only.
CREATE TABLE mlb.player_traits (
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
CREATE INDEX player_traits_panel_trait_idx ON mlb.player_traits (panel, trait);
ALTER TABLE mlb.player_traits ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON mlb.player_traits FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON mlb.player_traits TO anon, authenticated;
GRANT ALL ON mlb.player_traits TO service_role;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.player_traits
  FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();

COMMIT;
