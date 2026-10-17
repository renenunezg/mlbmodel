BEGIN;

-- Contract seasons from Baseball-Reference, replaced weekly. Nightly player
-- valuation reads this table, so the fetch can run wherever the site is reachable.
CREATE TABLE mlb.player_contracts (
  player_id bigint NOT NULL,
  season integer NOT NULL,
  salary_musd numeric,
  salary_kind text NOT NULL CHECK (salary_kind IN ('guaranteed', 'option', 'buyout', 'arbitration', 'pre_arb', 'free_agent')),
  option_musd numeric,
  option_kind text,
  declined_to text,
  service_time numeric,
  how_acquired text,
  fa_year integer,
  contract_start integer,
  contract_years integer,
  contract_total_musd numeric,
  team text,
  player_name text,
  source_url text,
  fetched_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (player_id, season)
);
ALTER TABLE mlb.player_contracts ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON mlb.player_contracts FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON mlb.player_contracts TO anon, authenticated;
GRANT ALL ON mlb.player_contracts TO service_role;

COMMIT;
