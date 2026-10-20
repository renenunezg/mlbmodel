BEGIN;

-- The per-player narrative (outlook and tip), read only for the players a visitor opens.
-- Kept apart from player_values so the site's full-table read stays a few dozen heap pages.
CREATE TABLE mlb.player_details (
  player_id bigint PRIMARY KEY,
  outlook jsonb NOT NULL DEFAULT '[]'::jsonb,
  tip jsonb NOT NULL DEFAULT '{}'::jsonb,
  updated_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE mlb.player_details ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON mlb.player_details FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON mlb.player_details TO anon, authenticated;
GRANT ALL ON mlb.player_details TO service_role;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.player_details
  FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();

INSERT INTO mlb.player_details (player_id, outlook, tip, updated_at)
SELECT player_id, outlook, tip, updated_at FROM mlb.player_values ORDER BY player_id;

-- Until the site reads player_details, player_values keeps its own outlook and tip. Once a row
-- passes the 2 kB toast threshold, push both out of line whole instead of stopping at 2 kB,
-- and keep the text the list reads inline.
ALTER TABLE mlb.player_values SET (toast_tuple_target = 128);
ALTER TABLE mlb.player_values
  ALTER COLUMN outlook SET STORAGE EXTERNAL,
  ALTER COLUMN tip SET STORAGE EXTERNAL,
  ALTER COLUMN player_name SET STORAGE MAIN,
  ALTER COLUMN team SET STORAGE MAIN,
  ALTER COLUMN level SET STORAGE MAIN,
  ALTER COLUMN position_group SET STORAGE MAIN,
  ALTER COLUMN secondary_positions SET STORAGE MAIN,
  ALTER COLUMN bats SET STORAGE MAIN,
  ALTER COLUMN throws SET STORAGE MAIN,
  ALTER COLUMN status SET STORAGE MAIN,
  ALTER COLUMN model_version SET STORAGE MAIN,
  ALTER COLUMN grade_reason SET STORAGE MAIN,
  ALTER COLUMN contract_status SET STORAGE MAIN,
  ALTER COLUMN headshot_url SET STORAGE MAIN;

COMMIT;
