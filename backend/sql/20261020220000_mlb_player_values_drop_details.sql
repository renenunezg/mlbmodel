BEGIN;

-- The site reads outlook and tip from player_details, so player_values drops its copies and
-- returns to default storage: its rows now sit far under the toast threshold.
ALTER TABLE mlb.player_values DROP COLUMN outlook, DROP COLUMN tip;
ALTER TABLE mlb.player_values RESET (toast_tuple_target);
ALTER TABLE mlb.player_values
  ALTER COLUMN player_name SET STORAGE EXTENDED,
  ALTER COLUMN team SET STORAGE EXTENDED,
  ALTER COLUMN level SET STORAGE EXTENDED,
  ALTER COLUMN position_group SET STORAGE EXTENDED,
  ALTER COLUMN secondary_positions SET STORAGE EXTENDED,
  ALTER COLUMN bats SET STORAGE EXTENDED,
  ALTER COLUMN throws SET STORAGE EXTENDED,
  ALTER COLUMN status SET STORAGE EXTENDED,
  ALTER COLUMN model_version SET STORAGE EXTENDED,
  ALTER COLUMN grade_reason SET STORAGE EXTENDED,
  ALTER COLUMN contract_status SET STORAGE EXTENDED,
  ALTER COLUMN headshot_url SET STORAGE EXTENDED;

COMMIT;
