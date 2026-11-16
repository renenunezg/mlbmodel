BEGIN;

-- xWAR (v2.xwar): fWAR with the luck on contact taken out.
--   hitters:   xwar = war - (realized_gap_runs - earned_gap_runs) / runs per win
--   pitchers:  xwar = war + (13 / 9) * (home_run_gap - earned_home_runs) / runs per win
-- A two-way player gets both. Written by v2.xwar.publish.

-- One jsonb row per edition: the seasons covered, how much of a gap a player keeps (k, in tracked balls, per side),
-- whether each side's gap repeated on the development pairs, and the single read of the 2024-25 pairs.
CREATE TABLE mlb.xwar_study (
  season integer PRIMARY KEY,
  model_version text NOT NULL,
  payload jsonb NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE mlb.xwar_study ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON mlb.xwar_study FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON mlb.xwar_study TO anon, authenticated;
GRANT ALL ON mlb.xwar_study TO service_role;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.xwar_study
  FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();

-- One row per player-season (the latest season only).
-- war: fWAR, hitting and pitching summed; the number player_values shows as Impact WAR.
-- xwar: war with the unearned part of each gap removed.
-- Hitting columns, null for a player who did not bat:
--   balls: tracked balls in play.
--   realized_gap_runs: what those balls did against the contact model (exit velocity, launch angle, sprint speed),
--     park removed, the season centred on zero.
--   earned_gap_runs: the part of it his own record since 2021 supports; he keeps this much.
-- Pitching columns, null for a player who did not pitch:
--   balls_allowed: tracked balls in play.
--   home_runs_allowed, expected_home_runs: on those balls, as hit and as the contact model expected.
--   home_run_gap: their difference with the park removed and the season centred on zero; the swap uses this.
--   earned_home_runs: the part of the gap his own record supports.
CREATE TABLE mlb.xwar (
  player_id bigint NOT NULL,
  season smallint NOT NULL,
  player_name text NOT NULL,
  team text NOT NULL,
  war real NOT NULL,
  xwar real NOT NULL,
  balls smallint,
  realized_gap_runs real,
  earned_gap_runs real,
  balls_allowed smallint,
  home_runs_allowed real,
  expected_home_runs real,
  home_run_gap real,
  earned_home_runs real,
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (player_id, season)
);
ALTER TABLE mlb.xwar ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON mlb.xwar FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON mlb.xwar TO anon, authenticated;
GRANT ALL ON mlb.xwar TO service_role;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.xwar
  FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();

COMMIT;
