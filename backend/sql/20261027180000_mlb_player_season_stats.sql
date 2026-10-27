BEGIN;

-- One row per published player (player_values: pool 'mlb'; prospect_values: pool 'prospect') with a line in the
-- pool's latest season, written by v2.season_stats. levels: the affiliated levels of a prospect's line, highest
-- first ('aaa,aa'); null for MLB. stats: integers in a fixed order per role (the role is the card's position group),
-- scaled as v2/season_stats.py BAT and PIT document; null where a stat does not exist for the player.
-- Hitters: pa, war*10, avg, obp, slg, ops (*1000), hr, rbi, sb, k, bb, xwoba*1000, barrel, hard_hit (*1000), ev*10,
-- la*10, chase*1000, ev90*10, bbe. Pitchers: bf, outs, war*10, era, fip, whip (*100), k, bb, hr, whiff*1000,
-- fb_velo*10, gb, xwoba, hard_hit (*1000), bbe, pitches.
CREATE TABLE mlb.player_season_stats (
  player_id bigint NOT NULL,
  pool text NOT NULL CHECK (pool IN ('mlb', 'prospect')),
  season smallint NOT NULL,
  levels text,
  stats smallint[] NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (player_id, pool)
);
ALTER TABLE mlb.player_season_stats ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON mlb.player_season_stats FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON mlb.player_season_stats TO anon, authenticated;
GRANT ALL ON mlb.player_season_stats TO service_role;
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.player_season_stats
  FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();

COMMIT;
