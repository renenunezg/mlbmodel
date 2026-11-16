BEGIN;

-- A hitter's wOBA on his tracked balls in play, as hit and as the contact model (exit velocity, launch angle,
-- sprint speed) expected them, with nothing removed. Shown beside the gap in runs, which is what moves his WAR.
-- Null for a player who did not bat. Written by v2.xwar.publish.
ALTER TABLE mlb.xwar
  ADD COLUMN woba_contact real,
  ADD COLUMN expected_woba_contact real;

COMMIT;
