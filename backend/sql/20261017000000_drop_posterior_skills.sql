BEGIN;

-- The Diagnostics leaderboard was replaced by the player valuation tables
-- (player_values, player_traits); nothing writes or reads this table now.
DROP TABLE mlb.posterior_skills;

COMMIT;
