-- Apply before deploying postseason schedule ingestion and scoring.
-- Refresh the schedule afterward to classify upcoming games from MLB's API.
-- Leave older games unclassified rather than guessing their rules from dates.
BEGIN;
ALTER TABLE mlb.games
    ADD COLUMN IF NOT EXISTS game_type text
    CHECK (game_type IN ('R', 'F', 'D', 'L', 'W'));
COMMIT;
