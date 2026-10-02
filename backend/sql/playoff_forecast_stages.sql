-- Apply before publishing a playoff forecast for a later round.
-- One snapshot per round: the bracket prediction going into the Wild Card,
-- Division Series, Championship Series and World Series. The single snapshot
-- stored before this migration was taken before the Wild Card round.
BEGIN;
ALTER TABLE mlb.playoff_forecasts
    ADD COLUMN IF NOT EXISTS stage text NOT NULL DEFAULT 'WC'
    CHECK (stage IN ('WC', 'DS', 'CS', 'WS'));
ALTER TABLE mlb.playoff_forecasts ALTER COLUMN stage DROP DEFAULT;
ALTER TABLE mlb.playoff_forecasts DROP CONSTRAINT playoff_forecasts_pkey;
ALTER TABLE mlb.playoff_forecasts ADD PRIMARY KEY (season, stage);
COMMIT;
