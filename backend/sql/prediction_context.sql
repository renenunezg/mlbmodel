-- Apply before deploying scoring code that writes prediction_context.
-- Existing forecasts stay NULL: their raw probabilities cannot be recovered.
BEGIN;
ALTER TABLE mlb.model_outputs
    ADD COLUMN IF NOT EXISTS prediction_context jsonb;
ALTER TABLE mlb.model_outputs_season
    ADD COLUMN IF NOT EXISTS prediction_context jsonb;
COMMIT;
