BEGIN;

-- calibration_bins duplicated mlb.model_calibration, which the nightly job writes for the season window and the
-- site reads. Nothing read the jsonb copy, and the nightly job no longer writes it. Apply only after the code that
-- stopped writing it is on the release branch.
ALTER TABLE mlb.model_evaluation DROP COLUMN calibration_bins;

NOTIFY pgrst, 'reload schema';

COMMIT;
