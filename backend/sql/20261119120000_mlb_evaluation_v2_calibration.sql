BEGIN;

-- Calibration by window. Every stored row is the season window (v1 and v2 together); season_v2 holds v2 forecasts
-- only (dates on or after 2026-05-12), the population model_evaluation already grades as season_v2. The column
-- keeps no default so a writer must name its window. Apply together with the release that writes eval_window:
-- the previous code's ON CONFLICT (date, bin_mid) fails once that key is gone.
ALTER TABLE mlb.model_calibration ADD COLUMN eval_window character varying;
UPDATE mlb.model_calibration SET eval_window = 'season';
ALTER TABLE mlb.model_calibration
  ALTER COLUMN eval_window SET NOT NULL,
  ADD CONSTRAINT model_calibration_eval_window_check CHECK (eval_window IN ('season', 'season_v2')),
  DROP CONSTRAINT model_calibration_date_bin_mid_key,
  ADD CONSTRAINT model_calibration_date_window_bin_mid_key UNIQUE (date, eval_window, bin_mid);

COMMENT ON COLUMN mlb.model_calibration.eval_window IS
  'season covers v1 and v2 together; season_v2 is season to date for v2 forecasts only. Read one window at a time.';

-- Run-interval coverage that can be read against the nominal level. interval_coverage_* counts the actual runs
-- inside an inclusive integer interval, which holds more than the nominal mass, so a calibrated forecast scores
-- about 0.61, 0.86 and 0.95 on 2026 v2 games. pit_coverage_* is the expected share of randomized PIT values inside
-- the central band, exactly the nominal level for a calibrated forecast. Same team rows as
-- interval_coverage_predictions.
ALTER TABLE mlb.model_evaluation
  ADD COLUMN pit_coverage_50 numeric,
  ADD COLUMN pit_coverage_80 numeric,
  ADD COLUMN pit_coverage_90 numeric;

COMMENT ON COLUMN mlb.model_evaluation.pit_coverage_50 IS
  'Per team row, the randomized PIT of its own runs is uniform on [F(y-1), F(y)] of its frozen run PMF; this is the mean probability that it falls in the central 50% band [0.25, 0.75]. A calibrated forecast scores 0.50; compare with the nominal level, unlike interval_coverage_50. NULL on rows written before 2026-11-19.';
COMMENT ON COLUMN mlb.model_evaluation.pit_coverage_80 IS 'As pit_coverage_50, for the central 80% band [0.10, 0.90].';
COMMENT ON COLUMN mlb.model_evaluation.pit_coverage_90 IS 'As pit_coverage_50, for the central 90% band [0.05, 0.95].';

NOTIFY pgrst, 'reload schema';

COMMIT;
