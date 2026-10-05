-- Live summaries are provisional until the model-owned nightly reconciliation.
ALTER TABLE mlb.model_evaluation
  ADD COLUMN evaluation_state text NOT NULL DEFAULT 'legacy'
    CHECK (evaluation_state IN ('legacy', 'provisional', 'canonical')),
  ADD COLUMN evaluation_started_at timestamptz;

CREATE FUNCTION mlb.live_evaluation_started_at() RETURNS timestamptz
LANGUAGE sql VOLATILE SET search_path = '' AS $$ SELECT clock_timestamp() $$;

CREATE FUNCTION mlb.publish_live_evaluation(p_started_at timestamptz, p_rows jsonb)
RETURNS void LANGUAGE plpgsql SET search_path = '' AS $$
DECLARE item jsonb; row mlb.model_evaluation;
BEGIN
 IF p_started_at IS NULL OR p_started_at > clock_timestamp()
    OR jsonb_typeof(p_rows) <> 'array' OR jsonb_array_length(p_rows) <> 4 THEN
   RAISE EXCEPTION 'Invalid live evaluation snapshot';
 END IF;
 IF (SELECT count(DISTINCT value->>'date') FROM jsonb_array_elements(p_rows)) <> 1
    OR (SELECT array_agg(DISTINCT value->>'eval_window' ORDER BY value->>'eval_window')
        FROM jsonb_array_elements(p_rows)) <> ARRAY['30d','7d','day','season'] THEN
   RAISE EXCEPTION 'Live evaluation must contain all four windows for one date';
 END IF;
 PERFORM pg_advisory_xact_lock(20261002, 1);
 FOR item IN SELECT value FROM jsonb_array_elements(p_rows) LOOP
   row := jsonb_populate_record(NULL::mlb.model_evaluation, item);
   INSERT INTO mlb.model_evaluation (date, eval_window, evaluation_state, evaluation_started_at,
     total_correct,
     total_predictions,
     total_accuracy,
     ml_correct,
     ml_predictions,
     ml_accuracy,
     run_line_correct,
     run_line_predictions,
     run_line_accuracy,
     totals_correct,
     totals_predictions,
     totals_accuracy,
     average_total_diff,
     average_win_prob,
     roi_favorites,
     n_favorites,
     favorites_correct,
     roi_underdogs,
     n_underdogs,
     underdogs_correct,
     avg_ml_line,
     overs_correct,
     overs_predictions,
     overs_roi,
     unders_correct,
     unders_predictions,
     unders_roi)
   VALUES (row.date, row.eval_window, 'provisional', p_started_at,
     row.total_correct,
     row.total_predictions,
     row.total_accuracy,
     row.ml_correct,
     row.ml_predictions,
     row.ml_accuracy,
     row.run_line_correct,
     row.run_line_predictions,
     row.run_line_accuracy,
     row.totals_correct,
     row.totals_predictions,
     row.totals_accuracy,
     row.average_total_diff,
     row.average_win_prob,
     row.roi_favorites,
     row.n_favorites,
     row.favorites_correct,
     row.roi_underdogs,
     row.n_underdogs,
     row.underdogs_correct,
     row.avg_ml_line,
     row.overs_correct,
     row.overs_predictions,
     row.overs_roi,
     row.unders_correct,
     row.unders_predictions,
     row.unders_roi)
   ON CONFLICT (date, eval_window) DO UPDATE SET
     evaluation_state = 'provisional', evaluation_started_at = p_started_at,
     total_correct = EXCLUDED.total_correct,
     total_predictions = EXCLUDED.total_predictions,
     total_accuracy = EXCLUDED.total_accuracy,
     ml_correct = EXCLUDED.ml_correct,
     ml_predictions = EXCLUDED.ml_predictions,
     ml_accuracy = EXCLUDED.ml_accuracy,
     run_line_correct = EXCLUDED.run_line_correct,
     run_line_predictions = EXCLUDED.run_line_predictions,
     run_line_accuracy = EXCLUDED.run_line_accuracy,
     totals_correct = EXCLUDED.totals_correct,
     totals_predictions = EXCLUDED.totals_predictions,
     totals_accuracy = EXCLUDED.totals_accuracy,
     average_total_diff = EXCLUDED.average_total_diff,
     average_win_prob = EXCLUDED.average_win_prob,
     roi_favorites = EXCLUDED.roi_favorites,
     n_favorites = EXCLUDED.n_favorites,
     favorites_correct = EXCLUDED.favorites_correct,
     roi_underdogs = EXCLUDED.roi_underdogs,
     n_underdogs = EXCLUDED.n_underdogs,
     underdogs_correct = EXCLUDED.underdogs_correct,
     avg_ml_line = EXCLUDED.avg_ml_line,
     overs_correct = EXCLUDED.overs_correct,
     overs_predictions = EXCLUDED.overs_predictions,
     overs_roi = EXCLUDED.overs_roi,
     unders_correct = EXCLUDED.unders_correct,
     unders_predictions = EXCLUDED.unders_predictions,
     unders_roi = EXCLUDED.unders_roi
   WHERE mlb.model_evaluation.evaluation_state <> 'canonical'
     AND (mlb.model_evaluation.evaluation_started_at IS NULL
          OR mlb.model_evaluation.evaluation_started_at < p_started_at);
 END LOOP;
END $$;
REVOKE ALL ON FUNCTION mlb.live_evaluation_started_at() FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION mlb.publish_live_evaluation(timestamptz, jsonb) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION mlb.live_evaluation_started_at() TO service_role;
GRANT EXECUTE ON FUNCTION mlb.publish_live_evaluation(timestamptz, jsonb) TO service_role;
NOTIFY pgrst, 'reload schema';
