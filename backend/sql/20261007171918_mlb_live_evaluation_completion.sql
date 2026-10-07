BEGIN;

-- A final score is not proof that the evaluation transaction committed.
CREATE TABLE mlb.live_evaluation_completions (
  game_pk integer PRIMARY KEY REFERENCES mlb.games(game_pk) ON DELETE CASCADE,
  input_version text NOT NULL,
  eval_date date NOT NULL,
  completed_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
ALTER TABLE mlb.live_evaluation_completions ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON mlb.live_evaluation_completions FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT, UPDATE ON mlb.live_evaluation_completions TO service_role;

-- Only inputs consumed by live grading participate. Ingestion heartbeats and
-- unrelated forecast metadata must not cause another full-history scan.
CREATE FUNCTION mlb.live_evaluation_input_version(p_game_pk integer)
RETURNS text LANGUAGE sql STABLE SET search_path = '' AS $$
  SELECT md5(jsonb_build_array(
    g.game_date, g.home_team, g.away_team, g.home_score, g.away_score,
    (SELECT jsonb_agg(to_jsonb(p) ORDER BY p.team, p.date)
     FROM (
       SELECT game_pk, date, team, expected_runs, win_prob, ev_flag,
         run_line_ev_flag, spread, total, total_play, moneyline,
         kelly_quarter_ml, kelly_quarter_total, total_over_odds, total_under_odds
       FROM mlb.model_outputs_season_unified
       WHERE game_pk = p_game_pk AND date::date = g.game_date
         AND team IN (g.home_team, g.away_team)
     ) p)
  )::text)
  FROM mlb.games g
  WHERE g.game_pk = p_game_pk AND g.status = 'Final'
    AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
    AND EXISTS (SELECT 1 FROM mlb.model_outputs_season_unified p
      WHERE p.game_pk = p_game_pk AND p.date::date = g.game_date
        AND p.team IN (g.home_team, g.away_team));
$$;

CREATE FUNCTION mlb.live_evaluation_status(p_game_pk integer)
RETURNS TABLE(input_version text, eval_date date)
LANGUAGE sql STABLE SET search_path = '' AS $$
  WITH input AS MATERIALIZED (
    SELECT mlb.live_evaluation_input_version(p_game_pk) AS version
  )
  SELECT v.version, c.eval_date
  FROM input v
  LEFT JOIN mlb.live_evaluation_completions c
    ON c.game_pk = p_game_pk AND c.input_version = v.version;
$$;

-- Keep the existing publication API for older deployments and canonical
-- reconciliation. The new API commits completion and all windows together.
CREATE FUNCTION mlb.complete_live_evaluation(
  p_game_pk integer, p_input_version text, p_started_at timestamptz, p_rows jsonb
) RETURNS void LANGUAGE plpgsql SET search_path = '' AS $$
DECLARE evaluation_date date;
BEGIN
  PERFORM pg_advisory_xact_lock(20261002, 1);
  IF p_input_version IS NULL OR
      mlb.live_evaluation_input_version(p_game_pk) IS DISTINCT FROM p_input_version THEN
    RAISE EXCEPTION 'Live evaluation inputs changed; retry required';
  END IF;
  evaluation_date := (p_rows->0->>'date')::date;
  IF evaluation_date IS NULL OR evaluation_date <
      (SELECT game_date FROM mlb.games WHERE game_pk = p_game_pk) THEN
    RAISE EXCEPTION 'Live evaluation does not cover the completed game';
  END IF;
  -- Another instance may have finished while this caller was reading history.
  IF EXISTS (SELECT 1 FROM mlb.live_evaluation_completions
      WHERE game_pk = p_game_pk AND input_version = p_input_version) THEN
    RETURN;
  END IF;
  PERFORM mlb.publish_live_evaluation(p_started_at, p_rows);
  INSERT INTO mlb.live_evaluation_completions(game_pk, input_version, eval_date)
  VALUES (p_game_pk, p_input_version, evaluation_date)
  ON CONFLICT (game_pk) DO UPDATE SET input_version = EXCLUDED.input_version,
    eval_date = EXCLUDED.eval_date, completed_at = clock_timestamp();
END $$;

REVOKE ALL ON FUNCTION mlb.live_evaluation_input_version(integer) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION mlb.live_evaluation_status(integer) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION mlb.complete_live_evaluation(integer, text, timestamptz, jsonb) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION mlb.live_evaluation_input_version(integer) TO service_role;
GRANT EXECUTE ON FUNCTION mlb.live_evaluation_status(integer) TO service_role;
GRANT EXECUTE ON FUNCTION mlb.complete_live_evaluation(integer, text, timestamptz, jsonb) TO service_role;
NOTIFY pgrst, 'reload schema';

COMMIT;
