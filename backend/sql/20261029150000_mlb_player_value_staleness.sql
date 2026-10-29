BEGIN;

-- Let the site detect grades scored by a different fit than the evaluation it shows. Nightly scoring stamps each
-- player_values row with the fitted_at of the model row it used; the evaluation view exposes that row's fitted_at
-- and its release fingerprint. Served grades are current only when player_values.model_fitted_at equals
-- player_value_evaluation.fitted_at for the same model_version.
ALTER TABLE mlb.player_values ADD COLUMN model_fitted_at timestamptz;
COMMENT ON COLUMN mlb.player_values.model_fitted_at IS
  'fitted_at of the mlb.player_value_models row that scored this player; compare with mlb.player_value_evaluation.fitted_at.';

CREATE OR REPLACE VIEW mlb.player_value_evaluation AS
SELECT model_version, fitted_at, payload->'evaluation'->'forward' AS forward,
       payload->'evaluation'->'market'->>'model_fingerprint' AS model_fingerprint
FROM mlb.player_value_models;
COMMENT ON COLUMN mlb.player_value_evaluation.model_fingerprint IS
  'Hash binding the release evidence to the model parameters and every v2/players source file; the nightly scorer refuses a row whose fingerprint no longer matches the code.';

NOTIFY pgrst, 'reload schema';

COMMIT;
