BEGIN;

-- The site reads the grade's forward evaluation (payload->'evaluation'->'forward') with the anon key.
-- mlb.player_value_models itself stays unreadable to anon and authenticated: its payload also carries the fitted
-- weights, curves and prices. The view runs as its owner (no security_invoker) and exposes only the evaluation
-- block, so anon never touches the base table.
CREATE VIEW mlb.player_value_evaluation AS
SELECT model_version, fitted_at, payload->'evaluation'->'forward' AS forward
FROM mlb.player_value_models;
GRANT SELECT ON mlb.player_value_evaluation TO anon, authenticated;

-- The page that renders the block is cached; refresh it when the model row changes.
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON mlb.player_value_models
  FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();

COMMIT;
