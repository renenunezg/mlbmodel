-- Apply before enabling playoff forecast publication.
BEGIN;
CREATE TABLE IF NOT EXISTS mlb.playoff_forecasts (
    season integer PRIMARY KEY CHECK (season >= 2022),
    generated_at timestamptz NOT NULL,
    payload jsonb NOT NULL CHECK (payload->>'schema_version' = '1')
);
ALTER TABLE mlb.playoff_forecasts ENABLE ROW LEVEL SECURITY;
GRANT SELECT ON mlb.playoff_forecasts TO anon, authenticated;
DROP POLICY IF EXISTS playoff_forecasts_read ON mlb.playoff_forecasts;
CREATE POLICY playoff_forecasts_read ON mlb.playoff_forecasts FOR SELECT USING (true);
DROP TRIGGER IF EXISTS playoff_forecasts_revalidate ON mlb.playoff_forecasts;
CREATE TRIGGER playoff_forecasts_revalidate
    AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON mlb.playoff_forecasts
    FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();
COMMIT;
