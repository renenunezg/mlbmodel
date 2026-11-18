BEGIN;

-- The site reads mlb with the anon key and writes nothing through it; the pipeline writes as service_role or postgres.
-- Tables and views created before the schema's read-only default privileges still carried Supabase's original
-- full grants to anon and authenticated. RLS, with no write policies, already refused those writes through the API;
-- revoking them means a table that loses RLS, or a view that runs as its owner, cannot become writable from the browser.
REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON ALL TABLES IN SCHEMA mlb FROM anon, authenticated;

COMMIT;
