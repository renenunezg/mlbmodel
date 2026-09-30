-- Separate live state from immutable pregame forecasts.
BEGIN;
CREATE TABLE IF NOT EXISTS mlb.live_win_probability (
    game_pk integer PRIMARY KEY CHECK (game_pk > 0),
    updated_at timestamptz NOT NULL,
    source_timestamp timestamptz NOT NULL,
    payload jsonb NOT NULL CHECK (
        payload->>'schema_version' = '1'
        AND (payload->>'game_pk')::integer = game_pk
    )
);
ALTER TABLE mlb.live_win_probability ENABLE ROW LEVEL SECURITY;
GRANT SELECT ON mlb.live_win_probability TO anon, authenticated;
DROP POLICY IF EXISTS live_win_probability_read ON mlb.live_win_probability;
CREATE POLICY live_win_probability_read ON mlb.live_win_probability FOR SELECT USING (true);
DROP TRIGGER IF EXISTS live_win_probability_revalidate ON mlb.live_win_probability;
CREATE TRIGGER live_win_probability_revalidate
    AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON mlb.live_win_probability
    FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();

-- Check only local serving tables overnight. The final snapshot takes priority
-- over games.status, which can retain its pregame value until the nightly job.
SELECT cron.schedule('live-win-probability-dispatch', '*/5 * * * *', $dispatch$
    SELECT net.http_post(
        url := 'https://api.github.com/repos/renenunezg/mlbmodel/dispatches',
        headers := jsonb_build_object(
            'Authorization', 'Bearer ' || (SELECT decrypted_secret
                FROM vault.decrypted_secrets WHERE name = 'github_dispatch_pat'),
            'Accept', 'application/vnd.github+json',
            'User-Agent', 'supabase-pg-cron',
            'X-GitHub-Api-Version', '2022-11-28'
        ),
        body := jsonb_build_object('event_type', 'live-win-probability'),
        timeout_milliseconds := 15000
    ) WHERE EXISTS (
        SELECT 1 FROM mlb.games AS g
        LEFT JOIN mlb.live_win_probability AS live USING (game_pk)
        WHERE g.game_date BETWEEN (now() AT TIME ZONE 'America/Los_Angeles')::date - 1
                              AND (now() AT TIME ZONE 'America/Los_Angeles')::date + 1
          AND g.start_time <= now() + interval '15 minutes'
          AND g.status NOT IN ('Final', 'Postponed', 'Cancelled', 'Canceled', 'Suspended', 'If Necessary')
          AND coalesce(live.payload->>'abstract_state', '') <> 'Final'
    );
$dispatch$);
COMMIT;
