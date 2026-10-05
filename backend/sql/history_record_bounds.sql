BEGIN;
DROP FUNCTION mlb.bet_record_summary(text, text);
CREATE FUNCTION mlb.bet_record_summary(p_from text DEFAULT NULL, p_team text DEFAULT NULL, p_to text DEFAULT NULL)
RETURNS TABLE(bet_type text, wins bigint, losses bigint, pushes bigint)
LANGUAGE sql STABLE SET search_path = '' AS $$
 SELECT bet_type,
        count(*) FILTER (WHERE won),
        count(*) FILTER (WHERE NOT won AND NOT push),
        count(*) FILTER (WHERE push)
 FROM mlb.bet_ledger_agg_v
 WHERE (nullif(p_from, '') IS NULL OR date::date >= p_from::date)
   AND (nullif(p_to, '') IS NULL OR date::date <= p_to::date)
   AND (nullif(p_team, '') IS NULL OR team = p_team)
 GROUP BY bet_type
$$;
GRANT EXECUTE ON FUNCTION mlb.bet_record_summary(text, text, text) TO anon, authenticated, service_role;
NOTIFY pgrst, 'reload schema';
COMMIT;
