-- Fresh homepage aggregates over the canonical graded ledger; no stored snapshot.
CREATE OR REPLACE FUNCTION mlb.betting_headline()
RETURNS TABLE (
  bet_type text, wins bigint, losses bigint, pushes bigint,
  total_stake double precision, total_payout double precision
)
LANGUAGE sql STABLE SECURITY INVOKER
SET search_path = pg_catalog, mlb
AS $$
  SELECT bet_type,
    count(*) FILTER (WHERE won) AS wins,
    count(*) - count(*) FILTER (WHERE won) - count(*) FILTER (WHERE push) AS losses,
    count(*) FILTER (WHERE push) AS pushes,
    -- Match the frontend's existing stable ledger order and float sums, so
    -- four-decimal ROI rounding remains identical at the display boundary.
    sum(stake::double precision ORDER BY date, game_pk, bet_type, team) AS total_stake,
    sum(payout::double precision ORDER BY date, game_pk, bet_type, team) AS total_payout
  FROM mlb.bet_ledger_v
  GROUP BY bet_type;
$$;
REVOKE ALL ON FUNCTION mlb.betting_headline() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION mlb.betting_headline() TO anon, authenticated, service_role;
