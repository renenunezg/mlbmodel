BEGIN;

-- One row per organization, ranked by the trade value of its 30 most valuable prospects. Past thirty, prospects rarely
-- reach the majors, their small values lean on the least certain probabilities, and an org cannot roster them all, so
-- the deep tail is shown (total_value_musd) but does not set the order. A view over mlb.prospect_values, so it cannot
-- drift from the prospect board and is refreshed by that table's site_revalidate trigger. The best prospect is the
-- highest peak grade, ties to trade value.
DROP VIEW IF EXISTS mlb.farm_systems;
CREATE VIEW mlb.farm_systems WITH (security_invoker = true) AS
WITH ranked AS (
  SELECT
    org,
    org_id,
    player_id,
    player_name,
    position_group,
    headshot_url,
    trade_value_musd,
    peak_grade,
    p_regular,
    expected_control_war,
    season,
    as_of,
    row_number() OVER (
      PARTITION BY org ORDER BY trade_value_musd DESC NULLS LAST, peak_grade DESC NULLS LAST, player_id
    ) AS org_rank,
    row_number() OVER (
      PARTITION BY org ORDER BY peak_grade DESC NULLS LAST, trade_value_musd DESC NULLS LAST, player_id
    ) AS grade_rank
  FROM mlb.prospect_values
  WHERE org IS NOT NULL
),
orgs AS (
  SELECT
    org,
    max(org_id) AS org_id,
    coalesce(sum(trade_value_musd) FILTER (WHERE org_rank <= 30), 0) AS top30_value_musd,
    coalesce(sum(trade_value_musd), 0) AS total_value_musd,
    coalesce(sum(p_regular) FILTER (WHERE org_rank <= 30), 0) AS top30_expected_regulars,
    coalesce(sum(expected_control_war) FILTER (WHERE org_rank <= 30), 0) AS top30_control_war,
    count(*) AS prospects,
    count(*) FILTER (WHERE peak_grade >= 50) AS grade_50_plus,
    max(season) AS season,
    max(as_of) AS as_of
  FROM ranked
  GROUP BY org
)
SELECT
  rank() OVER (ORDER BY o.top30_value_musd DESC) AS farm_rank,
  o.org,
  o.org_id,
  o.top30_value_musd,
  o.total_value_musd,
  o.top30_expected_regulars,
  o.top30_control_war,
  o.prospects,
  o.grade_50_plus,
  b.player_id AS best_player_id,
  b.player_name AS best_player_name,
  b.position_group AS best_position_group,
  b.headshot_url AS best_headshot_url,
  b.peak_grade AS best_peak_grade,
  b.trade_value_musd AS best_value_musd,
  o.season,
  o.as_of
FROM orgs o
JOIN ranked b ON b.org = o.org AND b.grade_rank = 1;

GRANT SELECT ON mlb.farm_systems TO anon, authenticated, service_role;

COMMIT;
