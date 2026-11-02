BEGIN;

-- Daily pick policy (mlb-picks-v3, backend/strategy.py PICK_POLICY). The scorer now publishes at most one pick
-- per game, the moneyline or run-line side with the highest expected value at its best price, and records it
-- in model_outputs_season.prediction_context -> 'pick' (market, team, probability, price, book,
-- expected_value, edge, tier, policy). Every such pick is graded, whatever its edge, at a flat 1u stake, with
-- edge = the pick's probability minus the vig-inclusive implied probability. Rows without a pick (every row
-- before this policy) keep the quarter-Kelly stake and the grading_rules min_edge threshold, so the existing
-- ledger is unchanged (checked: all 1,291 rows identical on 2026-11-02).
-- Apply before promoting the scorer change; the site's provisional live grading must also accept pick rows.

UPDATE mlb.grading_rules SET stake_rule = stake_rule
  || ' From mlb-picks-v3 (prediction_context.pick present): every published pick is graded at a flat 1u stake regardless of min_edge, edge = the pick probability minus the vig-inclusive implied probability.'
WHERE bet_type IN ('ml', 'rl');

CREATE OR REPLACE VIEW mlb.bet_ledger_v WITH (security_invoker='true') AS
WITH preds AS (
         SELECT model_outputs_season_v1_archive.game_pk,
            model_outputs_season_v1_archive.date,
            model_outputs_season_v1_archive.team,
            model_outputs_season_v1_archive.ev_flag,
            model_outputs_season_v1_archive.run_line_ev_flag,
            model_outputs_season_v1_archive.total_play,
            model_outputs_season_v1_archive.win_prob,
            model_outputs_season_v1_archive.p_cover,
            model_outputs_season_v1_archive.p_over,
            model_outputs_season_v1_archive.p_under,
            model_outputs_season_v1_archive.moneyline,
            model_outputs_season_v1_archive.spread,
            model_outputs_season_v1_archive.spread_odds,
            model_outputs_season_v1_archive.total,
            model_outputs_season_v1_archive.total_over_odds,
            model_outputs_season_v1_archive.total_under_odds,
            model_outputs_season_v1_archive.kelly_quarter_ml,
            model_outputs_season_v1_archive.kelly_quarter_rl,
            model_outputs_season_v1_archive.kelly_quarter_total,
            'v1'::text AS model_version,
            NULL::timestamp with time zone AS forecast_at,
            NULL::jsonb AS pick
           FROM mlb.model_outputs_season_v1_archive
          WHERE model_outputs_season_v1_archive.date::date < '2026-05-12'::date
        UNION ALL
         SELECT model_outputs_season.game_pk,
            model_outputs_season.date,
            model_outputs_season.team,
            model_outputs_season.ev_flag,
            model_outputs_season.run_line_ev_flag,
            model_outputs_season.total_play,
            model_outputs_season.win_prob,
            model_outputs_season.p_cover,
            model_outputs_season.p_over,
            model_outputs_season.p_under,
            model_outputs_season.moneyline,
            model_outputs_season.spread,
            model_outputs_season.spread_odds,
            model_outputs_season.total,
            model_outputs_season.total_over_odds,
            model_outputs_season.total_under_odds,
            model_outputs_season.kelly_quarter_ml,
            model_outputs_season.kelly_quarter_rl,
            model_outputs_season.kelly_quarter_total,
            'v2'::text AS model_version,
            model_outputs_season.prediction_updated_at AS forecast_at,
            NULLIF(model_outputs_season.prediction_context -> 'pick'::text, 'null'::jsonb) AS pick
           FROM mlb.model_outputs_season
          WHERE model_outputs_season.date::date >= '2026-05-12'::date
        ), joined AS (
         SELECT m.game_pk,
            m.date,
            m.team,
            m.ev_flag,
            m.run_line_ev_flag,
            m.total_play,
            m.win_prob,
            m.p_cover,
            m.p_over,
            m.p_under,
            m.moneyline,
            m.spread,
            m.spread_odds,
            m.total,
            m.total_over_odds,
            m.total_under_odds,
            m.kelly_quarter_ml,
            m.kelly_quarter_rl,
            m.kelly_quarter_total,
            m.model_version,
            m.forecast_at,
            m.pick,
            g.start_time,
            g.home_team,
            g.home_score,
            g.away_score,
            g.home_score + g.away_score AS game_total,
                CASE
                    WHEN g.home_team::text = m.team THEN g.home_score
                    ELSE g.away_score
                END AS team_score,
                CASE
                    WHEN g.home_team::text = m.team THEN g.away_score
                    ELSE g.home_score
                END AS opp_score,
                CASE
                    WHEN g.home_team::text = m.team THEN g.home_score - g.away_score
                    ELSE g.away_score - g.home_score
                END AS margin
           FROM preds m
             JOIN mlb.games g ON g.game_pk = m.game_pk AND g.game_date = m.date::date
          WHERE g.status::text = 'Final'::text AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
        ), ml AS (
         SELECT joined.date,
            joined.team,
            joined.game_pk,
            'ml'::text AS bet_type,
            joined.pick IS NOT NULL AS policy_pick,
                CASE
                    WHEN joined.pick IS NOT NULL THEN 1::numeric
                    ELSE joined.kelly_quarter_ml
                END AS stake,
                CASE
                    WHEN joined.moneyline > 0::double precision THEN 1::double precision + joined.moneyline / 100.0::double precision
                    ELSE 1::double precision + 100.0::double precision / abs(joined.moneyline)
                END AS decimal_odds,
            joined.moneyline::numeric AS american_odds,
            NULL::text AS totals_side,
            joined.team_score > joined.opp_score AS won,
                CASE
                    WHEN joined.pick IS NOT NULL THEN (joined.pick ->> 'edge'::text)::double precision
                    ELSE joined.win_prob -
                    CASE
                        WHEN joined.moneyline > 0::double precision THEN 100.0::double precision / (joined.moneyline + 100.0::double precision)
                        ELSE (- joined.moneyline) / ((- joined.moneyline) + 100.0::double precision)
                    END
                END AS edge,
            false AS push,
            joined.model_version,
            joined.forecast_at,
            joined.start_time
           FROM joined
          WHERE joined.ev_flag = joined.team AND joined.moneyline IS NOT NULL AND (joined.pick IS NOT NULL AND (joined.pick ->> 'market'::text) = 'ml'::text OR joined.pick IS NULL AND joined.kelly_quarter_ml IS NOT NULL AND joined.kelly_quarter_ml > 0::numeric AND joined.win_prob IS NOT NULL)
        ), rl AS (
         SELECT joined.date,
            joined.team,
            joined.game_pk,
            'rl'::text AS bet_type,
            joined.pick IS NOT NULL AS policy_pick,
                CASE
                    WHEN joined.pick IS NOT NULL THEN 1::numeric
                    ELSE joined.kelly_quarter_rl
                END AS stake,
                CASE
                    WHEN joined.spread_odds > 0::double precision THEN 1::double precision + joined.spread_odds / 100.0::double precision
                    ELSE 1::double precision + 100.0::double precision / abs(joined.spread_odds)
                END AS decimal_odds,
            NULL::numeric AS american_odds,
            NULL::text AS totals_side,
                CASE
                    WHEN joined.spread < 0::double precision THEN joined.margin::double precision >= abs(joined.spread)
                    WHEN joined.spread > 0::double precision THEN joined.team_score > joined.opp_score OR joined.margin::double precision >= (- joined.spread)
                    ELSE false
                END AS won,
                CASE
                    WHEN joined.pick IS NOT NULL THEN (joined.pick ->> 'edge'::text)::double precision
                    ELSE joined.p_cover::double precision -
                    CASE
                        WHEN joined.spread_odds > 0::double precision THEN 100.0::double precision / (joined.spread_odds + 100.0::double precision)
                        ELSE (- joined.spread_odds) / ((- joined.spread_odds) + 100.0::double precision)
                    END
                END AS edge,
            false AS push,
            joined.model_version,
            joined.forecast_at,
            joined.start_time
           FROM joined
          WHERE joined.run_line_ev_flag = joined.team AND joined.spread IS NOT NULL AND joined.spread_odds IS NOT NULL AND (joined.pick IS NOT NULL AND (joined.pick ->> 'market'::text) = 'rl'::text OR joined.pick IS NULL AND joined.kelly_quarter_rl IS NOT NULL AND joined.kelly_quarter_rl > 0::numeric AND joined.p_cover IS NOT NULL)
        ), tot_raw AS (
         SELECT joined.date,
            joined.team,
            joined.game_pk,
            'total'::text AS bet_type,
            joined.kelly_quarter_total AS stake,
            lower(joined.total_play) AS totals_side,
                CASE
                    WHEN joined.total_play = 'Over'::text THEN
                    CASE
                        WHEN joined.total_over_odds > 0::numeric THEN 1::numeric + joined.total_over_odds / 100.0
                        ELSE 1::numeric + 100.0 / abs(joined.total_over_odds)
                    END
                    ELSE
                    CASE
                        WHEN joined.total_under_odds > 0::numeric THEN 1::numeric + joined.total_under_odds / 100.0
                        ELSE 1::numeric + 100.0 / abs(joined.total_under_odds)
                    END
                END AS decimal_odds,
            NULL::numeric AS american_odds,
                CASE
                    WHEN joined.total_play = 'Over'::text THEN joined.game_total::double precision > joined.total
                    WHEN joined.total_play = 'Under'::text THEN joined.game_total::double precision < joined.total
                    ELSE false
                END AS won,
                CASE
                    WHEN joined.total_play = 'Over'::text THEN joined.p_over
                    ELSE joined.p_under
                END -
                CASE
                    WHEN joined.total_play = 'Over'::text THEN
                    CASE
                        WHEN joined.total_over_odds > 0::numeric THEN 100.0 / (joined.total_over_odds + 100.0)
                        ELSE (- joined.total_over_odds) / ((- joined.total_over_odds) + 100.0)
                    END
                    ELSE
                    CASE
                        WHEN joined.total_under_odds > 0::numeric THEN 100.0 / (joined.total_under_odds + 100.0)
                        ELSE (- joined.total_under_odds) / ((- joined.total_under_odds) + 100.0)
                    END
                END AS edge,
            joined.game_total::double precision = joined.total AS push,
            joined.model_version,
            joined.forecast_at,
            joined.start_time,
            row_number() OVER (PARTITION BY joined.game_pk ORDER BY joined.team) AS rn
           FROM joined
          WHERE (joined.total_play = ANY (ARRAY['Over'::text, 'Under'::text])) AND joined.total IS NOT NULL AND joined.game_total IS NOT NULL AND joined.kelly_quarter_total IS NOT NULL AND joined.kelly_quarter_total > 0::numeric AND
                CASE
                    WHEN joined.total_play = 'Over'::text THEN joined.p_over
                    ELSE joined.p_under
                END IS NOT NULL AND
                CASE
                    WHEN joined.total_play = 'Over'::text THEN joined.total_over_odds
                    ELSE joined.total_under_odds
                END IS NOT NULL
        ), tot AS (
         SELECT tot_raw.date,
            tot_raw.team,
            tot_raw.game_pk,
            tot_raw.bet_type,
            tot_raw.stake,
            tot_raw.decimal_odds,
            tot_raw.american_odds,
            tot_raw.totals_side,
            tot_raw.won,
            tot_raw.edge,
            tot_raw.push,
            tot_raw.model_version,
            tot_raw.forecast_at,
            tot_raw.start_time
           FROM tot_raw
          WHERE tot_raw.rn = 1
        )
 SELECT date,
    team,
    game_pk,
    bet_type,
    stake,
    decimal_odds,
    american_odds,
    totals_side,
    won,
    edge,
    stake::double precision *
        CASE
            WHEN won THEN decimal_odds
            WHEN push THEN 1::double precision
            ELSE 0::double precision
        END AS payout,
    push,
    model_version,
    forecast_at,
    start_time,
    forecast_at >= start_time AS written_after_start,
    (EXISTS ( SELECT 1
           FROM mlb.rewritten_forecast_days d
          WHERE d.date = u.date::date)) AS rewritten
   FROM ( SELECT ml.date,
            ml.team,
            ml.game_pk,
            ml.bet_type,
            ml.stake,
            ml.decimal_odds,
            ml.american_odds,
            ml.totals_side,
            ml.won,
            ml.edge,
            ml.push,
            ml.model_version,
            ml.forecast_at,
            ml.start_time
           FROM ml
          WHERE ml.policy_pick OR ml.edge >= (( SELECT r.min_edge
                   FROM mlb.grading_rules r
                  WHERE r.bet_type = 'ml'::text))::double precision
        UNION ALL
         SELECT rl.date,
            rl.team,
            rl.game_pk,
            rl.bet_type,
            rl.stake,
            rl.decimal_odds,
            rl.american_odds,
            rl.totals_side,
            rl.won,
            rl.edge,
            rl.push,
            rl.model_version,
            rl.forecast_at,
            rl.start_time
           FROM rl
          WHERE rl.policy_pick OR rl.edge >= (( SELECT r.min_edge
                   FROM mlb.grading_rules r
                  WHERE r.bet_type = 'rl'::text))::double precision
        UNION ALL
         SELECT tot.date,
            tot.team,
            tot.game_pk,
            tot.bet_type,
            tot.stake,
            tot.decimal_odds,
            tot.american_odds,
            tot.totals_side,
            tot.won,
            tot.edge,
            tot.push,
            tot.model_version,
            tot.forecast_at,
            tot.start_time
           FROM tot
          WHERE tot.edge >= (( SELECT r.min_edge
                   FROM mlb.grading_rules r
                  WHERE r.bet_type = 'total'::text))) u;

COMMIT;
