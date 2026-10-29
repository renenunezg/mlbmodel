BEGIN;

-- Evaluation integrity: graded populations a reader can query, integrity flags instead of deletions, the
-- grading rules as data, a market baseline on identical games, intervals and n, and a v2-only season window.
-- Apply before promoting the evaluate_model.py change that writes these columns.

-- Days whose graded forecasts were replaced after the games started. 2026-04-30: a nightly-eval hand-run at
-- 22:15 PT rewrote that day's v1 predictions with the bullpen-rest feature set (commit fac316a). The rows stay
-- graded; the ledger and the evaluation mark them so a reader can exclude them.
CREATE TABLE mlb.rewritten_forecast_days (
  date date PRIMARY KEY,
  note text NOT NULL,
  recorded_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE mlb.rewritten_forecast_days ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON mlb.rewritten_forecast_days FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON mlb.rewritten_forecast_days TO anon, authenticated;
GRANT ALL ON mlb.rewritten_forecast_days TO service_role;
INSERT INTO mlb.rewritten_forecast_days (date, note) VALUES
  ('2026-04-30', 'v1 predictions rewritten after the games by a 22:15 PT hand-run with the bullpen-rest feature set (fac316a)');

-- The grading rule for each market, read by bet_ledger_v and by the site's provisional live grading. The nightly
-- evaluation refuses to run when these differ from backend/strategy.py EV_THRESHOLDS.
CREATE TABLE mlb.grading_rules (
  bet_type text PRIMARY KEY CHECK (bet_type IN ('ml', 'rl', 'total')),
  min_edge numeric NOT NULL CHECK (min_edge > 0),
  stake_column text NOT NULL,
  stake_rule text NOT NULL,
  push_rule text NOT NULL
);
ALTER TABLE mlb.grading_rules ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON mlb.grading_rules FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON mlb.grading_rules TO anon, authenticated;
GRANT ALL ON mlb.grading_rules TO service_role;
INSERT INTO mlb.grading_rules (bet_type, min_edge, stake_column, stake_rule, push_rule) VALUES
  ('ml', 0.045, 'kelly_quarter_ml',
   'Graded only when the flag names the team, the price exists, quarter Kelly > 0 and win_prob minus the vig-inclusive implied probability >= min_edge. Stake is quarter Kelly of a fixed 1u bankroll; no floor.',
   'No push; a moneyline game always has a winner.'),
  ('rl', 0.045, 'kelly_quarter_rl',
   'Graded only for spread +1.5 or -1.5 when the flag names the team, quarter Kelly > 0 and p_cover minus the vig-inclusive implied probability >= min_edge. Stake is quarter Kelly; no floor.',
   'No push; a 1.5-run line cannot land exactly.'),
  ('total', 0.065, 'kelly_quarter_total',
   'Graded once per game when total_play is Over or Under, quarter Kelly > 0 and p_over or p_under minus the vig-inclusive implied probability >= min_edge. Stake is quarter Kelly; no floor.',
   'Push when the game total equals the line: the stake is returned, it counts as a bet and is excluded from the hit rate.');

-- Same grading as before, with the thresholds read from grading_rules and five columns appended:
-- model_version (v1 before 2026-05-12, v2 after), forecast_at (v2 prediction_updated_at; v1 stored none),
-- start_time (the game's scheduled first pitch), written_after_start (forecast_at >= start_time; NULL for v1)
-- and rewritten (the forecast date is in rewritten_forecast_days).
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
            NULL::timestamp with time zone AS forecast_at
           FROM mlb.model_outputs_season_v1_archive
          WHERE ((model_outputs_season_v1_archive.date)::date < '2026-05-12'::date)
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
            model_outputs_season.prediction_updated_at AS forecast_at
           FROM mlb.model_outputs_season
          WHERE ((model_outputs_season.date)::date >= '2026-05-12'::date)
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
            g.start_time,
            g.home_team,
            g.home_score,
            g.away_score,
            (g.home_score + g.away_score) AS game_total,
                CASE
                    WHEN ((g.home_team)::text = m.team) THEN g.home_score
                    ELSE g.away_score
                END AS team_score,
                CASE
                    WHEN ((g.home_team)::text = m.team) THEN g.away_score
                    ELSE g.home_score
                END AS opp_score,
                CASE
                    WHEN ((g.home_team)::text = m.team) THEN (g.home_score - g.away_score)
                    ELSE (g.away_score - g.home_score)
                END AS margin
           FROM (preds m
             JOIN mlb.games g ON (((g.game_pk = m.game_pk) AND (g.game_date = (m.date)::date))))
          WHERE (((g.status)::text = 'Final'::text) AND (g.home_score IS NOT NULL) AND (g.away_score IS NOT NULL))
        ), ml AS (
         SELECT joined.date,
            joined.team,
            joined.game_pk,
            'ml'::text AS bet_type,
            joined.kelly_quarter_ml AS stake,
                CASE
                    WHEN (joined.moneyline > (0)::double precision) THEN ((1)::double precision + (joined.moneyline / (100.0)::double precision))
                    ELSE ((1)::double precision + ((100.0)::double precision / abs(joined.moneyline)))
                END AS decimal_odds,
            (joined.moneyline)::numeric AS american_odds,
            NULL::text AS totals_side,
            (joined.team_score > joined.opp_score) AS won,
            (joined.win_prob -
                CASE
                    WHEN (joined.moneyline > (0)::double precision) THEN ((100.0)::double precision / (joined.moneyline + (100.0)::double precision))
                    ELSE ((- joined.moneyline) / ((- joined.moneyline) + (100.0)::double precision))
                END) AS edge,
            false AS push,
            joined.model_version,
            joined.forecast_at,
            joined.start_time
           FROM joined
          WHERE ((joined.ev_flag = joined.team) AND (joined.moneyline IS NOT NULL) AND (joined.kelly_quarter_ml IS NOT NULL) AND (joined.kelly_quarter_ml > (0)::numeric) AND (joined.win_prob IS NOT NULL))
        ), rl AS (
         SELECT joined.date,
            joined.team,
            joined.game_pk,
            'rl'::text AS bet_type,
            joined.kelly_quarter_rl AS stake,
                CASE
                    WHEN (joined.spread_odds > (0)::double precision) THEN ((1)::double precision + (joined.spread_odds / (100.0)::double precision))
                    ELSE ((1)::double precision + ((100.0)::double precision / abs(joined.spread_odds)))
                END AS decimal_odds,
            NULL::numeric AS american_odds,
            NULL::text AS totals_side,
                CASE
                    WHEN (joined.spread < (0)::double precision) THEN ((joined.margin)::double precision >= abs(joined.spread))
                    WHEN (joined.spread > (0)::double precision) THEN ((joined.team_score > joined.opp_score) OR ((joined.margin)::double precision >= (- joined.spread)))
                    ELSE false
                END AS won,
            ((joined.p_cover)::double precision -
                CASE
                    WHEN (joined.spread_odds > (0)::double precision) THEN ((100.0)::double precision / (joined.spread_odds + (100.0)::double precision))
                    ELSE ((- joined.spread_odds) / ((- joined.spread_odds) + (100.0)::double precision))
                END) AS edge,
            false AS push,
            joined.model_version,
            joined.forecast_at,
            joined.start_time
           FROM joined
          WHERE ((joined.run_line_ev_flag = joined.team) AND (joined.spread IS NOT NULL) AND (joined.spread_odds IS NOT NULL) AND (joined.kelly_quarter_rl IS NOT NULL) AND (joined.kelly_quarter_rl > (0)::numeric) AND (joined.p_cover IS NOT NULL))
        ), tot_raw AS (
         SELECT joined.date,
            joined.team,
            joined.game_pk,
            'total'::text AS bet_type,
            joined.kelly_quarter_total AS stake,
            lower(joined.total_play) AS totals_side,
                CASE
                    WHEN (joined.total_play = 'Over'::text) THEN
                    CASE
                        WHEN (joined.total_over_odds > (0)::numeric) THEN ((1)::numeric + (joined.total_over_odds / 100.0))
                        ELSE ((1)::numeric + (100.0 / abs(joined.total_over_odds)))
                    END
                    ELSE
                    CASE
                        WHEN (joined.total_under_odds > (0)::numeric) THEN ((1)::numeric + (joined.total_under_odds / 100.0))
                        ELSE ((1)::numeric + (100.0 / abs(joined.total_under_odds)))
                    END
                END AS decimal_odds,
            NULL::numeric AS american_odds,
                CASE
                    WHEN (joined.total_play = 'Over'::text) THEN ((joined.game_total)::double precision > joined.total)
                    WHEN (joined.total_play = 'Under'::text) THEN ((joined.game_total)::double precision < joined.total)
                    ELSE false
                END AS won,
            (
                CASE
                    WHEN (joined.total_play = 'Over'::text) THEN joined.p_over
                    ELSE joined.p_under
                END -
                CASE
                    WHEN (joined.total_play = 'Over'::text) THEN
                    CASE
                        WHEN (joined.total_over_odds > (0)::numeric) THEN (100.0 / (joined.total_over_odds + 100.0))
                        ELSE ((- joined.total_over_odds) / ((- joined.total_over_odds) + 100.0))
                    END
                    ELSE
                    CASE
                        WHEN (joined.total_under_odds > (0)::numeric) THEN (100.0 / (joined.total_under_odds + 100.0))
                        ELSE ((- joined.total_under_odds) / ((- joined.total_under_odds) + 100.0))
                    END
                END) AS edge,
            ((joined.game_total)::double precision = joined.total) AS push,
            joined.model_version,
            joined.forecast_at,
            joined.start_time,
            row_number() OVER (PARTITION BY joined.game_pk ORDER BY joined.team) AS rn
           FROM joined
          WHERE ((joined.total_play = ANY (ARRAY['Over'::text, 'Under'::text])) AND (joined.total IS NOT NULL) AND (joined.game_total IS NOT NULL) AND (joined.kelly_quarter_total IS NOT NULL) AND (joined.kelly_quarter_total > (0)::numeric) AND (
                CASE
                    WHEN (joined.total_play = 'Over'::text) THEN joined.p_over
                    ELSE joined.p_under
                END IS NOT NULL) AND (
                CASE
                    WHEN (joined.total_play = 'Over'::text) THEN joined.total_over_odds
                    ELSE joined.total_under_odds
                END IS NOT NULL))
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
          WHERE (tot_raw.rn = 1)
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
    ((stake)::double precision *
        CASE
            WHEN won THEN decimal_odds
            WHEN push THEN (1)::double precision
            ELSE (0)::double precision
        END) AS payout,
    push,
    model_version,
    forecast_at,
    start_time,
    (forecast_at >= start_time) AS written_after_start,
    (EXISTS ( SELECT 1
           FROM mlb.rewritten_forecast_days d
          WHERE (d.date = (u.date)::date))) AS rewritten
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
          WHERE (ml.edge >= ( SELECT r.min_edge FROM mlb.grading_rules r WHERE (r.bet_type = 'ml'::text)))
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
          WHERE (rl.edge >= ( SELECT r.min_edge FROM mlb.grading_rules r WHERE (r.bet_type = 'rl'::text)))
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
          WHERE (tot.edge >= ( SELECT r.min_edge FROM mlb.grading_rules r WHERE (r.bet_type = 'total'::text)))) u;

COMMENT ON COLUMN mlb.bet_ledger_v.written_after_start IS
  'True when the stored forecast was published at or after the scheduled first pitch. NULL for v1, which stored no publish time.';
COMMENT ON COLUMN mlb.bet_ledger_v.rewritten IS
  'True when the forecast date is in mlb.rewritten_forecast_days (forecasts replaced after the games started).';

-- Nightly evaluation additions. Rows with eval_window = 'season_v2' cover v2 forecasts only (dates on or after
-- 2026-05-12); every other window keeps the combined v1 and v2 population.
ALTER TABLE mlb.model_evaluation
  ADD COLUMN scored_games integer CHECK (scored_games >= 0),
  ADD COLUMN n_bets integer CHECK (n_bets >= 0),
  ADD COLUMN brier_ci95_low numeric,
  ADD COLUMN brier_ci95_high numeric,
  ADD COLUMN total_accuracy_ci95_low numeric,
  ADD COLUMN total_accuracy_ci95_high numeric,
  ADD COLUMN roi_ci95_low numeric,
  ADD COLUMN roi_ci95_high numeric,
  ADD COLUMN roi_favorites_ci95_low numeric,
  ADD COLUMN roi_favorites_ci95_high numeric,
  ADD COLUMN roi_underdogs_ci95_low numeric,
  ADD COLUMN roi_underdogs_ci95_high numeric,
  ADD COLUMN roi_run_line_ci95_low numeric,
  ADD COLUMN roi_run_line_ci95_high numeric,
  ADD COLUMN overs_roi_ci95_low numeric,
  ADD COLUMN overs_roi_ci95_high numeric,
  ADD COLUMN unders_roi_ci95_low numeric,
  ADD COLUMN unders_roi_ci95_high numeric,
  ADD COLUMN paired_games integer CHECK (paired_games >= 0),
  ADD COLUMN paired_brier numeric,
  ADD COLUMN paired_log_loss numeric,
  ADD COLUMN market_brier numeric,
  ADD COLUMN market_log_loss numeric,
  ADD COLUMN brier_vs_market_ci95_low numeric,
  ADD COLUMN brier_vs_market_ci95_high numeric,
  ADD COLUMN calibration_bins jsonb,
  ADD COLUMN rewritten_bets integer CHECK (rewritten_bets >= 0),
  ADD COLUMN late_forecast_games integer CHECK (late_forecast_games >= 0),
  ADD COLUMN late_forecast_bets integer CHECK (late_forecast_bets >= 0),
  ADD COLUMN excluded_games jsonb;

COMMENT ON COLUMN mlb.model_evaluation.eval_window IS
  'day, 7d, 30d and season cover v1 and v2 together; season_v2 is season to date for v2 forecasts only.';
COMMENT ON COLUMN mlb.model_evaluation.scored_games IS
  'Games behind brier_score, log_loss, mae and calibration (each game contributes both team rows).';
COMMENT ON COLUMN mlb.model_evaluation.n_bets IS 'Graded bets behind roi and the P&L columns.';
COMMENT ON COLUMN mlb.model_evaluation.brier_ci95_low IS
  'All *_ci95_* columns: 2.5th and 97.5th percentiles of 2,000 bootstrap resamples of games (bets on one game resample together).';
COMMENT ON COLUMN mlb.model_evaluation.paired_games IS
  'Games with both sides'' moneylines stored on the forecast, the population for paired_* and market_* columns.';
COMMENT ON COLUMN mlb.model_evaluation.paired_brier IS 'Model Brier score on the paired_games population.';
COMMENT ON COLUMN mlb.model_evaluation.paired_log_loss IS 'Model log loss on the paired_games population.';
COMMENT ON COLUMN mlb.model_evaluation.market_brier IS
  'Brier score of the de-vigged pregame market on paired_games: home and away prices stored with the forecast, proportionally normalized.';
COMMENT ON COLUMN mlb.model_evaluation.market_log_loss IS 'Log loss of the same de-vigged market probabilities.';
COMMENT ON COLUMN mlb.model_evaluation.brier_vs_market_ci95_low IS
  'Bootstrap interval of paired_brier minus market_brier; above zero means the market scored better.';
COMMENT ON COLUMN mlb.model_evaluation.calibration_bins IS
  'Ten win-probability bins for this window: [{bin_mid, predicted_mean, observed_rate, count}], count in team rows.';
COMMENT ON COLUMN mlb.model_evaluation.equity_end_units IS
  '1 plus cumulative P&L in units from the first graded bet through this date, for the window''s model population.';
COMMENT ON COLUMN mlb.model_evaluation.rewritten_bets IS 'Graded bets in the window whose forecasts were rewritten after the games.';
COMMENT ON COLUMN mlb.model_evaluation.late_forecast_games IS 'Graded games in the window whose stored forecast was published at or after first pitch.';
COMMENT ON COLUMN mlb.model_evaluation.late_forecast_bets IS 'Graded bets in the window on those games.';
COMMENT ON COLUMN mlb.model_evaluation.excluded_games IS
  'Final games in the window without a graded forecast, counted by reason: no_forecast, v1_outage_hindsight_only, forecast_dated_other_day.';
COMMENT ON COLUMN mlb.model_evaluation.predictions_rewritten IS
  'True when the window includes a date in mlb.rewritten_forecast_days.';
COMMENT ON COLUMN mlb.model_evaluation.interval_coverage_50 IS
  'Share of team rows (not games, not game totals) whose own runs fell inside the central 50% interval of that team''s frozen run PMF. Integer runs with inclusive endpoints cover at least the nominal level by construction.';
COMMENT ON COLUMN mlb.model_evaluation.interval_coverage_80 IS 'As interval_coverage_50, for the central 80% interval, per team row.';
COMMENT ON COLUMN mlb.model_evaluation.interval_coverage_90 IS 'As interval_coverage_50, for the central 90% interval, per team row.';
COMMENT ON COLUMN mlb.model_evaluation.interval_coverage_predictions IS
  'Team rows (two per game) with a recoverable frozen PMF used for interval coverage. NULL denotes legacy surrogate coverage.';

NOTIFY pgrst, 'reload schema';

COMMIT;
