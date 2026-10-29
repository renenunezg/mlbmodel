BEGIN;

-- One-time data repair for rows written before 20261029120000_mlb_evaluation_integrity.sql. Apply after it.

-- equity_end_units on day, 7d and 30d rows held 1 + that window's P&L, which the site plotted as cumulative units.
-- Every combined-population row now holds 1 + cumulative P&L of the whole ledger through its date, as the
-- nightly job writes from now on (season rows already did).
UPDATE mlb.model_evaluation e
SET equity_end_units = round((1 + coalesce((
      SELECT sum(l.payout - l.stake) FROM mlb.bet_ledger_v l WHERE l.date::date <= e.date), 0))::numeric, 4)
WHERE e.eval_window IN ('day', '7d', '30d', 'season');

-- The 2026-04-30 flag was lost when these rows were re-created on 2026-06-03. A row is flagged when its window
-- contains a rewritten day, matching the nightly job.
UPDATE mlb.model_evaluation e
SET predictions_rewritten = EXISTS (
  SELECT 1 FROM mlb.rewritten_forecast_days d
  WHERE d.date <= e.date
    AND d.date >= CASE e.eval_window
                   WHEN 'day' THEN e.date
                   WHEN '7d' THEN e.date - 6
                   WHEN '30d' THEN e.date - 29
                   ELSE '-infinity'::date
                 END)
WHERE e.eval_window IN ('day', '7d', '30d', 'season');

COMMIT;
