begin;
alter table mlb.model_evaluation add column if not exists interval_coverage_predictions integer
  check (interval_coverage_predictions >= 0);
comment on column mlb.model_evaluation.interval_coverage_predictions is
  'Number of team forecasts with a recoverable frozen PMF used for interval coverage. NULL denotes legacy surrogate coverage.';
commit;
