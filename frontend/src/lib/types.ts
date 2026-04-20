export interface ModelOutput {
  game_pk: number;
  date: string | null;
  team: string;
  starter: string | null;
  expected_runs: number;
  win_prob: number;
  our_odds: number | null;
  moneyline: number | null;
  total: number | null;
  spread: number | null;
  spread_odds: number | null;
  our_total: number | null;
  total_diff: number | null;
  total_play: string;
  ev_flag: string;
  run_line_ev_flag: string;
  ml_confidence: number | null;
  run_line_confidence: number | null;
  high_variance_flag: string;
}

export interface GameInfo {
  game_pk: number;
  game_date: string;
  home_team: string;
  away_team: string;
  home_score: number | null;
  away_score: number | null;
  status: string;
  venue: string | null;
  start_time: string | null;
}

export interface ModelEvaluation {
  date: string;
  eval_window: string;
  total_correct: number;
  total_predictions: number;
  total_accuracy: number;
  ml_correct: number;
  ml_predictions: number;
  ml_accuracy: number;
  run_line_correct: number;
  run_line_predictions: number;
  run_line_accuracy: number;
  totals_correct: number | null;
  totals_predictions: number | null;
  totals_accuracy: number | null;
  average_total_diff: number;
  average_win_prob: number;
  // Regression metrics
  mae: number | null;
  rmse: number | null;
  mape: number | null;
  r2: number | null;
  // Probabilistic metrics
  brier_score: number | null;
  log_loss: number | null;
  sharpness: number | null;
  interval_coverage_80: number | null;
  // Financial metrics
  roi: number | null;
  sharpe: number | null;
  sortino: number | null;
  max_drawdown: number | null;
  total_staked_units: number | null;
  net_profit_units: number | null;
  equity_end_units: number | null;
}

export interface CalibrationBin {
  date: string;
  bin_mid: number;
  predicted_mean: number;
  observed_rate: number;
  count: number;
}

export interface FeatureImportance {
  date: string;
  feature: string;
  importance: number;
}

export interface EdgeBucket {
  date: string;
  eval_window: string;
  bucket_label: string;
  n_bets: number;
  hit_rate: number;
  roi: number;
}

export interface GameMatchup {
  game_pk: number;
  home_team: string;
  away_team: string;
  venue: string | null;
  start_time: string | null;
  date: string;
  away: ModelOutput;
  home: ModelOutput;
  home_score: number | null;
  away_score: number | null;
  status: string | null;
  // Optional live fields — populated by /api/live-scores polling, not in DB
  current_inning?: number | null;
  inning_state?: string | null;
}
