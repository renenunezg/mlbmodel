// Per-game tally math, ported from backend/evaluate_model.py:_compute_base_row.
// Only covers the count columns shown on the Evaluation History table.
// Regression / probabilistic / financial metrics stay with the nightly batch.

export type EvalRow = {
  game_pk: number;
  team: string;
  expected_runs: number;
  win_prob: number;
  ev_flag: string;
  run_line_ev_flag: string;
  spread: number | null;
  total: number | null;
  total_play: string;
  actual_runs: number;
  actual_win: 0 | 1;
  actual_margin: number;
  game_total: number;
  moneyline?: number | null;
  kelly_quarter_ml?: number | null;
  kelly_quarter_total?: number | null;
  total_over_odds?: number | null;
  total_under_odds?: number | null;
};

export type SegmentRow = {
  roi_favorites: number | null;
  n_favorites: number;
  favorites_correct: number;
  roi_underdogs: number | null;
  n_underdogs: number;
  underdogs_correct: number;
  avg_ml_line: number | null;
  overs_correct: number;
  overs_predictions: number;
  overs_roi: number | null;
  unders_correct: number;
  unders_predictions: number;
  unders_roi: number | null;
};

export type BaseRow = {
  total_correct: number;
  total_predictions: number;
  total_accuracy: number | null;
  ml_correct: number;
  ml_predictions: number;
  ml_accuracy: number | null;
  run_line_correct: number;
  run_line_predictions: number;
  run_line_accuracy: number | null;
  totals_correct: number;
  totals_predictions: number;
  totals_accuracy: number | null;
  average_total_diff: number | null;
  average_win_prob: number | null;
};

export type EvalWindow = "day" | "7d" | "30d" | "season";

function dedupeByGamePk<T extends { game_pk: number }>(rows: T[]): T[] {
  const seen = new Set<number>();
  const out: T[] = [];
  for (const r of rows) {
    if (seen.has(r.game_pk)) continue;
    seen.add(r.game_pk);
    out.push(r);
  }
  return out;
}

export function calcRunLinePick(row: EvalRow): 0 | 1 | null {
  if (row.spread == null) return null;
  const spread = row.spread;
  const margin = row.actual_margin;
  if (Number.isNaN(margin)) return null;
  if (spread < 0) return margin >= Math.abs(spread) ? 1 : 0;
  if (spread > 0) return row.actual_win === 1 || margin >= -spread ? 1 : 0;
  return null;
}

export function calcTotalPick(row: EvalRow): 0 | 1 | null {
  if (row.total == null) return null;
  const actual = row.game_total;
  if (Number.isNaN(actual)) return null;
  const dir = row.total_play.trim().toLowerCase();
  if (actual === row.total) return null;
  if (dir === "over") return actual > row.total ? 1 : 0;
  if (dir === "under") return actual < row.total ? 1 : 0;
  return null;
}

function round4(n: number): number {
  return Math.round(n * 10000) / 10000;
}

export function computeBaseRow(rows: EvalRow[]): BaseRow {
  const total_predictions = rows.length;
  const total_correct = rows.reduce(
    (acc, r) => acc + ((r.win_prob > 0.5 ? 1 : 0) === r.actual_win ? 1 : 0),
    0,
  );
  const total_accuracy =
    total_predictions > 0 ? round4(total_correct / total_predictions) : null;

  const runs_mae =
    total_predictions > 0
      ? rows.reduce((acc, r) => acc + Math.abs(r.expected_runs - r.actual_runs), 0) /
        total_predictions
      : null;

  const win_prob_mean =
    total_predictions > 0
      ? rows.reduce((acc, r) => acc + r.win_prob, 0) / total_predictions
      : null;

  const ml = dedupeByGamePk(rows.filter((r) => r.ev_flag === r.team));
  const ml_correct = ml.reduce((a, r) => a + (r.actual_win === 1 ? 1 : 0), 0);

  const rl = dedupeByGamePk(rows.filter((r) => r.run_line_ev_flag === r.team));
  let rl_correct = 0;
  let rl_total = 0;
  for (const r of rl) {
    const v = calcRunLinePick(r);
    if (v == null) continue;
    rl_total += 1;
    rl_correct += v;
  }

  const tot = dedupeByGamePk(
    rows.filter((r) => r.total_play === "Over" || r.total_play === "Under"),
  );
  let tot_correct = 0;
  let tot_total = 0;
  for (const r of tot) {
    const v = calcTotalPick(r);
    if (v == null) continue;
    tot_total += 1;
    tot_correct += v;
  }

  return {
    total_correct,
    total_predictions,
    total_accuracy,
    ml_correct,
    ml_predictions: ml.length,
    ml_accuracy: ml.length > 0 ? round4(ml_correct / ml.length) : null,
    run_line_correct: rl_correct,
    run_line_predictions: rl_total,
    run_line_accuracy: rl_total > 0 ? round4(rl_correct / rl_total) : null,
    totals_correct: tot_correct,
    totals_predictions: tot_total,
    totals_accuracy: tot_total > 0 ? round4(tot_correct / tot_total) : null,
    average_total_diff: runs_mae != null ? round4(runs_mae) : null,
    average_win_prob: win_prob_mean != null ? round4(win_prob_mean) : null,
  };
}

function americanToDecimal(american: number): number {
  return american > 0 ? 1 + american / 100 : 1 + 100 / Math.abs(american);
}

export function computeSegmentRow(rows: EvalRow[]): SegmentRow {
  const out: SegmentRow = {
    roi_favorites: null,
    n_favorites: 0,
    favorites_correct: 0,
    roi_underdogs: null,
    n_underdogs: 0,
    underdogs_correct: 0,
    avg_ml_line: null,
    overs_correct: 0,
    overs_predictions: 0,
    overs_roi: null,
    unders_correct: 0,
    unders_predictions: 0,
    unders_roi: null,
  };

  const mlBets = dedupeByGamePk(
    rows.filter((r) => r.ev_flag === r.team && r.moneyline != null),
  );
  if (mlBets.length > 0) {
    let mlSum = 0;
    let favStake = 0,
      favPayout = 0,
      favN = 0,
      favWins = 0;
    let dogStake = 0,
      dogPayout = 0,
      dogN = 0,
      dogWins = 0;
    for (const r of mlBets) {
      const am = r.moneyline as number;
      mlSum += am;
      const dec = americanToDecimal(am);
      let stake = r.kelly_quarter_ml ?? 0;
      if (stake <= 0) stake = 0.01;
      const won = r.actual_win === 1;
      const payout = won ? stake * dec : 0;
      if (am < 0) {
        favN += 1;
        favStake += stake;
        favPayout += payout;
        if (won) favWins += 1;
      } else if (am > 0) {
        dogN += 1;
        dogStake += stake;
        dogPayout += payout;
        if (won) dogWins += 1;
      }
    }
    out.n_favorites = favN;
    out.favorites_correct = favWins;
    out.n_underdogs = dogN;
    out.underdogs_correct = dogWins;
    out.roi_favorites =
      favStake > 0 ? round4((favPayout - favStake) / favStake) : null;
    out.roi_underdogs =
      dogStake > 0 ? round4((dogPayout - dogStake) / dogStake) : null;
    out.avg_ml_line = Math.round((mlSum / mlBets.length) * 100) / 100;
  }

  const totalBets = dedupeByGamePk(
    rows.filter((r) => r.total_play === "Over" || r.total_play === "Under"),
  );
  let oN = 0,
    oWins = 0,
    oStake = 0,
    oPayout = 0;
  let uN = 0,
    uWins = 0,
    uStake = 0,
    uPayout = 0;
  for (const r of totalBets) {
    const dir = r.total_play.toLowerCase();
    const v = calcTotalPick(r);
    if (v == null) continue;
    const am = dir === "over" ? r.total_over_odds : r.total_under_odds;
    if (am == null) continue;
    const dec = americanToDecimal(am);
    let stake = r.kelly_quarter_total ?? 0;
    if (stake <= 0) stake = 0.01;
    const payout = v === 1 ? stake * dec : 0;
    if (dir === "over") {
      oN += 1;
      oWins += v;
      oStake += stake;
      oPayout += payout;
    } else {
      uN += 1;
      uWins += v;
      uStake += stake;
      uPayout += payout;
    }
  }
  out.overs_predictions = oN;
  out.overs_correct = oWins;
  out.overs_roi = oStake > 0 ? round4((oPayout - oStake) / oStake) : null;
  out.unders_predictions = uN;
  out.unders_correct = uWins;
  out.unders_roi = uStake > 0 ? round4((uPayout - uStake) / uStake) : null;

  return out;
}
