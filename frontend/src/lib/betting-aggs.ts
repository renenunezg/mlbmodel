import type { BetLedgerRow, LiveKpis } from "./types";

// Mirrors backend/metrics.py::financial_summary + segment_summary so the
// Performance betting tab computes the same numbers live as the nightly
// model_evaluation snapshot, without depending on it.

const ANNUAL_FACTOR = 252;

function impliedFromAmerican(odds: number): number {
  return odds > 0 ? 100 / (odds + 100) : -odds / (-odds + 100);
}

function safeRound(x: number, decimals: number): number {
  const f = 10 ** decimals;
  return Math.round(x * f) / f;
}

function daysReturns(rows: BetLedgerRow[]): number[] {
  const byDate = new Map<string, { stake: number; pnl: number }>();
  for (const r of rows) {
    const d = r.date.slice(0, 10);
    const cur = byDate.get(d) ?? { stake: 0, pnl: 0 };
    cur.stake += r.stake;
    cur.pnl += r.payout - r.stake;
    byDate.set(d, cur);
  }
  return [...byDate.values()].map((v) => (v.stake > 0 ? v.pnl / v.stake : 0));
}

function stdev(xs: number[]): number {
  if (xs.length < 2) return 0;
  const mu = xs.reduce((a, b) => a + b, 0) / xs.length;
  const variance =
    xs.reduce((a, b) => a + (b - mu) * (b - mu), 0) / (xs.length - 1);
  return Math.sqrt(variance);
}

function mean(xs: number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}

function maxDrawdown(rows: BetLedgerRow[]): number | null {
  if (rows.length < 2) return null;
  const byDate = new Map<string, number>();
  for (const r of rows) {
    const d = r.date.slice(0, 10);
    byDate.set(d, (byDate.get(d) ?? 0) + (r.payout - r.stake));
  }
  const dates = [...byDate.keys()].sort();
  if (dates.length < 2) return null;
  let equity = 1.0;
  let peak = 1.0;
  let worst = 0;
  for (const d of dates) {
    equity += byDate.get(d) ?? 0;
    if (equity > peak) peak = equity;
    const dd = equity - peak;
    if (dd < worst) worst = dd;
  }
  return safeRound(worst, 4);
}

export function aggregateLedger(rows: BetLedgerRow[]): LiveKpis {
  const empty: LiveKpis = {
    roi: null,
    sharpe: null,
    sortino: null,
    max_drawdown: null,
    total_staked_units: 0,
    net_profit_units: 0,
    roi_favorites: null,
    n_favorites: 0,
    favorites_correct: 0,
    roi_underdogs: null,
    n_underdogs: 0,
    underdogs_correct: 0,
    roi_run_line: null,
    n_run_line: 0,
    run_line_bets_correct: 0,
    avg_ml_line: null,
    overs_correct: 0,
    overs_predictions: 0,
    overs_roi: null,
    unders_correct: 0,
    unders_predictions: 0,
    unders_roi: null,
  };
  if (!rows.length) return empty;

  const totalStake = rows.reduce((a, r) => a + r.stake, 0);
  const totalPayout = rows.reduce((a, r) => a + r.payout, 0);
  const net = totalPayout - totalStake;
  const roi = totalStake > 0 ? safeRound(net / totalStake, 4) : null;

  const dr = daysReturns(rows);
  let sharpe: number | null = null;
  let sortino: number | null = null;
  if (dr.length >= 2) {
    const mu = mean(dr);
    const sd = stdev(dr);
    if (sd > 0) sharpe = safeRound((mu / sd) * Math.sqrt(ANNUAL_FACTOR), 4);
    const downside = dr.filter((x) => x < 0);
    if (downside.length >= 1) {
      const dsd = stdev(downside);
      if (dsd > 0)
        sortino = safeRound((mu / dsd) * Math.sqrt(ANNUAL_FACTOR), 4);
    }
  }

  const mlRows = rows.filter((r) => r.bet_type === "ml" && r.american_odds != null);
  const favRows = mlRows.filter((r) => (r.american_odds as number) < 0);
  const dogRows = mlRows.filter((r) => (r.american_odds as number) > 0);

  function segRoi(sub: BetLedgerRow[]): number | null {
    const s = sub.reduce((a, r) => a + r.stake, 0);
    if (s <= 0) return null;
    const p = sub.reduce((a, r) => a + r.payout, 0);
    return safeRound((p - s) / s, 4);
  }

  let avgMlLine: number | null = null;
  if (mlRows.length) {
    const meanImplied =
      mlRows.reduce((a, r) => a + impliedFromAmerican(r.american_odds as number), 0) /
      mlRows.length;
    const avgAmerican =
      meanImplied >= 0.5
        ? (-100 * meanImplied) / (1 - meanImplied)
        : (100 * (1 - meanImplied)) / meanImplied;
    avgMlLine = safeRound(avgAmerican, 2);
  }

  const rlRows = rows.filter((r) => r.bet_type === "rl");
  const overRows = rows.filter((r) => r.totals_side === "over");
  const underRows = rows.filter((r) => r.totals_side === "under");

  return {
    roi,
    sharpe,
    sortino,
    max_drawdown: maxDrawdown(rows),
    total_staked_units: safeRound(totalStake, 4),
    net_profit_units: safeRound(net, 4),
    roi_favorites: segRoi(favRows),
    n_favorites: favRows.length,
    favorites_correct: favRows.filter((r) => r.won).length,
    roi_underdogs: segRoi(dogRows),
    n_underdogs: dogRows.length,
    underdogs_correct: dogRows.filter((r) => r.won).length,
    roi_run_line: segRoi(rlRows),
    n_run_line: rlRows.length,
    run_line_bets_correct: rlRows.filter((r) => r.won).length,
    avg_ml_line: avgMlLine,
    overs_correct: overRows.filter((r) => r.won).length,
    overs_predictions: overRows.length,
    overs_roi: segRoi(overRows),
    unders_correct: underRows.filter((r) => r.won).length,
    unders_predictions: underRows.length,
    unders_roi: segRoi(underRows),
  };
}
