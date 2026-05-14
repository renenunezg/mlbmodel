import type { SupabaseClient } from "@supabase/supabase-js";
import {
  computeBaseRow,
  computeSegmentRow,
  type EvalRow,
  type EvalWindow,
} from "@/lib/eval";

// Best-effort live eval triggered when a game flips to Final. Verifies via MLB
// API, writes back the score, and partial-upserts today's evaluation windows.
// Nightly Python batch is the canonical reconciliation.

interface MLBGameStatus {
  gamePk: number;
  status?: { abstractGameState?: string; detailedState?: string };
  teams?: {
    home?: { score?: number; team?: { abbreviation?: string } };
    away?: { score?: number; team?: { abbreviation?: string } };
  };
}

const LIVE_UPSERT_COLUMNS = [
  // count + accuracy
  "total_correct",
  "total_predictions",
  "total_accuracy",
  "ml_correct",
  "ml_predictions",
  "ml_accuracy",
  "run_line_correct",
  "run_line_predictions",
  "run_line_accuracy",
  "totals_correct",
  "totals_predictions",
  "totals_accuracy",
  "average_total_diff",
  "average_win_prob",
  // segment
  "roi_favorites",
  "n_favorites",
  "favorites_correct",
  "roi_underdogs",
  "n_underdogs",
  "underdogs_correct",
  "avg_ml_line",
  "overs_correct",
  "overs_predictions",
  "overs_roi",
  "unders_correct",
  "unders_predictions",
  "unders_roi",
] as const;

function ptDateString(d: Date): string {
  return d.toLocaleDateString("en-CA", { timeZone: "America/Los_Angeles" });
}

function shiftDays(date: string, days: number): string {
  const d = new Date(date + "T00:00:00Z");
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

async function fetchMlbGame(gamePk: number): Promise<MLBGameStatus | null> {
  const url = `https://statsapi.mlb.com/api/v1.1/game/${gamePk}/feed/live`;
  const res = await fetch(url, {
    headers: { "User-Agent": "mlb-model-dashboard" },
    cache: "no-store",
  });
  if (!res.ok) return null;
  const data = await res.json();
  const game = data?.gameData;
  const linescore = data?.liveData?.linescore;
  if (!game) return null;
  return {
    gamePk,
    status: game.status,
    teams: {
      home: {
        score: linescore?.teams?.home?.runs,
        team: { abbreviation: game.teams?.home?.abbreviation },
      },
      away: {
        score: linescore?.teams?.away?.runs,
        team: { abbreviation: game.teams?.away?.abbreviation },
      },
    },
  };
}

export type EvalResult =
  | { ok: true; game_pk: number; eval_date: string; windows_updated: EvalWindow[] }
  | { ok: false; reason: string }
  | { ok: false; error: string };

export async function runEvalForGame(
  sb: SupabaseClient,
  game_pk: number,
): Promise<EvalResult> {
  const mlb = await fetchMlbGame(game_pk);
  if (!mlb || mlb.status?.abstractGameState !== "Final") {
    return { ok: false, reason: "not final per MLB API" };
  }
  const homeScore = mlb.teams?.home?.score;
  const awayScore = mlb.teams?.away?.score;
  if (homeScore == null || awayScore == null) {
    return { ok: false, reason: "missing scores" };
  }

  const { data: existingGame } = await sb
    .from("games")
    .select("status, home_score, away_score")
    .eq("game_pk", game_pk)
    .maybeSingle();

  const needsGameWrite =
    !existingGame ||
    existingGame.status !== "Final" ||
    existingGame.home_score !== homeScore ||
    existingGame.away_score !== awayScore;

  if (needsGameWrite) {
    const { error } = await sb
      .from("games")
      .update({
        status: "Final",
        home_score: homeScore,
        away_score: awayScore,
      })
      .eq("game_pk", game_pk);
    if (error) {
      return { ok: false, error: `games update failed: ${error.message}` };
    }
  }

  const { data: preds, error: predErr } = await sb
    .from("model_outputs_season_unified")
    .select(
      "game_pk, team, expected_runs, win_prob, ev_flag, run_line_ev_flag, spread, total, total_play, moneyline, kelly_quarter_ml, kelly_quarter_total, total_over_odds, total_under_odds",
    );
  if (predErr) {
    return { ok: false, error: `predictions query failed: ${predErr.message}` };
  }
  const { data: finals, error: gamesErr } = await sb
    .from("games")
    .select("game_pk, game_date, home_team, away_team, home_score, away_score")
    .eq("status", "Final")
    .not("home_score", "is", null)
    .not("away_score", "is", null);
  if (gamesErr) {
    return { ok: false, error: `games query failed: ${gamesErr.message}` };
  }

  type GameRow = {
    game_pk: number;
    game_date: string;
    home_team: string;
    away_team: string;
    home_score: number;
    away_score: number;
  };
  const gameByPk = new Map<number, GameRow>();
  for (const g of (finals ?? []) as GameRow[]) gameByPk.set(g.game_pk, g);

  type Pred = {
    game_pk: number;
    team: string;
    expected_runs: number;
    win_prob: number;
    ev_flag: string;
    run_line_ev_flag: string;
    spread: number | null;
    total: number | null;
    total_play: string;
    moneyline: number | null;
    kelly_quarter_ml: number | null;
    kelly_quarter_total: number | null;
    total_over_odds: number | null;
    total_under_odds: number | null;
  };

  const evalRows: (EvalRow & { game_date: string })[] = [];
  for (const p of (preds ?? []) as Pred[]) {
    const g = gameByPk.get(p.game_pk);
    if (!g) continue;
    const isHome = g.home_team === p.team;
    if (!isHome && g.away_team !== p.team) continue;
    const teamScore = isHome ? g.home_score : g.away_score;
    const oppScore = isHome ? g.away_score : g.home_score;
    evalRows.push({
      game_pk: p.game_pk,
      team: p.team,
      expected_runs: p.expected_runs,
      win_prob: p.win_prob,
      ev_flag: p.ev_flag,
      run_line_ev_flag: p.run_line_ev_flag,
      spread: p.spread,
      total: p.total,
      total_play: p.total_play,
      actual_runs: teamScore,
      actual_win: teamScore > oppScore ? 1 : 0,
      actual_margin: teamScore - oppScore,
      game_total: g.home_score + g.away_score,
      moneyline: p.moneyline,
      kelly_quarter_ml: p.kelly_quarter_ml,
      kelly_quarter_total: p.kelly_quarter_total,
      total_over_odds: p.total_over_odds,
      total_under_odds: p.total_under_odds,
      game_date: g.game_date,
    });
  }

  if (evalRows.length === 0) {
    return { ok: true, game_pk, eval_date: ptDateString(new Date()), windows_updated: [] };
  }

  const evalDate = ptDateString(new Date());
  const latestDate = evalRows.reduce(
    (mx, r) => (r.game_date > mx ? r.game_date : mx),
    evalRows[0].game_date,
  );

  const windows: Array<{ name: EvalWindow; rows: EvalRow[] }> = [
    { name: "day", rows: evalRows.filter((r) => r.game_date === latestDate) },
    { name: "7d", rows: evalRows.filter((r) => r.game_date >= shiftDays(latestDate, -7)) },
    { name: "30d", rows: evalRows.filter((r) => r.game_date >= shiftDays(latestDate, -30)) },
    { name: "season", rows: evalRows },
  ];

  for (const w of windows) {
    if (w.rows.length === 0) continue;
    const base = computeBaseRow(w.rows);
    const seg = computeSegmentRow(w.rows);
    const merged = { ...base, ...seg } as unknown as Record<string, unknown>;
    const partial: Record<string, unknown> = {
      date: evalDate,
      eval_window: w.name,
    };
    for (const col of LIVE_UPSERT_COLUMNS) {
      partial[col] = merged[col];
    }

    const { data: existing } = await sb
      .from("model_evaluation")
      .select("date")
      .eq("date", evalDate)
      .eq("eval_window", w.name)
      .maybeSingle();

    const { error } = existing
      ? await sb
          .from("model_evaluation")
          .update(partial)
          .eq("date", evalDate)
          .eq("eval_window", w.name)
      : await sb.from("model_evaluation").insert(partial);

    if (error) {
      return { ok: false, error: `eval write failed (${w.name}): ${error.message}` };
    }
  }

  return {
    ok: true,
    game_pk,
    eval_date: evalDate,
    windows_updated: windows.filter((w) => w.rows.length > 0).map((w) => w.name),
  };
}
