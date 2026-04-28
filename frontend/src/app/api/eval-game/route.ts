import { NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";
import {
  computeBaseRow,
  type EvalRow,
  type EvalWindow,
} from "@/lib/eval";

// Live per-game eval, fired from the games page when a game flips to Final.
// Verifies Final via MLB API, writes the score back to `games`, and recomputes
// today's day/7d/30d/season tally rows in `model_evaluation`. Idempotent.
// The nightly Python batch is the canonical reconciliation path.

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

interface MLBGameStatus {
  gamePk: number;
  status?: { abstractGameState?: string; detailedState?: string };
  teams?: {
    home?: { score?: number; team?: { abbreviation?: string } };
    away?: { score?: number; team?: { abbreviation?: string } };
  };
}

const COUNT_COLUMNS = [
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

function originAllowed(req: Request): boolean {
  const origin = req.headers.get("origin") ?? req.headers.get("referer") ?? "";
  if (!origin) return false;
  const allow = [
    "http://localhost:3000",
    "https://renenunez.work",
    "https://www.renenunez.work",
  ];
  if (allow.some((a) => origin.startsWith(a))) return true;
  return /https:\/\/[a-z0-9-]+\.vercel\.app/.test(origin);
}

export async function POST(req: Request) {
  if (!originAllowed(req)) {
    return NextResponse.json({ error: "forbidden origin" }, { status: 403 });
  }

  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !serviceKey) {
    return NextResponse.json(
      { error: "supabase service-role env not configured" },
      { status: 500 },
    );
  }

  let body: { game_pk?: number };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "bad json" }, { status: 400 });
  }
  const game_pk = body.game_pk;
  if (!game_pk || typeof game_pk !== "number") {
    return NextResponse.json({ error: "missing game_pk" }, { status: 400 });
  }

  const sb = createClient(url, serviceKey, { auth: { persistSession: false } });

  const mlb = await fetchMlbGame(game_pk);
  if (!mlb || mlb.status?.abstractGameState !== "Final") {
    return NextResponse.json({ ok: false, reason: "not final per MLB API" });
  }
  const homeScore = mlb.teams?.home?.score;
  const awayScore = mlb.teams?.away?.score;
  if (homeScore == null || awayScore == null) {
    return NextResponse.json({ ok: false, reason: "missing scores" });
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
      return NextResponse.json(
        { error: `games update failed: ${error.message}` },
        { status: 500 },
      );
    }
  }

  const { data: preds, error: predErr } = await sb
    .from("model_outputs_season")
    .select(
      "game_pk, team, expected_runs, win_prob, ev_flag, run_line_ev_flag, spread, total, total_play",
    );
  if (predErr) {
    return NextResponse.json(
      { error: `predictions query failed: ${predErr.message}` },
      { status: 500 },
    );
  }
  const { data: finals, error: gamesErr } = await sb
    .from("games")
    .select("game_pk, game_date, home_team, away_team, home_score, away_score")
    .eq("status", "Final")
    .not("home_score", "is", null)
    .not("away_score", "is", null);
  if (gamesErr) {
    return NextResponse.json(
      { error: `games query failed: ${gamesErr.message}` },
      { status: 500 },
    );
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
      game_date: g.game_date,
    });
  }

  if (evalRows.length === 0) {
    return NextResponse.json({ ok: true, note: "no eval rows" });
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

  // Partial upsert: only the count columns. Regression / probabilistic /
  // financial columns stay as the nightly batch left them.
  for (const w of windows) {
    if (w.rows.length === 0) continue;
    const base = computeBaseRow(w.rows);
    const partial: Record<string, unknown> = {
      date: evalDate,
      eval_window: w.name,
    };
    for (const col of COUNT_COLUMNS) {
      partial[col] = (base as unknown as Record<string, unknown>)[col];
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
      return NextResponse.json(
        { error: `eval write failed (${w.name}): ${error.message}` },
        { status: 500 },
      );
    }
  }

  return NextResponse.json({
    ok: true,
    game_pk,
    eval_date: evalDate,
    windows_updated: windows.filter((w) => w.rows.length > 0).map((w) => w.name),
  });
}
