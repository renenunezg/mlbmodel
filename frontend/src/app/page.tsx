import { supabase } from "@/lib/supabase";
import { TableOfContents } from "./toc";
import { MethodologyContent, type DistributionGameData } from "./methodology-content";

export const revalidate = 1800;

interface PredRow {
  game_pk: number;
  date: string | null;
  team: string;
  expected_runs: number;
  expected_runs_p10: number | null;
  expected_runs_p50: number | null;
  expected_runs_p90: number | null;
  total: number | null;
  total_p10: number | null;
  total_p50: number | null;
  total_p90: number | null;
  win_prob: number;
  win_prob_p10: number | null;
  win_prob_p90: number | null;
  runs_hist: number[] | null;
}

interface GameRow {
  game_pk: number;
  home_team: string;
  away_team: string;
  start_time: string | null;
}

async function fetchFeaturedGame(): Promise<DistributionGameData | null> {
  const today = new Date().toLocaleDateString("en-CA", {
    timeZone: "America/Los_Angeles",
  });

  const cols =
    "game_pk, date, team, expected_runs, expected_runs_p10, expected_runs_p50, expected_runs_p90, " +
    "total, total_p10, total_p50, total_p90, win_prob, win_prob_p10, win_prob_p90, runs_hist";

  const { data: predsRaw } = await supabase
    .from("model_outputs")
    .select(cols)
    .eq("date", today)
    .order("game_pk");

  const preds = (predsRaw ?? []) as unknown as PredRow[];
  if (preds.length === 0) return null;

  // group by game_pk; pick the first scheduled (lowest game_pk) with both teams + band data
  const byGame = new Map<number, PredRow[]>();
  for (const p of preds) {
    const arr = byGame.get(p.game_pk) ?? [];
    arr.push(p);
    byGame.set(p.game_pk, arr);
  }

  const gamePks = Array.from(byGame.keys()).sort((a, b) => a - b);
  const { data: gamesRaw } = await supabase
    .from("games")
    .select("game_pk, home_team, away_team, start_time")
    .in("game_pk", gamePks)
    .order("start_time");
  const games = (gamesRaw ?? []) as unknown as GameRow[];

  if (games.length === 0) return null;

  for (const g of games) {
    const rows = byGame.get(g.game_pk);
    if (!rows || rows.length < 2) continue;
    const home = rows.find((r) => r.team === g.home_team);
    const away = rows.find((r) => r.team === g.away_team);
    if (!home || !away) continue;
    if (
      home.expected_runs_p10 == null ||
      home.expected_runs_p90 == null ||
      away.expected_runs_p10 == null ||
      away.expected_runs_p90 == null ||
      !Array.isArray(home.runs_hist) ||
      !Array.isArray(away.runs_hist)
    )
      continue;

    return {
      date: home.date ?? today,
      home: {
        team: g.home_team,
        mean: Number(home.expected_runs),
        p10: Number(home.expected_runs_p10),
        p50: Number(home.expected_runs_p50 ?? home.expected_runs),
        p90: Number(home.expected_runs_p90),
        hist: (home.runs_hist as number[]).map((v) => Number(v)),
      },
      away: {
        team: g.away_team,
        mean: Number(away.expected_runs),
        p10: Number(away.expected_runs_p10),
        p50: Number(away.expected_runs_p50 ?? away.expected_runs),
        p90: Number(away.expected_runs_p90),
        hist: (away.runs_hist as number[]).map((v) => Number(v)),
      },
      homeWinProb: Number(home.win_prob),
      homeWinProbP10: home.win_prob_p10 != null ? Number(home.win_prob_p10) : null,
      homeWinProbP90: home.win_prob_p90 != null ? Number(home.win_prob_p90) : null,
      totalLine: home.total != null ? Number(home.total) : null,
      totalMean: Number(home.total_p50 ?? home.expected_runs + away.expected_runs),
      totalP10: Number(home.total_p10 ?? home.expected_runs_p10 + away.expected_runs_p10),
      totalP90: Number(home.total_p90 ?? home.expected_runs_p90 + away.expected_runs_p90),
      startTimeUtc: g.start_time ?? null,
    };
  }

  return null;
}

export default async function Page() {
  const featured = await fetchFeaturedGame();

  return (
    <main className="mx-auto w-full max-w-6xl min-w-0 px-4 py-8">
      <div className="mb-6">
        <h1 className="font-heading text-2xl tracking-tight">Methodology</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Hierarchical Bayesian skill model and per-PA Monte Carlo simulator, end to end
        </p>
      </div>

      <div className="lg:grid lg:grid-cols-[180px_1fr] lg:gap-8">
        <TableOfContents />
        <MethodologyContent featured={featured} />
      </div>
    </main>
  );
}
