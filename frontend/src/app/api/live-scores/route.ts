import { NextResponse } from "next/server";

// Cached proxy to MLB Stats API so N browsers polling us don't become N requests to MLB.
// Next's fetch cache + revalidate=30 means we hit MLB at most ~2x/minute regardless of traffic.

export const revalidate = 30;

interface MLBTeam {
  score?: number;
}

interface MLBGame {
  gamePk: number;
  status?: {
    detailedState?: string;
    abstractGameState?: string; // "Preview" | "Live" | "Final"
  };
  teams?: {
    away?: { score?: number; team?: { abbreviation?: string } };
    home?: { score?: number; team?: { abbreviation?: string } };
  };
  linescore?: {
    currentInning?: number;
    inningState?: string; // "Top" | "Middle" | "Bottom" | "End"
  };
}

export interface LiveScore {
  game_pk: number;
  status: string | null;
  abstract_state: string | null;
  home_score: number | null;
  away_score: number | null;
  current_inning: number | null;
  inning_state: string | null;
}

export async function GET() {
  // Use PT date to match pipeline/games page
  const today = new Date().toLocaleDateString("en-CA", {
    timeZone: "America/Los_Angeles",
  });

  const url = `https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=${today}&hydrate=linescore`;

  try {
    const res = await fetch(url, {
      next: { revalidate: 30 },
      headers: { "User-Agent": "mlb-model-dashboard" },
    });
    if (!res.ok) {
      return NextResponse.json(
        { error: `MLB API ${res.status}` },
        { status: 502 }
      );
    }
    const data = await res.json();
    const games: MLBGame[] = data?.dates?.[0]?.games ?? [];

    const scores: LiveScore[] = games.map((g) => ({
      game_pk: g.gamePk,
      status: g.status?.detailedState ?? null,
      abstract_state: g.status?.abstractGameState ?? null,
      home_score: g.teams?.home?.score ?? null,
      away_score: g.teams?.away?.score ?? null,
      current_inning: g.linescore?.currentInning ?? null,
      inning_state: g.linescore?.inningState ?? null,
    }));

    return NextResponse.json(
      { scores, fetched_at: new Date().toISOString() },
      {
        headers: {
          // Edge/CDN can also cache for 30s
          "Cache-Control": "public, s-maxage=30, stale-while-revalidate=60",
        },
      }
    );
  } catch (err) {
    return NextResponse.json(
      { error: (err as Error).message },
      { status: 500 }
    );
  }
}
