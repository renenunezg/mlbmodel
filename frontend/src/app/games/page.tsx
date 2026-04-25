import { supabase } from "@/lib/supabase";
import type { ModelOutput, GameMatchup, GameInfo } from "@/lib/types";
import { GamesLive } from "@/components/games-live";
import { SummaryStats } from "@/components/summary-stats";
import { GameCardUnavailable } from "@/components/game-card-unavailable";
import { LastUpdated } from "@/components/last-updated";
import { RealtimeRefresh } from "@/components/realtime-refresh";
import type { LiveScore } from "@/app/api/live-scores/route";

// Re-render every 30s so SSR includes fresh live scores from MLB API
export const revalidate = 30;

async function fetchLiveScores(): Promise<Map<number, LiveScore>> {
  const today = new Date().toLocaleDateString("en-CA", {
    timeZone: "America/Los_Angeles",
  });
  const url = `https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=${today}&hydrate=linescore`;
  try {
    const res = await fetch(url, {
      next: { revalidate: 30 },
      headers: { "User-Agent": "mlb-model-dashboard" },
    });
    if (!res.ok) return new Map();
    const data = await res.json();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const games: any[] = data?.dates?.[0]?.games ?? [];
    return new Map(
      games.map((g) => [
        g.gamePk as number,
        {
          game_pk: g.gamePk,
          status: g.status?.detailedState ?? null,
          abstract_state: g.status?.abstractGameState ?? null,
          home_score: g.teams?.home?.score ?? null,
          away_score: g.teams?.away?.score ?? null,
          current_inning: g.linescore?.currentInning ?? null,
          inning_state: g.linescore?.inningState ?? null,
        } as LiveScore,
      ])
    );
  } catch {
    return new Map();
  }
}

export default async function Page() {
  const today = new Date().toLocaleDateString("en-CA", {
    timeZone: "America/Los_Angeles",
  }); // "YYYY-MM-DD"

  const [{ data: predictions }, { data: allGames }, { data: latest }, liveScores] = await Promise.all([
    supabase.from("model_outputs").select("*").eq("date", today).order("game_pk"),
    supabase
      .from("games")
      .select("game_pk, game_date, home_team, away_team, venue, start_time, home_score, away_score, status")
      .eq("game_date", today)
      .order("start_time"),
    supabase
      .from("games")
      .select("updated_at")
      .eq("game_date", today)
      .order("updated_at", { ascending: false })
      .limit(1),
    fetchLiveScores(),
  ]);

  const lastUpdated: string | null = latest?.[0]?.updated_at ?? null;

  if ((!predictions || predictions.length === 0) && (!allGames || allGames.length === 0)) {
    return (
      <main className="mx-auto max-w-6xl px-4 py-8">
        <h1 className="font-heading text-2xl tracking-tight">
          Today&apos;s Games
        </h1>
        <p className="mt-4 text-muted-foreground">
          No predictions available. Run the pipeline to generate today&apos;s
          games.
        </p>
      </main>
    );
  }

  const predictionPks = new Set((predictions ?? []).map((p: ModelOutput) => p.game_pk));
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const gameMap = new Map((allGames ?? []).map((g: any) => [g.game_pk, g]));

  const matchups: GameMatchup[] = (predictions ?? [])
    .reduce((acc: number[], p: ModelOutput) => {
      if (!acc.includes(p.game_pk)) acc.push(p.game_pk);
      return acc;
    }, [])
    .map((pk: number) => {
      const game = gameMap.get(pk);
      if (!game) return null;
      const rows = (predictions ?? []).filter((p: ModelOutput) => p.game_pk === pk);
      const away = rows.find((r: ModelOutput) => r.team === game.away_team);
      const home = rows.find((r: ModelOutput) => r.team === game.home_team);
      if (!away || !home) return null;
      const live = liveScores.get(pk);
      return {
        game_pk: pk,
        home_team: game.home_team,
        away_team: game.away_team,
        venue: game.venue,
        start_time: game.start_time,
        date: game.game_date,
        away,
        home,
        home_score: live?.home_score ?? game.home_score,
        away_score: live?.away_score ?? game.away_score,
        status: live?.status ?? game.status ?? null,
        current_inning: live?.current_inning ?? null,
        inning_state: live?.inning_state ?? null,
      };
    })
    .filter(Boolean) as GameMatchup[];

  const unavailableGames: GameInfo[] = (allGames ?? []).filter(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (g: any) => !predictionPks.has(g.game_pk)
  ) as GameInfo[];

  matchups.sort((a, b) => {
    const aHasEv =
      a.away.ev_flag !== "No Play" || a.home.ev_flag !== "No Play" ? 1 : 0;
    const bHasEv =
      b.away.ev_flag !== "No Play" || b.home.ev_flag !== "No Play" ? 1 : 0;
    if (aHasEv !== bHasEv) return bHasEv - aHasEv;
    const aTime = a.start_time ?? "";
    const bTime = b.start_time ?? "";
    return aTime.localeCompare(bTime);
  });

  const displayDate = new Date().toLocaleDateString("en-US", {
    timeZone: "America/Los_Angeles",
    month: "short",
    day: "numeric",
    year: "numeric",
  });

  return (
    <main className="mx-auto max-w-6xl px-4 py-8">
      <div className="mb-6 flex items-start justify-between gap-4">
        <div>
          <h1 className="font-heading text-2xl tracking-tight">
            Today&apos;s Games
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">{displayDate}</p>
        </div>
        <LastUpdated
          timestamp={lastUpdated}
          schedule="Predictions ~5 AM PT • Scores live"
        />
      </div>

      <div className="mb-6">
        <SummaryStats matchups={matchups} />
      </div>

      <RealtimeRefresh tables={["games", "model_outputs"]} />
      <GamesLive initial={matchups} />

      {unavailableGames.length > 0 && (
        <div className="mt-4 space-y-3">
          {unavailableGames.map((game) => (
            <GameCardUnavailable key={game.game_pk} game={game} />
          ))}
        </div>
      )}
    </main>
  );
}
