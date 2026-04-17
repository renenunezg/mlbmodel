import { supabase } from "@/lib/supabase";
import type { ModelOutput, GameMatchup } from "@/lib/types";
import { GamesLive } from "@/components/games-live";
import { SummaryStats } from "@/components/summary-stats";

export const revalidate = 300;

export default async function Page() {
  const today = new Date().toLocaleDateString("en-CA", {
    timeZone: "America/Los_Angeles",
  }); // "YYYY-MM-DD"

  const displayDate = new Date().toLocaleDateString("en-US", {
    timeZone: "America/Los_Angeles",
    month: "short",
    day: "numeric",
    year: "numeric",
  });

  const { data: predictions } = await supabase
    .from("model_outputs")
    .select("*")
    .eq("date", today)
    .order("game_pk");

  if (!predictions || predictions.length === 0) {
    // Fall back to showing today's scheduled games from the games table
    const { data: scheduledGames } = await supabase
      .from("games")
      .select("game_pk, home_team, away_team, venue, start_time, status")
      .eq("game_date", today)
      .order("start_time");

    return (
      <main className="mx-auto max-w-6xl px-4 py-8">
        <div className="mb-6">
          <h1 className="font-heading text-2xl tracking-tight">
            Today&apos;s Games
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">{displayDate}</p>
        </div>
        <p className="mb-4 text-sm text-amber-500">
          Predictions not yet available — pipeline runs at 7 AM PT.
        </p>
        {scheduledGames && scheduledGames.length > 0 ? (
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            {scheduledGames.map((g) => {
              const startTime = g.start_time
                ? new Date(g.start_time).toLocaleTimeString("en-US", {
                    hour: "numeric",
                    minute: "2-digit",
                    timeZone: "America/Los_Angeles",
                  })
                : null;
              return (
                <div
                  key={g.game_pk}
                  className="rounded-lg border border-border bg-card p-4 font-mono text-sm"
                >
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="font-semibold">
                      {g.away_team} @ {g.home_team}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {startTime ?? "TBD"}
                    </span>
                  </div>
                  {g.venue && (
                    <p className="mt-0.5 text-xs text-muted-foreground font-sans">
                      {g.venue}
                    </p>
                  )}
                </div>
              );
            })}
          </div>
        ) : (
          <p className="text-muted-foreground">No games scheduled for today.</p>
        )}
      </main>
    );
  }

  const gamePks = [...new Set(predictions.map((p: ModelOutput) => p.game_pk))];

  const { data: games } = await supabase
    .from("games")
    .select(
      "game_pk, game_date, home_team, away_team, venue, start_time, home_score, away_score, status"
    )
    .in("game_pk", gamePks);

  const gameMap = new Map(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (games || []).map((g: any) => [g.game_pk, g])
  );

  const matchups: GameMatchup[] = gamePks
    .map((pk) => {
      const game = gameMap.get(pk);
      if (!game) return null;
      const rows = predictions.filter((p: ModelOutput) => p.game_pk === pk);
      const away = rows.find((r: ModelOutput) => r.team === game.away_team);
      const home = rows.find((r: ModelOutput) => r.team === game.home_team);
      if (!away || !home) return null;
      return {
        game_pk: pk,
        home_team: game.home_team,
        away_team: game.away_team,
        venue: game.venue,
        start_time: game.start_time,
        date: game.game_date,
        away,
        home,
        home_score: game.home_score,
        away_score: game.away_score,
        status: game.status ?? null,
      };
    })
    .filter(Boolean) as GameMatchup[];

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

  return (
    <main className="mx-auto max-w-6xl px-4 py-8">
      <div className="mb-6">
        <h1 className="font-heading text-2xl tracking-tight">
          Today&apos;s Games
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">{displayDate}</p>
      </div>

      <div className="mb-6">
        <SummaryStats matchups={matchups} />
      </div>

      <GamesLive initial={matchups} />
    </main>
  );
}
