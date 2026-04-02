import { supabase } from "@/lib/supabase";
import type { ModelOutput, GameMatchup } from "@/lib/types";
import { formatDate } from "@/lib/utils";
import { GameCard } from "@/components/game-card";
import { SummaryStats } from "@/components/summary-stats";

export const revalidate = 300;

export default async function Page() {
  const { data: predictions } = await supabase
    .from("model_outputs")
    .select("*")
    .order("game_pk");

  if (!predictions || predictions.length === 0) {
    return (
      <main className="mx-auto max-w-5xl px-4 py-8">
        <h1 className="font-heading text-2xl tracking-tight">
          Today&apos;s Picks
        </h1>
        <p className="mt-4 text-muted-foreground">
          No predictions available. Run the pipeline to generate today&apos;s
          picks.
        </p>
      </main>
    );
  }

  const gamePks = [...new Set(predictions.map((p: ModelOutput) => p.game_pk))];

  const { data: games } = await supabase
    .from("games")
    .select(
      "game_pk, game_date, home_team, away_team, venue, start_time, home_score, away_score"
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

  const displayDate =
    matchups.length > 0 ? formatDate(matchups[0].date) : null;

  return (
    <main className="mx-auto max-w-5xl px-4 py-8">
      <div className="mb-6">
        <h1 className="font-heading text-2xl tracking-tight">
          Today&apos;s Picks
        </h1>
        {displayDate && (
          <p className="mt-1 text-sm text-muted-foreground">{displayDate}</p>
        )}
      </div>

      <div className="mb-6">
        <SummaryStats matchups={matchups} />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {matchups.map((matchup) => (
          <GameCard key={matchup.game_pk} matchup={matchup} />
        ))}
      </div>
    </main>
  );
}
