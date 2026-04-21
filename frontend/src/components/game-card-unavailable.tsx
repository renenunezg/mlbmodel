import type { GameInfo } from "@/lib/types";
import { Card, CardHeader, CardContent } from "@/components/ui/card";

interface GameCardUnavailableProps {
  game: GameInfo;
}

export function GameCardUnavailable({ game }: GameCardUnavailableProps) {
  const startTime = game.start_time
    ? new Date(game.start_time).toLocaleTimeString("en-US", {
        hour: "numeric",
        minute: "2-digit",
      })
    : null;

  return (
    <Card size="sm">
      <CardHeader className="border-b pb-2">
        <div className="flex items-baseline justify-between gap-2">
          <span className="font-mono text-sm font-semibold">
            {game.away_team} @ {game.home_team}
          </span>
          <span className="text-xs text-muted-foreground font-mono">
            {game.venue && <>{game.venue} &middot; </>}
            {startTime}
          </span>
        </div>
      </CardHeader>
      <CardContent className="py-0">
        <div className="divide-y divide-border/50">
          {[game.away_team, game.home_team].map((team) => (
            <div
              key={team}
              className="grid grid-cols-[1fr_2.75rem_2.75rem_2.75rem_6rem] sm:grid-cols-[1fr_3.5rem_3.5rem_3.5rem_7rem_7rem] items-center gap-x-1.5 py-1.5 font-mono text-sm"
            >
              <div className="min-w-0">
                <div className="font-bold tracking-wide">{team}</div>
                <span className="block text-xs text-muted-foreground font-sans leading-tight">
                  Starter not announced
                </span>
              </div>
              <div className="text-center">
                <div className="tabular-nums text-muted-foreground">—</div>
                <div className="text-[10px] text-muted-foreground">xR</div>
              </div>
              <div className="text-center">
                <div className="tabular-nums text-muted-foreground">—</div>
                <div className="text-[10px] text-muted-foreground">Win</div>
              </div>
              <div className="text-center">
                <div className="tabular-nums text-muted-foreground">—</div>
                <div className="text-[10px] text-muted-foreground">Edge</div>
              </div>
              <div className="hidden text-center sm:block">
                <div className="tabular-nums text-muted-foreground">—</div>
                <div className="text-[10px] text-muted-foreground">Model / Book</div>
              </div>
              <div />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
