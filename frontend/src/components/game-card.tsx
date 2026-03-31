import { cn, formatDate } from "@/lib/utils";
import type { GameMatchup } from "@/lib/types";
import { Card, CardHeader, CardContent } from "@/components/ui/card";
import { PredictionRow } from "@/components/prediction-row";

interface GameCardProps {
  matchup: GameMatchup;
}

export function GameCard({ matchup }: GameCardProps) {
  const hasEvPlay =
    matchup.away.ev_flag !== "No Play" ||
    matchup.home.ev_flag !== "No Play" ||
    matchup.away.run_line_ev_flag !== "No Play" ||
    matchup.home.run_line_ev_flag !== "No Play";

  const startTime = matchup.start_time
    ? new Date(matchup.start_time).toLocaleTimeString("en-US", {
        hour: "numeric",
        minute: "2-digit",
      })
    : null;

  return (
    <Card
      size="sm"
      className={cn(hasEvPlay && "ring-green-500/40 dark:ring-green-400/30")}
    >
      <CardHeader className="border-b pb-2">
        <div className="flex items-baseline justify-between gap-2">
          <span className="text-sm font-semibold">
            {matchup.away_team} @ {matchup.home_team}
          </span>
          <span className="text-xs text-muted-foreground">
            {matchup.venue && <>{matchup.venue} &middot; </>}
            {startTime && <>{startTime} &middot; </>}
            {formatDate(matchup.date)}
          </span>
        </div>
      </CardHeader>

      <CardContent className="space-y-0 py-0">
        {/* Column headers (visible on sm+) */}
        <div className="hidden text-[11px] font-medium uppercase tracking-wider text-muted-foreground sm:grid sm:grid-cols-[5rem_8rem_3.5rem_4rem_7rem_4.5rem_auto] sm:gap-x-4 sm:px-0 sm:pb-1">
          <span>Team</span>
          <span>Starter</span>
          <span>xR</span>
          <span>Win%</span>
          <span>Model / Book</span>
          <span>Edge</span>
          <span>Flags</span>
        </div>

        <div className="divide-y divide-border/50">
          <PredictionRow prediction={matchup.away} isHome={false} />
          <PredictionRow prediction={matchup.home} isHome={true} />
        </div>
      </CardContent>
    </Card>
  );
}
