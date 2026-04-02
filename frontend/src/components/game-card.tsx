import { cn, formatDate, formatRuns, formatPct, formatOdds, formatConfidence } from "@/lib/utils";
import type { GameMatchup, ModelOutput } from "@/lib/types";
import { Card, CardHeader, CardContent } from "@/components/ui/card";
import { EvBadge } from "@/components/ev-badge";

interface GameCardProps {
  matchup: GameMatchup;
}

function TeamRow({
  prediction,
  isHome,
  score,
}: {
  prediction: ModelOutput;
  isHome: boolean;
  score: number | null;
}) {
  const hasEvPlay = prediction.ev_flag !== "No Play";
  const confidence = prediction.ml_confidence;
  const isPositiveEdge = confidence != null && confidence > 0;
  const isFinal = score != null;

  return (
    <div className="flex items-center gap-3 py-2">
      {/* Team + Starter */}
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span
            className={cn(
              "text-sm font-bold tracking-wide",
              hasEvPlay && "text-green-600 dark:text-green-400"
            )}
          >
            {prediction.team}
          </span>
          {isFinal && (
            <span className="text-sm font-bold tabular-nums">{score}</span>
          )}
        </div>
        <span className="block truncate text-xs text-muted-foreground">
          {prediction.starter ?? "TBD"}
        </span>
      </div>

      {/* xR */}
      <div className="text-center">
        <div className="text-sm font-medium tabular-nums">
          {formatRuns(prediction.expected_runs)}
        </div>
        <div className="text-[10px] text-muted-foreground">xR</div>
      </div>

      {/* Win% */}
      <div className="text-center">
        <div className="text-sm font-medium tabular-nums">
          {formatPct(prediction.win_prob)}
        </div>
        <div className="text-[10px] text-muted-foreground">Win</div>
      </div>

      {/* Edge */}
      <div className="text-center">
        <div
          className={cn(
            "text-sm font-semibold tabular-nums",
            isPositiveEdge
              ? "text-green-600 dark:text-green-400"
              : confidence != null
                ? "text-red-600 dark:text-red-400"
                : "text-muted-foreground"
          )}
        >
          {formatConfidence(confidence)}
        </div>
        <div className="text-[10px] text-muted-foreground">Edge</div>
      </div>

      {/* Odds: Model / Book */}
      <div className="hidden text-center sm:block">
        <div className="text-sm tabular-nums">
          <span>{formatOdds(prediction.our_odds)}</span>
          <span className="text-muted-foreground mx-0.5">/</span>
          <span className="text-muted-foreground">
            {formatOdds(prediction.moneyline)}
          </span>
        </div>
        <div className="text-[10px] text-muted-foreground">Model / Book</div>
      </div>

      {/* Badges */}
      <div className="shrink-0">
        <EvBadge prediction={prediction} />
      </div>
    </div>
  );
}

export function GameCard({ matchup }: GameCardProps) {
  const hasEvPlay =
    matchup.away.ev_flag !== "No Play" ||
    matchup.home.ev_flag !== "No Play" ||
    matchup.away.run_line_ev_flag !== "No Play" ||
    matchup.home.run_line_ev_flag !== "No Play";

  const isFinal = matchup.home_score != null && matchup.away_score != null;

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
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold">
              {matchup.away_team} @ {matchup.home_team}
            </span>
            {isFinal && (
              <span className="rounded bg-muted px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground">
                Final
              </span>
            )}
          </div>
          <span className="text-xs text-muted-foreground">
            {matchup.venue && <>{matchup.venue} &middot; </>}
            {startTime && <>{startTime}</>}
          </span>
        </div>
      </CardHeader>

      <CardContent className="space-y-0 py-0">
        <div className="divide-y divide-border/50">
          <TeamRow
            prediction={matchup.away}
            isHome={false}
            score={isFinal ? matchup.away_score : null}
          />
          <TeamRow
            prediction={matchup.home}
            isHome={true}
            score={isFinal ? matchup.home_score : null}
          />
        </div>
      </CardContent>
    </Card>
  );
}
