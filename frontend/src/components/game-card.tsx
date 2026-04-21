import { cn, formatRuns, formatPct, formatOdds, formatConfidence } from "@/lib/utils";
import type { GameMatchup, ModelOutput } from "@/lib/types";
import { Card, CardHeader, CardContent } from "@/components/ui/card";
import { EvBadge } from "@/components/ev-badge";

interface GameCardProps {
  matchup: GameMatchup;
}

function TeamRow({
  prediction,
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
    <div className="grid grid-cols-[1fr_2.75rem_2.75rem_2.75rem_6rem] sm:grid-cols-[1fr_3.5rem_3.5rem_3.5rem_7rem_7rem] items-center gap-x-1.5 py-1.5 font-mono text-sm">
      {/* Team + Starter */}
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span
            className={cn(
              "font-bold tracking-wide",
              hasEvPlay && "text-positive"
            )}
          >
            {prediction.team}
          </span>
          {isFinal && (
            <span className="font-bold tabular-nums">{score}</span>
          )}
        </div>
        <span className="block text-xs text-muted-foreground font-sans leading-tight">
          {prediction.starter ?? "TBD"}
        </span>
      </div>

      {/* xR */}
      <div className="text-center">
        <div className="font-medium tabular-nums">
          {formatRuns(prediction.expected_runs)}
        </div>
        <div className="text-[10px] text-muted-foreground">xR</div>
      </div>

      {/* Win% */}
      <div className="text-center">
        <div className="font-medium tabular-nums">
          {formatPct(prediction.win_prob)}
        </div>
        <div className="text-[10px] text-muted-foreground">Win</div>
      </div>

      {/* Edge */}
      <div className="text-center">
        <div
          className={cn(
            "font-semibold tabular-nums",
            isPositiveEdge
              ? "text-positive"
              : confidence != null
                ? "text-negative"
                : "text-muted-foreground"
          )}
        >
          {formatConfidence(confidence)}
        </div>
        <div className="text-[10px] text-muted-foreground">Edge</div>
      </div>

      {/* Odds: Model / Book */}
      <div className="hidden text-center sm:block">
        <div className="tabular-nums">
          <span>{formatOdds(prediction.our_odds)}</span>
          <span className="text-muted-foreground mx-0.5">/</span>
          <span className="text-muted-foreground">
            {formatOdds(prediction.moneyline)}
          </span>
        </div>
        <div className="text-[10px] text-muted-foreground">Model / Book</div>
      </div>

      {/* Badges */}
      <div className="text-right">
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

  const isFinal =
    matchup.status === "Final" &&
    matchup.home_score != null &&
    matchup.away_score != null;

  // In progress = have a status that isn't pregame/final and we have scores
  const isLive =
    !isFinal &&
    matchup.status != null &&
    matchup.status !== "Scheduled" &&
    matchup.status !== "Pre-Game" &&
    matchup.status !== "Warmup" &&
    matchup.status !== "Delayed Start" &&
    (matchup.home_score != null || matchup.away_score != null);

  const showScores = isFinal || isLive;

  const liveLabel = (() => {
    if (!isLive) return null;
    const inning = matchup.current_inning;
    const state = matchup.inning_state;
    if (!inning) return matchup.status;
    const half =
      state === "Top"
        ? "Top"
        : state === "Bottom"
          ? "Bot"
          : state === "Middle"
            ? "Mid"
            : state === "End"
              ? "End"
              : "";
    return half ? `${half} ${inning}` : `Inn ${inning}`;
  })();

  const startTime = matchup.start_time
    ? new Date(matchup.start_time).toLocaleTimeString("en-US", {
        hour: "numeric",
        minute: "2-digit",
      })
    : null;

  return (
    <Card
      size="sm"
      className={cn(hasEvPlay && "border-l-2 border-l-positive")}
    >
      <CardHeader className="border-b pb-2">
        <div className="flex items-baseline justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="font-mono text-sm font-semibold">
              {matchup.away_team} @ {matchup.home_team}
            </span>
            {isFinal && (
              <span className="bg-muted px-1.5 py-0.5 text-[10px] font-mono font-medium text-muted-foreground">
                Final
              </span>
            )}
            {isLive && (
              <span className="bg-positive/15 text-positive px-1.5 py-0.5 text-[10px] font-mono font-medium">
                {liveLabel}
              </span>
            )}
          </div>
          <span className="text-xs text-muted-foreground font-mono">
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
            score={showScores ? matchup.away_score : null}
          />
          <TeamRow
            prediction={matchup.home}
            isHome={true}
            score={showScores ? matchup.home_score : null}
          />
        </div>
      </CardContent>
    </Card>
  );
}
