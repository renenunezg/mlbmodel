import { cn, formatRuns, formatPct, formatOdds, formatConfidence } from "@/lib/utils";
import type { ModelOutput } from "@/lib/types";
import { EvBadge } from "@/components/ev-badge";

interface PredictionRowProps {
  prediction: ModelOutput;
  isHome: boolean;
}

export function PredictionRow({ prediction, isHome }: PredictionRowProps) {
  const hasEvPlay = prediction.ev_flag !== "No Play";
  const confidence = prediction.ml_confidence;
  const isPositiveEdge = confidence != null && confidence > 0;

  return (
    <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 items-center py-2 sm:grid-cols-[5rem_8rem_3.5rem_4rem_7rem_4.5rem_auto]">
      {/* Team name */}
      <span
        className={cn(
          "text-sm font-bold tracking-wide",
          hasEvPlay && "text-green-600 dark:text-green-400"
        )}
      >
        {prediction.team}
      </span>

      {/* Starter */}
      <span className="truncate text-xs text-muted-foreground">
        {prediction.starter ?? "TBD"}
      </span>

      {/* xR */}
      <div className="text-sm tabular-nums">
        <span className="text-muted-foreground text-xs mr-1 sm:hidden">xR</span>
        {formatRuns(prediction.expected_runs)}
      </div>

      {/* Win prob */}
      <div className="text-sm tabular-nums">
        <span className="text-muted-foreground text-xs mr-1 sm:hidden">Win</span>
        {formatPct(prediction.win_prob)}
      </div>

      {/* Model odds / Book odds */}
      <div className="text-sm tabular-nums">
        <span className="text-muted-foreground text-xs mr-1 sm:hidden">Odds</span>
        <span>{formatOdds(prediction.our_odds)}</span>
        <span className="text-muted-foreground mx-0.5">/</span>
        <span className="text-muted-foreground">{formatOdds(prediction.moneyline)}</span>
      </div>

      {/* Edge */}
      <div
        className={cn(
          "text-sm tabular-nums font-medium",
          isPositiveEdge
            ? "text-green-600 dark:text-green-400"
            : confidence != null
              ? "text-red-600 dark:text-red-400"
              : "text-muted-foreground"
        )}
      >
        <span className="text-muted-foreground text-xs mr-1 sm:hidden">Edge</span>
        {formatConfidence(confidence)}
      </div>

      {/* Badges */}
      <div className="col-span-2 sm:col-span-1">
        <EvBadge prediction={prediction} />
      </div>
    </div>
  );
}
