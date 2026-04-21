import type { ModelOutput } from "@/lib/types";

interface EvBadgeProps {
  prediction: ModelOutput;
}

function fmtKelly(v: number | null): string {
  if (v == null || v <= 0) return "";
  return `${(v * 100).toFixed(1)}%`;
}

export function EvBadge({ prediction }: EvBadgeProps) {
  const ml = prediction.ev_flag !== "No Play";
  const rl = prediction.run_line_ev_flag !== "No Play";
  const isOver = prediction.total_play === "Over";
  const isUnder = prediction.total_play === "Under";
  const variance = prediction.high_variance_flag === "Yes";

  const hasAny = ml || rl || isOver || isUnder || variance;
  if (!hasAny) return null;

  const mlLabel = ml ? fmtKelly(prediction.kelly_quarter_ml) || "+EV" : null;
  const totalLine = prediction.total != null ? ` ${prediction.total}` : "";

  return (
    <div className="flex items-center gap-0.5 font-mono text-xs font-medium whitespace-nowrap">
      {/* Slot 1: ML — Kelly % when +EV */}
      <span className="inline-block w-10 text-right">
        {mlLabel ? <span className="text-positive">{mlLabel}</span> : null}
      </span>
      {/* Slot 2: Run line */}
      <span className="inline-block w-5 text-right">
        {rl ? <span className="text-accent-blue">RL</span> : null}
      </span>
      {/* Slot 3: Totals with line, or VAR */}
      <span className="inline-block w-10 text-right">
        {isOver ? (
          <span className="text-accent-amber">O{totalLine}</span>
        ) : isUnder ? (
          <span className="text-accent-amber">U{totalLine}</span>
        ) : variance ? (
          <span className="text-negative">VAR</span>
        ) : null}
      </span>
    </div>
  );
}
