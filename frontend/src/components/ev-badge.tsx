import type { ModelOutput } from "@/lib/types";

interface EvBadgeProps {
  prediction: ModelOutput;
}

export function EvBadge({ prediction }: EvBadgeProps) {
  const ml = prediction.ev_flag !== "No Play";
  const rl = prediction.run_line_ev_flag !== "No Play";
  const isOver = prediction.total_play === "Over";
  const isUnder = prediction.total_play === "Under";
  const variance = prediction.high_variance_flag === "Yes";

  const hasAny = ml || rl || isOver || isUnder || variance;
  if (!hasAny) return null;

  return (
    <div className="flex items-center gap-1 font-mono text-xs font-medium">
      {/* Slot 1: ML (+EV) */}
      <span className="inline-block w-6 text-right">
        {ml ? <span className="text-positive">+EV</span> : null}
      </span>
      {/* Slot 2: Run line (RL) */}
      <span className="inline-block w-4 text-right">
        {rl ? <span className="text-accent-blue">RL</span> : null}
      </span>
      {/* Slot 3: Totals (OVR/UND) or VAR */}
      <span className="inline-block w-7 text-right">
        {isOver ? (
          <span className="text-accent-amber">OVR</span>
        ) : isUnder ? (
          <span className="text-accent-amber">UND</span>
        ) : variance ? (
          <span className="text-negative">VAR</span>
        ) : null}
      </span>
    </div>
  );
}
