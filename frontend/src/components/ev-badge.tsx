import type { ModelOutput } from "@/lib/types";

interface EvBadgeProps {
  prediction: ModelOutput;
}

function fmtKelly(v: number | null): string {
  if (v == null || v <= 0) return "";
  return `${(v * 100).toFixed(1)}%`;
}

function fmtSpread(v: number | null): string {
  if (v == null) return "RL";
  return v > 0 ? `+${v}` : `${v}`;
}

export function EvBadge({ prediction }: EvBadgeProps) {
  const ml = prediction.ev_flag !== "No Play";
  const rl = prediction.run_line_ev_flag !== "No Play";
  const isOver = prediction.total_play === "Over";
  const isUnder = prediction.total_play === "Under";
  const variance = prediction.high_variance_flag === "Yes";

  const hasAny = ml || rl || isOver || isUnder || variance;
  if (!hasAny) return null;

  const mlValue = ml ? fmtKelly(prediction.kelly_quarter_ml) || "+EV" : null;
  const totalLine = prediction.total != null ? ` ${prediction.total}` : "";

  return (
    <div className="flex items-start gap-0.5 font-mono text-xs font-medium whitespace-nowrap">
      {/* Slot 1: ML Kelly % */}
      <div className="inline-block w-10 text-right">
        {mlValue ? (
          <>
            <div className="text-positive">{mlValue}</div>
            <div className="text-[10px] font-normal text-muted-foreground">Kelly</div>
          </>
        ) : null}
      </div>

      {/* Slot 2: Run line spread value + RL label */}
      <div className="inline-block w-8 text-right">
        {rl ? (
          <>
            <div className="text-accent-blue">{fmtSpread(prediction.spread)}</div>
            <div className="text-[10px] font-normal text-muted-foreground">RL</div>
          </>
        ) : null}
      </div>

      {/* Slot 3: Totals direction + O/U label, or VAR */}
      <div className="inline-block w-10 text-right">
        {isOver ? (
          <>
            <div className="text-accent-amber">O{totalLine}</div>
            <div className="text-[10px] font-normal text-muted-foreground">O/U</div>
          </>
        ) : isUnder ? (
          <>
            <div className="text-accent-amber">U{totalLine}</div>
            <div className="text-[10px] font-normal text-muted-foreground">O/U</div>
          </>
        ) : variance ? (
          <div className="text-negative">VAR</div>
        ) : null}
      </div>
    </div>
  );
}
