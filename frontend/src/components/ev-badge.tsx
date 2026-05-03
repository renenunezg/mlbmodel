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
    <div className="flex items-center gap-1 font-mono text-xs font-medium whitespace-nowrap">
      {/* Slot 1: ML Kelly % */}
      <div className="w-10 text-center">
        <div className="tabular-nums">{mlValue ? <span className="text-positive">{mlValue}</span> : <span className="invisible">-</span>}</div>
        <div className="text-[10px] font-normal text-muted-foreground">{mlValue ? "Kelly" : <span className="invisible">-</span>}</div>
      </div>

      {/* Slot 2: Run line spread + RL label */}
      <div className="w-8 text-center">
        <div className="tabular-nums">{rl ? <span className="text-accent-blue">{fmtSpread(prediction.spread)}</span> : <span className="invisible">-</span>}</div>
        <div className="text-[10px] font-normal text-muted-foreground">{rl ? "RL" : <span className="invisible">-</span>}</div>
      </div>

      {/* Slot 3: Totals + O/U label, or VAR */}
      <div className="w-10 text-center">
        {isOver || isUnder ? (
          <>
            <div className={isOver ? "text-accent-amber" : "text-accent-amber"}>
              {isOver ? `O${totalLine}` : `U${totalLine}`}
            </div>
            <div className="text-[10px] font-normal text-muted-foreground">O/U</div>
          </>
        ) : variance ? (
          <>
            <div className="text-negative">VAR</div>
            <div className="text-[10px] font-normal text-muted-foreground invisible">-</div>
          </>
        ) : (
          <>
            <div className="invisible">-</div>
            <div className="text-[10px] invisible">-</div>
          </>
        )}
      </div>
    </div>
  );
}
