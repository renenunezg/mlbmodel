import type { ModelOutput } from "@/lib/types";

interface EvBadgeProps {
  prediction: ModelOutput;
}

export function EvBadge({ prediction }: EvBadgeProps) {
  const badges: React.ReactNode[] = [];

  if (prediction.ev_flag !== "No Play") {
    badges.push(
      <span key="ev" className="text-positive">+EV</span>
    );
  }

  if (prediction.run_line_ev_flag !== "No Play") {
    badges.push(
      <span key="rl" className="text-accent-blue">RL</span>
    );
  }

  if (prediction.total_play === "Over") {
    badges.push(
      <span key="total" className="text-accent-amber">OVR</span>
    );
  } else if (prediction.total_play === "Under") {
    badges.push(
      <span key="total" className="text-accent-amber">UND</span>
    );
  }

  if (prediction.high_variance_flag === "Yes") {
    badges.push(
      <span key="var" className="text-negative">VAR</span>
    );
  }

  if (badges.length === 0) return null;

  return (
    <div className="flex items-center gap-1.5 font-mono text-xs font-medium">
      {badges}
    </div>
  );
}
