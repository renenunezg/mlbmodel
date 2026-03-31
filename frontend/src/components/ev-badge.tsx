import { cn } from "@/lib/utils";
import type { ModelOutput } from "@/lib/types";

interface EvBadgeProps {
  prediction: ModelOutput;
}

function Pill({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex h-5 shrink-0 items-center justify-center rounded-full px-2 text-xs font-medium whitespace-nowrap",
        className
      )}
    >
      {children}
    </span>
  );
}

export function EvBadge({ prediction }: EvBadgeProps) {
  const badges: React.ReactNode[] = [];

  if (prediction.ev_flag !== "No Play") {
    badges.push(
      <Pill key="ev" className="bg-green-500/15 text-green-700 dark:text-green-400">
        +EV
      </Pill>
    );
  }

  if (prediction.run_line_ev_flag !== "No Play") {
    badges.push(
      <Pill key="rl" className="bg-blue-500/15 text-blue-700 dark:text-blue-400">
        RL +EV
      </Pill>
    );
  }

  if (prediction.total_play === "Over") {
    badges.push(
      <Pill key="total" className="bg-amber-500/15 text-amber-700 dark:text-amber-400">
        Over
      </Pill>
    );
  } else if (prediction.total_play === "Under") {
    badges.push(
      <Pill key="total" className="bg-purple-500/15 text-purple-700 dark:text-purple-400">
        Under
      </Pill>
    );
  }

  if (prediction.high_variance_flag === "Yes") {
    badges.push(
      <Pill key="var" className="bg-red-500/15 text-red-700 dark:text-red-400">
        High Var
      </Pill>
    );
  }

  if (badges.length === 0) return null;

  return <div className="flex flex-wrap items-center gap-1">{badges}</div>;
}
