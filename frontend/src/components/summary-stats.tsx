import type { GameMatchup } from "@/lib/types";

interface SummaryStatsProps {
  matchups: GameMatchup[];
}

export function SummaryStats({ matchups }: SummaryStatsProps) {
  const gameCount = matchups.length;

  const evPlays = new Set<string>();
  for (const m of matchups) {
    if (m.away.ev_flag !== "No Play") evPlays.add(`${m.game_pk}-away`);
    if (m.home.ev_flag !== "No Play") evPlays.add(`${m.game_pk}-home`);
  }

  let totalPlays = 0;
  for (const m of matchups) {
    if (m.away.total_play !== "No Play") totalPlays++;
    if (m.home.total_play !== "No Play") totalPlays++;
  }

  const stats = [
    { label: "Games", value: gameCount },
    { label: "+EV Plays", value: evPlays.size },
    { label: "Total Plays", value: totalPlays },
  ];

  return (
    <div className="flex items-baseline gap-6 font-mono text-sm">
      {stats.map((stat) => (
        <div key={stat.label} className="flex items-baseline gap-1.5">
          <span className="text-xs uppercase tracking-wider text-muted-foreground">
            {stat.label}
          </span>
          <span className="font-bold tabular-nums">{stat.value}</span>
        </div>
      ))}
    </div>
  );
}
