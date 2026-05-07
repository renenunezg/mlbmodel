import type { GameMatchup } from "@/lib/types";

interface SummaryStatsProps {
  matchups: GameMatchup[];
}

export function SummaryStats({ matchups }: SummaryStatsProps) {
  const gameCount = matchups.length;

  const mlPlays = new Set<string>();
  const rlPlays = new Set<string>();
  const totalsGames = new Set<number>();
  for (const m of matchups) {
    if (m.away.ev_flag !== "No Play") mlPlays.add(`${m.game_pk}-away`);
    if (m.home.ev_flag !== "No Play") mlPlays.add(`${m.game_pk}-home`);
    if (m.away.run_line_ev_flag !== "No Play") rlPlays.add(`${m.game_pk}-away`);
    if (m.home.run_line_ev_flag !== "No Play") rlPlays.add(`${m.game_pk}-home`);
    if (m.away.total_play !== "No Play" || m.home.total_play !== "No Play") {
      totalsGames.add(m.game_pk);
    }
  }

  const stats = [
    { label: "Games", value: gameCount },
    { label: "ML", value: mlPlays.size },
    { label: "RL", value: rlPlays.size },
    { label: "Totals", value: totalsGames.size },
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
