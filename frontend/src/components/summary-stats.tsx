import type { GameMatchup } from "@/lib/types";
import { Card, CardContent } from "@/components/ui/card";

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
    <div className="grid grid-cols-3 gap-3">
      {stats.map((stat) => (
        <Card key={stat.label} size="sm">
          <CardContent className="flex flex-col items-center py-2">
            <span className="text-2xl font-bold tabular-nums">{stat.value}</span>
            <span className="text-xs text-muted-foreground">{stat.label}</span>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
