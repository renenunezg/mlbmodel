"use client";

import { useMemo, useState } from "react";
import type { PosteriorSkill } from "@/lib/types";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface Props {
  skills: PosteriorSkill[];
}

type Tab = {
  key: string;
  label: string;
  actor_type: "batter" | "pitcher";
  split_label: string;
  // For batters, higher xwOBA = better. For pitchers, lower xwOBA allowed = better.
  invertSort: boolean;
};

const TABS: Tab[] = [
  { key: "bat_rhp", label: "Batters vs RHP", actor_type: "batter", split_label: "vs_rhp", invertSort: false },
  { key: "bat_lhp", label: "Batters vs LHP", actor_type: "batter", split_label: "vs_lhp", invertSort: false },
  { key: "sp", label: "Starting Pitchers", actor_type: "pitcher", split_label: "sp", invertSort: true },
  { key: "rp", label: "Relief Pitchers", actor_type: "pitcher", split_label: "rp", invertSort: true },
];

function fmtScore(v: number): string {
  return v.toFixed(3);
}

export function TopSkillsLeaderboard({ skills }: Props) {
  const [tabKey, setTabKey] = useState<string>("bat_rhp");
  const tab = TABS.find((t) => t.key === tabKey)!;

  const { topRows, bottomRows } = useMemo(() => {
    const subset = skills.filter(
      (s) => s.actor_type === tab.actor_type && s.split_label === tab.split_label,
    );
    // For pitchers, "top" in the table means "best at preventing wOBA" = lowest
    // xwOBA allowed. In the DB we always ranked by ascending xwOBA so rank_type
    // 'top' = highest skill_score (best batter; worst pitcher). Swap for pitchers.
    const topKey = tab.invertSort ? "bottom" : "top";
    const bottomKey = tab.invertSort ? "top" : "bottom";
    const top = subset.filter((s) => s.rank_type === topKey).sort((a, b) => a.rank - b.rank);
    const bot = subset.filter((s) => s.rank_type === bottomKey).sort((a, b) => a.rank - b.rank);
    return { topRows: top, bottomRows: bot };
  }, [skills, tab]);

  const heading = tab.actor_type === "batter" ? "Hitter" : "Pitcher";
  const valueLabel = tab.actor_type === "batter" ? "xwOBA" : "xwOBA against";

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <div className="inline-flex rounded-sm border border-border p-0.5 text-xs font-mono">
          {TABS.map((t) => (
            <button
              key={t.key}
              type="button"
              onClick={() => setTabKey(t.key)}
              className={`px-3 py-1 transition-colors ${
                tabKey === t.key
                  ? "bg-foreground text-background"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      <div className="grid gap-6 sm:grid-cols-2">
        <div>
          <h3 className="mb-2 text-sm font-mono uppercase tracking-wider text-muted-foreground">
            Best {heading}s
          </h3>
          {topRows.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-8">#</TableHead>
                  <TableHead>{heading}</TableHead>
                  <TableHead>Team</TableHead>
                  <TableHead className="text-right">{valueLabel}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {topRows.map((r) => (
                  <TableRow key={`top-${r.rank}-${r.actor_id}`}>
                    <TableCell className="font-mono text-muted-foreground">{r.rank}</TableCell>
                    <TableCell className="font-medium">{r.actor_name || `id ${r.actor_id}`}</TableCell>
                    <TableCell className="text-muted-foreground">{r.team || "-"}</TableCell>
                    <TableCell className="text-right font-mono tabular-nums">
                      {fmtScore(r.skill_score)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <p className="text-sm text-muted-foreground">No data.</p>
          )}
        </div>
        <div>
          <h3 className="mb-2 text-sm font-mono uppercase tracking-wider text-muted-foreground">
            Worst {heading}s
          </h3>
          {bottomRows.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-8">#</TableHead>
                  <TableHead>{heading}</TableHead>
                  <TableHead>Team</TableHead>
                  <TableHead className="text-right">{valueLabel}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {bottomRows.map((r) => (
                  <TableRow key={`bot-${r.rank}-${r.actor_id}`}>
                    <TableCell className="font-mono text-muted-foreground">{r.rank}</TableCell>
                    <TableCell className="font-medium">{r.actor_name || `id ${r.actor_id}`}</TableCell>
                    <TableCell className="text-muted-foreground">{r.team || "-"}</TableCell>
                    <TableCell className="text-right font-mono tabular-nums">
                      {fmtScore(r.skill_score)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <p className="text-sm text-muted-foreground">No data.</p>
          )}
        </div>
      </div>
    </div>
  );
}
