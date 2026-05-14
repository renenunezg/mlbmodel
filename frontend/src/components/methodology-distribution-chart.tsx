"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
} from "recharts";

interface TeamDist {
  team: string;
  mean: number;
  p10: number;
  p50: number;
  p90: number;
  hist: number[]; // 21 bins, 0..20 runs, probabilities summing to ~1
}

interface Props {
  date: string;
  home: TeamDist;
  away: TeamDist;
  homeWinProb: number;
  homeWinProbP10: number | null;
  homeWinProbP90: number | null;
  totalLine: number | null;
  totalMean: number;
  totalP10: number;
  totalP90: number;
  startTimeUtc: string | null;
}

export function MethodologyDistributionChart({
  date,
  home,
  away,
  homeWinProb,
  homeWinProbP10,
  homeWinProbP90,
  totalLine,
  totalMean,
  totalP10,
  totalP90,
  startTimeUtc,
}: Props) {
  const homeHist = Array.isArray(home.hist) ? home.hist : [];
  const awayHist = Array.isArray(away.hist) ? away.hist : [];
  const MAX = Math.max(homeHist.length, awayHist.length, 1) - 1;

  const data = Array.from({ length: MAX + 1 }, (_, k) => ({
    runs: k,
    home: Number(((homeHist[k] ?? 0) * 100).toFixed(3)),
    away: Number(((awayHist[k] ?? 0) * 100).toFixed(3)),
  }));

  const ptTime =
    startTimeUtc != null
      ? new Date(startTimeUtc).toLocaleTimeString("en-US", {
          timeZone: "America/Los_Angeles",
          hour: "numeric",
          minute: "2-digit",
        })
      : null;

  const dateOnly = date.slice(0, 10);
  const dateDisplay = new Date(`${dateOnly}T12:00:00Z`).toLocaleDateString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    year: "numeric",
  });

  const pct = (x: number | null | undefined) =>
    x == null ? "—" : `${(x * 100).toFixed(1)}%`;
  const fmt = (x: number, d = 2) => x.toFixed(d);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2 text-sm">
        <p className="font-medium">
          {away.team} @ {home.team}
        </p>
        <p className="font-mono text-xs text-muted-foreground">
          {dateDisplay}
          {ptTime ? ` · ${ptTime} PT` : ""}
        </p>
      </div>

      <ResponsiveContainer width="100%" height={280}>
        <BarChart
          data={data}
          margin={{ top: 8, right: 12, left: 0, bottom: 24 }}
          barCategoryGap={2}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" vertical={false} />
          <XAxis
            dataKey="runs"
            tick={{
              fill: "#52525b",
              fontSize: 11,
              fontFamily: "var(--font-geist-mono)",
            }}
            stroke="#d4d4d8"
            label={{
              value: "Runs scored by team (capped at 20)",
              position: "insideBottom",
              offset: -10,
              fill: "#71717a",
              fontSize: 11,
              fontFamily: "var(--font-geist-mono)",
            }}
          />
          <YAxis
            tick={{
              fill: "#52525b",
              fontSize: 11,
              fontFamily: "var(--font-geist-mono)",
            }}
            stroke="#d4d4d8"
            tickFormatter={(v: number) => `${v.toFixed(0)}%`}
            label={{
              value: "share of sims",
              angle: -90,
              position: "insideLeft",
              fill: "#71717a",
              fontSize: 11,
              fontFamily: "var(--font-geist-mono)",
            }}
          />
          <Tooltip
            formatter={(value, name) => [
              `${Number(value ?? 0).toFixed(2)}%`,
              String(name ?? ""),
            ]}
            labelFormatter={(label) => `${label} runs`}
            contentStyle={{
              backgroundColor: "#ffffff",
              border: "1px solid #d4d4d8",
              borderRadius: "2px",
              fontFamily: "var(--font-geist-mono)",
              fontSize: "12px",
            }}
          />
          <Legend
            verticalAlign="top"
            height={28}
            wrapperStyle={{
              fontFamily: "var(--font-geist-mono)",
              fontSize: "11px",
              textTransform: "uppercase",
              letterSpacing: "0.05em",
            }}
          />
          <Bar
            dataKey="home"
            name={`${home.team} (home)`}
            fill="#2d7a4f"
            fillOpacity={0.7}
          />
          <Bar
            dataKey="away"
            name={`${away.team} (away)`}
            fill="#a855f7"
            fillOpacity={0.55}
          />
          <ReferenceLine
            x={Math.round(home.mean)}
            stroke="#2d7a4f"
            strokeDasharray="3 3"
            strokeWidth={1}
            label={{
              value: `μ ${fmt(home.mean, 1)}`,
              position: "top",
              fill: "#2d7a4f",
              fontSize: 10,
              fontFamily: "var(--font-geist-mono)",
            }}
          />
          <ReferenceLine
            x={Math.round(away.mean)}
            stroke="#a855f7"
            strokeDasharray="3 3"
            strokeWidth={1}
            label={{
              value: `μ ${fmt(away.mean, 1)}`,
              position: "top",
              fill: "#a855f7",
              fontSize: 10,
              fontFamily: "var(--font-geist-mono)",
            }}
          />
        </BarChart>
      </ResponsiveContainer>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3 text-sm">
        <div className="rounded-sm border border-border p-3">
          <p className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
            {home.team} expected runs
          </p>
          <p className="mt-1 text-base font-medium">{fmt(home.mean, 2)}</p>
          <p className="mt-0.5 font-mono text-xs text-muted-foreground">
            p10–p90: {fmt(home.p10, 1)} – {fmt(home.p90, 1)}
          </p>
        </div>
        <div className="rounded-sm border border-border p-3">
          <p className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
            {away.team} expected runs
          </p>
          <p className="mt-1 text-base font-medium">{fmt(away.mean, 2)}</p>
          <p className="mt-0.5 font-mono text-xs text-muted-foreground">
            p10–p90: {fmt(away.p10, 1)} – {fmt(away.p90, 1)}
          </p>
        </div>
        <div className="rounded-sm border border-border p-3">
          <p className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
            {home.team} win probability
          </p>
          <p className="mt-1 text-base font-medium">{pct(homeWinProb)}</p>
          <p className="mt-0.5 font-mono text-xs text-muted-foreground">
            p10–p90: {pct(homeWinProbP10)} – {pct(homeWinProbP90)}
          </p>
        </div>
      </div>

      <div className="rounded-sm border border-border bg-muted/30 p-3 text-xs leading-relaxed text-muted-foreground">
        <p>
          <span className="font-mono uppercase tracking-wider">Total runs:</span>{" "}
          μ {fmt(totalMean, 2)}, p10–p90 {fmt(totalP10, 1)}–{fmt(totalP90, 1)}.
          {totalLine != null ? (
            <>
              {" "}Book O/U line: <span className="font-mono">{fmt(totalLine, 1)}</span>.
            </>
          ) : null}{" "}
          Bars are the empirical PMF from the raw 10,000-sim run array, binned 0..20.
        </p>
      </div>
    </div>
  );
}
