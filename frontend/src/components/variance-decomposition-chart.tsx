"use client";

import type { PosteriorSigma } from "@/lib/types";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ErrorBar,
} from "recharts";

interface VarianceDecompositionChartProps {
  data: PosteriorSigma[];
}

const SIGMA_LABELS: Record<string, string> = {
  sigma_batter: "Batter skill",
  sigma_platoon: "Platoon split",
  sigma_pitcher: "Pitcher skill",
  sigma_park: "Park effect",
};

export function VarianceDecompositionChart({ data }: VarianceDecompositionChartProps) {
  if (data.length === 0) return null;

  const latestDate = data.reduce(
    (max, d) => (d.refit_date > max ? d.refit_date : max),
    "",
  );
  const latest = data.filter((d) => d.refit_date === latestDate);

  const chartData = latest
    .map((s) => ({
      name: SIGMA_LABELS[s.sigma_name] ?? s.sigma_name,
      mean: Number(s.mean.toFixed(4)),
      // ErrorBar expects [lowerDeviation, upperDeviation] as positive values
      error: [
        s.p10 != null ? Math.max(0, Number((s.mean - s.p10).toFixed(4))) : 0,
        s.p90 != null ? Math.max(0, Number((s.p90 - s.mean).toFixed(4))) : 0,
      ] as [number, number],
    }))
    .sort((a, b) => b.mean - a.mean);

  return (
    <div>
      <p className="text-xs text-muted-foreground mb-3 font-mono">
        Posterior sigma (logit scale) as of {latestDate}. Bars show posterior mean; whiskers show 10th-90th percentile.
      </p>
      <ResponsiveContainer width="100%" height={Math.max(160, chartData.length * 48)}>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 4, right: 40, left: 100, bottom: 4 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#e5e5e5"
            horizontal={false}
          />
          <XAxis
            type="number"
            tick={{ fill: "#52525b", fontSize: 11, fontFamily: "var(--font-geist-mono)" }}
            stroke="#d4d4d8"
            tickFormatter={(v: number) => v.toFixed(2)}
            domain={[0, "auto"]}
          />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fill: "#52525b", fontSize: 11, fontFamily: "var(--font-geist-mono)" }}
            stroke="#d4d4d8"
            width={96}
          />
          <Tooltip
            formatter={(value) => Number(value).toFixed(4)}
            contentStyle={{
              backgroundColor: "#ffffff",
              border: "1px solid #d4d4d8",
              borderRadius: "2px",
              fontFamily: "var(--font-geist-mono)",
              fontSize: "12px",
            }}
          />
          <Bar dataKey="mean" name="Posterior mean" fill="#2d7a4f" barSize={18} radius={2}>
            <ErrorBar
              dataKey="error"
              width={6}
              strokeWidth={1.5}
              stroke="#1a5c38"
              direction="x"
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
