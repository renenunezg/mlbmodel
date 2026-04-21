"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

export interface ResidualsChartProps {
  residuals: number[];
}

function buildBins(values: number[]) {
  if (values.length === 0) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const lo = Math.floor(min);
  const hi = Math.ceil(max);
  const bins: { bucket: string; mid: number; count: number }[] = [];
  for (let b = lo; b < hi; b++) {
    bins.push({ bucket: `${b}`, mid: b + 0.5, count: 0 });
  }
  for (const v of values) {
    const idx = Math.min(Math.floor(v) - lo, bins.length - 1);
    if (idx >= 0) bins[idx].count += 1;
  }
  return bins;
}

export function ResidualsChart({ residuals }: ResidualsChartProps) {
  const bins = buildBins(residuals);
  const mean =
    residuals.length > 0
      ? residuals.reduce((a, b) => a + b, 0) / residuals.length
      : 0;

  return (
    <div>
      <p className="text-xs text-muted-foreground mb-2">
        {residuals.length} graded team-games · mean bias{" "}
        {mean >= 0 ? "+" : ""}
        {mean.toFixed(2)} runs
      </p>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={bins}
          margin={{ top: 8, right: 16, left: 8, bottom: 8 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
          <XAxis
            dataKey="bucket"
            tick={{
              fill: "#52525b",
              fontSize: 11,
              fontFamily: "var(--font-geist-mono)",
            }}
            stroke="#d4d4d8"
            label={{
              value: "Actual − Predicted (runs)",
              position: "insideBottom",
              offset: -4,
              style: { fill: "#52525b", fontSize: 11 },
            }}
          />
          <YAxis
            tick={{
              fill: "#52525b",
              fontSize: 11,
              fontFamily: "var(--font-geist-mono)",
            }}
            stroke="#d4d4d8"
            allowDecimals={false}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#ffffff",
              border: "1px solid #d4d4d8",
              borderRadius: "2px",
              fontFamily: "var(--font-geist-mono)",
              fontSize: "12px",
            }}
            formatter={(value) => [value as number, "Count"]}
            labelFormatter={(label) => `Residual bin: ${label}`}
          />
          <ReferenceLine x="0" stroke="#b08a30" strokeDasharray="4 2" />
          <Bar dataKey="count" fill="#4a6fa5" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
