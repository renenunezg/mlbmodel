"use client";

import type { FeatureImportance } from "@/lib/types";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface FeatureImportanceChartProps {
  data: FeatureImportance[];
}

export function FeatureImportanceChart({ data }: FeatureImportanceChartProps) {
  const sorted = [...data].sort((a, b) => b.importance - a.importance);

  return (
    <ResponsiveContainer width="100%" height={Math.max(300, sorted.length * 32)}>
      <BarChart
        data={sorted}
        layout="vertical"
        margin={{ top: 8, right: 16, left: 100, bottom: 8 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" horizontal={false} />
        <XAxis
          type="number"
          tick={{ fill: "#52525b", fontSize: 11, fontFamily: "var(--font-geist-mono)" }}
          stroke="#d4d4d8"
          tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
        />
        <YAxis
          type="category"
          dataKey="feature"
          tick={{ fill: "#52525b", fontSize: 11, fontFamily: "var(--font-geist-mono)" }}
          stroke="#d4d4d8"
          width={90}
        />
        <Tooltip
          formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`}
          contentStyle={{
            backgroundColor: "#ffffff",
            border: "1px solid #d4d4d8",
            borderRadius: "2px",
            fontFamily: "var(--font-geist-mono)",
            fontSize: "12px",
          }}
        />
        <Bar
          dataKey="importance"
          name="Importance"
          fill="#4a6fa5"
          radius={[0, 2, 2, 0]}
        />
      </BarChart>
    </ResponsiveContainer>
  );
}
