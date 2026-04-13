"use client";

import type { ModelEvaluation } from "@/lib/types";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

interface EquityCurveChartProps {
  data: ModelEvaluation[];
}

function formatDateLabel(dateStr: string) {
  const date = new Date(dateStr + "T00:00:00");
  return date.toLocaleDateString("en-US", { month: "short", day: "2-digit" });
}

export function EquityCurveChart({ data }: EquityCurveChartProps) {
  return (
    <ResponsiveContainer width="100%" height={350}>
      <AreaChart data={data} margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
        <XAxis
          dataKey="date"
          tickFormatter={formatDateLabel}
          tick={{ fill: "#52525b", fontSize: 11, fontFamily: "var(--font-geist-mono)" }}
          stroke="#d4d4d8"
        />
        <YAxis
          tick={{ fill: "#52525b", fontSize: 11, fontFamily: "var(--font-geist-mono)" }}
          stroke="#d4d4d8"
          tickFormatter={(v: number) => `${v.toFixed(2)}u`}
        />
        <ReferenceLine y={1} stroke="#a1a1aa" strokeDasharray="4 4" />
        <Tooltip
          formatter={(value) => `${Number(value).toFixed(4)} units`}
          labelFormatter={(label) => formatDateLabel(String(label))}
          contentStyle={{
            backgroundColor: "#ffffff",
            border: "1px solid #d4d4d8",
            borderRadius: "2px",
            fontFamily: "var(--font-geist-mono)",
            fontSize: "12px",
          }}
        />
        <Area
          type="monotone"
          dataKey="equity_end_units"
          name="Equity"
          stroke="#2d7a4f"
          fill="#2d7a4f"
          fillOpacity={0.1}
          strokeWidth={2}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
