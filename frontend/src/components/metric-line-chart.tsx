"use client";

import type { ModelEvaluation } from "@/lib/types";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import { V2_CUTOVER_DATE } from "@/lib/constants";

interface MetricLineChartProps {
  data: ModelEvaluation[];
  dataKey: keyof ModelEvaluation;
  name: string;
  color?: string;
  formatter?: (value: number) => string;
}

function formatDateLabel(dateStr: string) {
  const date = new Date(dateStr + "T00:00:00");
  return date.toLocaleDateString("en-US", { month: "short", day: "2-digit" });
}

export function MetricLineChart({
  data,
  dataKey,
  name,
  color = "#2d7a4f",
  formatter = (v: number) => v.toFixed(3),
}: MetricLineChartProps) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data} margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
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
          tickFormatter={(v: number) => formatter(v)}
        />
        <Tooltip
          formatter={(value) => formatter(Number(value))}
          labelFormatter={(label) => formatDateLabel(String(label))}
          contentStyle={{
            backgroundColor: "#ffffff",
            border: "1px solid #d4d4d8",
            borderRadius: "2px",
            fontFamily: "var(--font-geist-mono)",
            fontSize: "12px",
          }}
        />
        <ReferenceLine
          x={V2_CUTOVER_DATE}
          stroke="#10b981"
          strokeDasharray="4 2"
          label={{ value: "v2", position: "insideTopRight", fill: "#10b981", fontSize: 10, fontFamily: "var(--font-geist-mono)" }}
        />
        <Line
          type="monotone"
          dataKey={dataKey as string}
          name={name}
          stroke={color}
          strokeWidth={1.5}
          dot={false}
          activeDot={{ r: 3 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
