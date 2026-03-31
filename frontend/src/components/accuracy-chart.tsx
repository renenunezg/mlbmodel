"use client";

import type { ModelEvaluation } from "@/lib/types";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface AccuracyChartProps {
  data: ModelEvaluation[];
}

function formatDateLabel(dateStr: string) {
  const date = new Date(dateStr + "T00:00:00");
  return date.toLocaleDateString("en-US", { month: "short", day: "2-digit" });
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(0)}%`;
}

export function AccuracyChart({ data }: AccuracyChartProps) {
  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart
        data={data}
        margin={{ top: 8, right: 16, left: 8, bottom: 8 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
        <XAxis
          dataKey="date"
          tickFormatter={formatDateLabel}
          tick={{ fill: "#a1a1aa", fontSize: 12 }}
          stroke="#555"
        />
        <YAxis
          domain={[0, 1]}
          tickFormatter={formatPercent}
          tick={{ fill: "#a1a1aa", fontSize: 12 }}
          stroke="#555"
        />
        <Tooltip
          formatter={(value) => formatPercent(Number(value))}
          labelFormatter={(label) => formatDateLabel(String(label))}
          contentStyle={{
            backgroundColor: "#1c1c1e",
            border: "1px solid #333",
            borderRadius: "8px",
            color: "#e4e4e7",
          }}
        />
        <Legend />
        <Line
          type="monotone"
          dataKey="ml_accuracy"
          name="ML Accuracy"
          stroke="#198754"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4 }}
        />
        <Line
          type="monotone"
          dataKey="run_line_accuracy"
          name="Run Line Accuracy"
          stroke="#3b82f6"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4 }}
        />
        <Line
          type="monotone"
          dataKey="total_accuracy"
          name="Overall Accuracy"
          stroke="#f59e0b"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
