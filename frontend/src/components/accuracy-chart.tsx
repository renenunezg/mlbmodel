"use client";

import { useState } from "react";
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

type SeriesKey = "ml_accuracy" | "run_line_accuracy" | "totals_accuracy" | "total_accuracy";

function formatDateLabel(dateStr: string) {
  const date = new Date(dateStr + "T00:00:00");
  return date.toLocaleDateString("en-US", { month: "short", day: "2-digit" });
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(0)}%`;
}

export function AccuracyChart({ data }: AccuracyChartProps) {
  const [isolated, setIsolated] = useState<SeriesKey | null>(null);
  const isHidden = (key: SeriesKey) => isolated !== null && isolated !== key;
  const handleLegendClick = (entry: { dataKey?: string | number }) => {
    const key = entry.dataKey as SeriesKey | undefined;
    if (!key) return;
    setIsolated((prev) => (prev === key ? null : key));
  };

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart
        data={data}
        margin={{ top: 8, right: 16, left: 8, bottom: 8 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
        <XAxis
          dataKey="date"
          tickFormatter={formatDateLabel}
          tick={{ fill: "#52525b", fontSize: 11, fontFamily: "var(--font-geist-mono)" }}
          stroke="#d4d4d8"
        />
        <YAxis
          domain={[0, 1]}
          tickFormatter={formatPercent}
          tick={{ fill: "#52525b", fontSize: 11, fontFamily: "var(--font-geist-mono)" }}
          stroke="#d4d4d8"
        />
        <Tooltip
          formatter={(value) => formatPercent(Number(value))}
          labelFormatter={(label) => formatDateLabel(String(label))}
          contentStyle={{
            backgroundColor: "#ffffff",
            border: "1px solid #d4d4d8",
            borderRadius: "2px",
            color: "#18181b",
            fontFamily: "var(--font-geist-mono)",
            fontSize: "12px",
          }}
        />
        <Legend
          wrapperStyle={{ fontFamily: "var(--font-geist-mono)", fontSize: "11px", cursor: "pointer" }}
          onClick={handleLegendClick}
        />
        <Line
          type="monotone"
          dataKey="ml_accuracy"
          name="ML Accuracy"
          stroke="#2d7a4f"
          strokeWidth={1.5}
          dot={false}
          activeDot={{ r: 3 }}
          hide={isHidden("ml_accuracy")}
        />
        <Line
          type="monotone"
          dataKey="run_line_accuracy"
          name="Run Line"
          stroke="#4a6fa5"
          strokeWidth={1.5}
          dot={false}
          activeDot={{ r: 3 }}
          hide={isHidden("run_line_accuracy")}
        />
        <Line
          type="monotone"
          dataKey="totals_accuracy"
          name="Totals"
          stroke="#9b59b6"
          strokeWidth={1.5}
          dot={false}
          activeDot={{ r: 3 }}
          hide={isHidden("totals_accuracy")}
        />
        <Line
          type="monotone"
          dataKey="total_accuracy"
          name="Pick Acc"
          stroke="#b08a30"
          strokeWidth={1.5}
          strokeDasharray="4 2"
          dot={false}
          activeDot={{ r: 3 }}
          hide={isHidden("total_accuracy")}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
