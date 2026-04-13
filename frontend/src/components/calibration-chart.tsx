"use client";

import type { CalibrationBin } from "@/lib/types";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

interface CalibrationChartProps {
  data: CalibrationBin[];
}

export function CalibrationChart({ data }: CalibrationChartProps) {
  return (
    <ResponsiveContainer width="100%" height={350}>
      <LineChart data={data} margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
        <XAxis
          dataKey="predicted_mean"
          label={{ value: "Predicted", position: "insideBottom", offset: -2, fill: "#52525b", fontSize: 11 }}
          domain={[0, 1]}
          tick={{ fill: "#52525b", fontSize: 11, fontFamily: "var(--font-geist-mono)" }}
          stroke="#d4d4d8"
          tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
        />
        <YAxis
          label={{ value: "Observed", angle: -90, position: "insideLeft", fill: "#52525b", fontSize: 11 }}
          domain={[0, 1]}
          tick={{ fill: "#52525b", fontSize: 11, fontFamily: "var(--font-geist-mono)" }}
          stroke="#d4d4d8"
          tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
        />
        <ReferenceLine
          segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
          stroke="#a1a1aa"
          strokeDasharray="4 4"
          ifOverflow="extendDomain"
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
        <Line
          type="monotone"
          dataKey="observed_rate"
          name="Observed Win Rate"
          stroke="#2d7a4f"
          strokeWidth={2}
          dot={{ r: 4, fill: "#2d7a4f" }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
