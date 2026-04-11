import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatOdds(odds: number | null): string {
  if (odds == null) return "—";
  return odds > 0 ? `+${odds}` : `${odds}`;
}

export function formatPct(value: number | null, decimals = 1): string {
  if (value == null) return "—";
  return `${(value * 100).toFixed(decimals)}%`;
}

export function formatRuns(value: number | null): string {
  if (value == null) return "—";
  return value.toFixed(1);
}

export function formatDate(dateStr: string | null): string {
  if (!dateStr) return "—";
  const datePart = dateStr.slice(0, 10); // "YYYY-MM-DD" prefix works for all formats
  const d = new Date(datePart + "T12:00:00");
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

export function formatConfidence(value: number | null): string {
  if (value == null) return "—";
  return `${(value * 100).toFixed(1)}%`;
}
