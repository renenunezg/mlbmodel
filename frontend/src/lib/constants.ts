// Model cutover boundary. Predictions on/after this date come from the v2
// Bayesian + Monte Carlo model. Earlier predictions are v1 XGBoost.
export const V2_CUTOVER_DATE = "2026-05-12";

export function isV2Era(dateStr: string | null | undefined): boolean {
  if (!dateStr) return false;
  return dateStr.slice(0, 10) >= V2_CUTOVER_DATE;
}

// Returns the smallest date >= V2_CUTOVER among the given rows. Use this to
// mark the boundary row in any table with a one-time "v2 starts" badge,
// regardless of whether the table is sorted ASC or DESC.
export function getFirstV2Date(
  rows: { date?: string | null }[],
): string | null {
  let firstDate: string | null = null;
  for (const r of rows) {
    if (!r.date) continue;
    const d = r.date.slice(0, 10);
    if (!isV2Era(d)) continue;
    if (firstDate === null || d < firstDate) firstDate = d;
  }
  return firstDate;
}
