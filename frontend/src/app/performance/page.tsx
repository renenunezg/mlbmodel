import { supabase } from "@/lib/supabase";
import type { ModelEvaluation } from "@/lib/types";
import { AccuracyChart } from "@/components/accuracy-chart";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

export const revalidate = 300;

function pct(value: number | undefined | null): string {
  if (value == null) return "\u2014";
  return `${(value * 100).toFixed(1)}%`;
}

export default async function PerformancePage() {
  const { data: evaluations } = await supabase
    .from("model_evaluation")
    .select("*")
    .order("date", { ascending: true });

  const rows = (evaluations ?? []) as ModelEvaluation[];

  if (rows.length === 0) {
    return (
      <main className="mx-auto max-w-5xl px-4 py-8">
        <h1 className="font-heading text-2xl tracking-tight">
          Model Performance
        </h1>
        <p className="mt-4 text-muted-foreground">
          No evaluation data available yet. Run the pipeline to generate
          performance metrics.
        </p>
      </main>
    );
  }

  const latest = rows[rows.length - 1];

  return (
    <main className="mx-auto max-w-5xl px-4 py-8 space-y-8">
      <h1 className="font-heading text-2xl tracking-tight">
        Model Performance
      </h1>

      {/* KPI inline metrics */}
      <div className="flex items-baseline gap-8 font-mono text-sm">
        <div className="flex items-baseline gap-1.5">
          <span className="text-xs uppercase tracking-wider text-muted-foreground">ML Accuracy</span>
          <span className="font-bold tabular-nums">{pct(latest.ml_accuracy)}</span>
        </div>
        <div className="flex items-baseline gap-1.5">
          <span className="text-xs uppercase tracking-wider text-muted-foreground">Run Line</span>
          <span className="font-bold tabular-nums">{pct(latest.run_line_accuracy)}</span>
        </div>
        <div className="flex items-baseline gap-1.5">
          <span className="text-xs uppercase tracking-wider text-muted-foreground">Overall</span>
          <span className="font-bold tabular-nums">{pct(latest.total_accuracy)}</span>
        </div>
      </div>

      {/* Chart */}
      <div className="border-t border-border pt-6">
        <h2 className="font-heading text-lg mb-4">Accuracy Over Time</h2>
        <AccuracyChart data={rows} />
      </div>

      {/* Stats table */}
      <div className="border-t border-border pt-6">
        <h2 className="font-heading text-lg mb-4">Evaluation History</h2>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Date</TableHead>
              <TableHead>Overall</TableHead>
              <TableHead>ML</TableHead>
              <TableHead>Run Line</TableHead>
              <TableHead className="text-right">Avg Total Diff</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {[...rows].reverse().map((row) => (
              <TableRow key={row.date}>
                <TableCell className="font-medium">{row.date}</TableCell>
                <TableCell>
                  {row.total_correct}/{row.total_predictions}{" "}
                  <span className="text-muted-foreground">
                    ({pct(row.total_accuracy)})
                  </span>
                </TableCell>
                <TableCell>
                  {row.ml_correct}/{row.ml_predictions}{" "}
                  <span className="text-muted-foreground">
                    ({pct(row.ml_accuracy)})
                  </span>
                </TableCell>
                <TableCell>
                  {row.run_line_correct}/{row.run_line_predictions}{" "}
                  <span className="text-muted-foreground">
                    ({pct(row.run_line_accuracy)})
                  </span>
                </TableCell>
                <TableCell className="text-right">
                  {row.average_total_diff?.toFixed(2) ?? "\u2014"}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </main>
  );
}
