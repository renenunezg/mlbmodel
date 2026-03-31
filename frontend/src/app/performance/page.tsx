import { supabase } from "@/lib/supabase";
import type { ModelEvaluation } from "@/lib/types";
import { AccuracyChart } from "@/components/accuracy-chart";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
        <h1 className="text-2xl font-bold tracking-tight">
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
    <main className="mx-auto max-w-5xl px-4 py-8">
      <h1 className="mb-6 text-2xl font-bold tracking-tight">
        Model Performance
      </h1>

      {/* KPI cards */}
      <div className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm text-muted-foreground">
              ML Accuracy
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-[#198754]">
              {pct(latest.ml_accuracy)}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-sm text-muted-foreground">
              Run Line Accuracy
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-[#3b82f6]">
              {pct(latest.run_line_accuracy)}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-sm text-muted-foreground">
              Overall Accuracy
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-[#f59e0b]">
              {pct(latest.total_accuracy)}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Chart */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle>Accuracy Over Time</CardTitle>
        </CardHeader>
        <CardContent>
          <AccuracyChart data={rows} />
        </CardContent>
      </Card>

      {/* Stats table */}
      <Card>
        <CardHeader>
          <CardTitle>Evaluation History</CardTitle>
        </CardHeader>
        <CardContent>
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
        </CardContent>
      </Card>
    </main>
  );
}
