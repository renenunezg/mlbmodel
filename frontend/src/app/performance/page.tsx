import { supabase } from "@/lib/supabase";
import type {
  ModelEvaluation,
  CalibrationBin,
  FeatureImportance,
  EdgeBucket,
} from "@/lib/types";
import { PerformanceTabs } from "./tabs";

export const revalidate = 300;

export default async function PerformancePage() {
  // Fetch all data in parallel
  const [evalRes, calRes, featRes, edgeRes, residRes] = await Promise.all([
    supabase
      .from("model_evaluation")
      .select("*")
      .order("date", { ascending: true }),
    supabase
      .from("model_calibration")
      .select("*")
      .order("date", { ascending: false })
      .limit(10),
    supabase
      .from("model_feature_importance")
      .select("*")
      .order("date", { ascending: false })
      .limit(20),
    supabase
      .from("model_edge_buckets")
      .select("*")
      .order("date", { ascending: false })
      .limit(40),
    supabase
      .from("model_outputs_season")
      .select(
        "team, expected_runs, games!inner(home_team, home_score, away_score, status)"
      )
      .eq("games.status", "Final")
      .order("date", { ascending: false })
      .limit(1000),
  ]);

  const evaluations = (evalRes.data ?? []) as ModelEvaluation[];
  const calibration = (calRes.data ?? []) as CalibrationBin[];
  const featureImportance = (featRes.data ?? []) as FeatureImportance[];
  const edgeBuckets = (edgeRes.data ?? []) as EdgeBucket[];

  type ResidualRow = {
    team: string;
    expected_runs: number | null;
    games:
      | {
          home_team: string;
          home_score: number | null;
          away_score: number | null;
          status: string;
        }
      | {
          home_team: string;
          home_score: number | null;
          away_score: number | null;
          status: string;
        }[]
      | null;
  };
  const residuals: number[] = ((residRes.data ?? []) as unknown as ResidualRow[])
    .map((r) => {
      const g = Array.isArray(r.games) ? r.games[0] : r.games;
      if (!g || r.expected_runs == null) return null;
      const actual =
        r.team === g.home_team ? g.home_score : g.away_score;
      if (actual == null) return null;
      return actual - r.expected_runs;
    })
    .filter((v): v is number => v != null);

  if (evaluations.length === 0) {
    return (
      <main className="mx-auto w-full max-w-5xl min-w-0 px-4 py-8">
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

  return (
    <main className="mx-auto w-full max-w-5xl min-w-0 px-4 py-8 space-y-6">
      <h1 className="font-heading text-2xl tracking-tight">
        Model Performance
      </h1>
      <PerformanceTabs
        evaluations={evaluations}
        calibration={calibration}
        featureImportance={featureImportance}
        edgeBuckets={edgeBuckets}
        residuals={residuals}
      />
    </main>
  );
}
