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
  const [evalRes, calRes, featRes, edgeRes] = await Promise.all([
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
  ]);

  const evaluations = (evalRes.data ?? []) as ModelEvaluation[];
  const calibration = (calRes.data ?? []) as CalibrationBin[];
  const featureImportance = (featRes.data ?? []) as FeatureImportance[];
  const edgeBuckets = (edgeRes.data ?? []) as EdgeBucket[];

  if (evaluations.length === 0) {
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

  return (
    <main className="mx-auto max-w-5xl px-4 py-8 space-y-6">
      <h1 className="font-heading text-2xl tracking-tight">
        Model Performance
      </h1>
      <PerformanceTabs
        evaluations={evaluations}
        calibration={calibration}
        featureImportance={featureImportance}
        edgeBuckets={edgeBuckets}
      />
    </main>
  );
}
