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
      .from("games")
      .select("game_pk, home_team, home_score, away_score")
      .eq("status", "Final")
      .order("game_date", { ascending: false })
      .limit(500),
  ]);

  const evaluations = (evalRes.data ?? []) as ModelEvaluation[];
  const calibration = (calRes.data ?? []) as CalibrationBin[];
  const featureImportance = (featRes.data ?? []) as FeatureImportance[];
  const edgeBuckets = (edgeRes.data ?? []) as EdgeBucket[];

  // Residuals: fetch model_outputs_season for graded game_pks and join client-side.
  // PostgREST embedded !inner requires an FK constraint, which these tables lack.
  type FinalGame = {
    game_pk: number;
    home_team: string;
    home_score: number | null;
    away_score: number | null;
  };
  const finalGames = (residRes.data ?? []) as FinalGame[];
  const gameByPk = new Map(finalGames.map((g) => [g.game_pk, g]));

  let residuals: number[] = [];
  if (finalGames.length > 0) {
    const predRes = await supabase
      .from("model_outputs_season")
      .select("game_pk, team, expected_runs")
      .in("game_pk", finalGames.map((g) => g.game_pk));

    residuals = (predRes.data ?? [])
      .map((r: { game_pk: number; team: string; expected_runs: number | null }) => {
        const g = gameByPk.get(r.game_pk);
        if (!g || r.expected_runs == null) return null;
        const actual = r.team === g.home_team ? g.home_score : g.away_score;
        if (actual == null) return null;
        return actual - r.expected_runs;
      })
      .filter((v): v is number => v != null);
  }

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
