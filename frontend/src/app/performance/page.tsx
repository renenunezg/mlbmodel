import { supabase } from "@/lib/supabase";
import type {
  ModelEvaluation,
  CalibrationBin,
  FeatureImportance,
  EdgeBucket,
  PosteriorSkill,
  PosteriorSigma,
  BetLedgerRow,
} from "@/lib/types";
import { aggregateLedger } from "@/lib/betting-aggs";
import { PerformanceTabs } from "./tabs";
import { LastUpdated } from "@/components/last-updated";
import { RealtimeRefresh } from "@/components/realtime-refresh";

export const revalidate = 300;

export default async function PerformancePage() {
  // Fetch all data in parallel
  const [evalRes, calRes, featRes, edgeRes, residRes, latestRes, skillsRes, sigmasRes, ledgerRes] = await Promise.all([
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
    supabase
      .from("model_evaluation")
      .select("created_at")
      .order("created_at", { ascending: false })
      .limit(1),
    supabase
      .from("posterior_skills")
      .select("*")
      .order("refit_date", { ascending: false })
      .limit(80),
    supabase
      .from("posterior_sigmas")
      .select("*")
      .order("refit_date", { ascending: false })
      .limit(20),
    // Live bet ledger - drives the headline betting KPIs without depending
    // on yesterday's model_evaluation snapshot. Range cap defeats the
    // Supabase JS default 1000-row limit.
    supabase
      .from("bet_ledger_v")
      .select("date, team, game_pk, bet_type, stake, decimal_odds, american_odds, totals_side, won, edge, payout")
      .range(0, 9999),
  ]);

  const evaluations = (evalRes.data ?? []) as ModelEvaluation[];
  const lastUpdated: string | null = latestRes.data?.[0]?.created_at ?? null;
  const calibration = (calRes.data ?? []) as CalibrationBin[];
  const featureImportance = (featRes.data ?? []) as FeatureImportance[];
  const edgeBuckets = (edgeRes.data ?? []) as EdgeBucket[];
  const posteriorSkills = (skillsRes.data ?? []) as PosteriorSkill[];
  const posteriorSigmas = (sigmasRes.data ?? []) as PosteriorSigma[];
  const ledger = (ledgerRes.data ?? []) as BetLedgerRow[];
  const liveKpis = aggregateLedger(ledger);

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
      <div className="flex items-start justify-between gap-4">
        <h1 className="font-heading text-2xl tracking-tight">
          Model Performance
        </h1>
        <LastUpdated
          timestamp={lastUpdated}
          schedule="Updates nightly ~midnight PT"
        />
      </div>
      <PerformanceTabs
        evaluations={evaluations}
        calibration={calibration}
        featureImportance={featureImportance}
        edgeBuckets={edgeBuckets}
        residuals={residuals}
        posteriorSkills={posteriorSkills}
        posteriorSigmas={posteriorSigmas}
        liveKpis={liveKpis}
      />
      <RealtimeRefresh tables={["model_evaluation"]} />
    </main>
  );
}
