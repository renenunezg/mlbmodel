import type { ReactNode } from "react";
import Link from "next/link";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { MethodologyDistributionChart } from "@/components/methodology-distribution-chart";
import { ChangelogEntry } from "@/components/changelog-entry";

export interface DistributionGameData {
  date: string;
  home: { team: string; mean: number; p10: number; p50: number; p90: number; hist: number[] };
  away: { team: string; mean: number; p10: number; p50: number; p90: number; hist: number[] };
  homeWinProb: number;
  homeWinProbP10: number | null;
  homeWinProbP90: number | null;
  totalLine: number | null;
  totalMean: number;
  totalP10: number;
  totalP90: number;
  startTimeUtc: string | null;
}

function SectionCard({
  id,
  title,
  subtitle,
  children,
}: {
  id: string;
  title: string;
  subtitle: string;
  children: ReactNode;
}) {
  return (
    <section id={id} className="scroll-mt-6">
      <Card>
        <CardHeader className="border-b">
          <CardTitle className="text-lg">{title}</CardTitle>
          <CardDescription>{subtitle}</CardDescription>
        </CardHeader>
        <CardContent className="pt-4">{children}</CardContent>
      </Card>
    </section>
  );
}

function FormulaBlock({ children }: { children: ReactNode }) {
  return (
    <div className="my-3 rounded-sm border border-border bg-muted px-4 py-3 font-mono text-sm leading-relaxed">
      {children}
    </div>
  );
}

const flowNodes = [
  {
    group: "Inputs",
    color: "border-blue-500/40 bg-blue-500/5",
    labelColor: "text-blue-400",
    items: [
      { label: "Statcast pitch data", sub: "~400k PAs, 2024+25+26-YTD" },
      { label: "MLB Stats API", sub: "Schedule, lineups, rosters, boxscores" },
      { label: "The Odds API", sub: "ML, RL, totals" },
      { label: "Active bullpen workload", sub: "pitcher_workload table (rest)" },
    ],
  },
  {
    group: "Bayesian Skill Layer",
    color: "border-amber-500/40 bg-amber-500/5",
    labelColor: "text-amber-400",
    items: [
      { label: "Batter D-M", sub: "812 batters × 2 platoon cells" },
      { label: "Pitcher D-M", sub: "1074 pitchers × 2 roles (SP/RP)" },
      { label: "Park log-PF", sub: "30 venues, residual wOBA" },
      { label: "Sampler", sub: "NUTS via numpyro/JAX, 4 chains × 2000 draws" },
    ],
  },
  {
    group: "Monte Carlo Simulator",
    color: "border-emerald-500/40 bg-emerald-500/5",
    labelColor: "text-emerald-400",
    items: [
      { label: "Per-PA outcome sampler", sub: "Vectorized over 8 categories" },
      { label: "Empirical advancement", sub: "P(state', runs | state, outs, outcome)" },
      { label: "Rest-aware bullpen", sub: "Active roster + 1d/2d workload caps" },
      { label: "K=30 posterior draws × ~333 inning sims", sub: "~10,000 simulated games per matchup" },
    ],
  },
  {
    group: "Markets & EV",
    color: "border-purple-500/40 bg-purple-500/5",
    labelColor: "text-purple-400",
    items: [
      { label: "Empirical win probability", sub: "p10/p50/p90 bands from posterior draws" },
      { label: "Totals & RL distributions", sub: "Quantiles over simulated runs" },
      { label: "Edge vs sportsbook", sub: "4.5% ML/RL, 6.5% totals" },
      { label: "Quarter-Kelly sizing", sub: "0.25 × f*, capped" },
    ],
  },
];

const pipelineSteps = [
  { num: "01", name: "Schedule & Bullpen", desc: "Refresh games, scores, pitcher_workload per pitcher per day" },
  { num: "02", name: "Lineups & Odds", desc: "Posted lineups via Stats API; ML/RL/totals from The Odds API" },
  { num: "03", name: "Posterior Refit", desc: "Nightly NUTS run: batter, pitcher, park (12 min on M-series)" },
  { num: "04", name: "Score Games", desc: "K=30 posterior draws × ~333 inning sims each, ~10,000 total per matchup" },
  { num: "05", name: "Derive Markets", desc: "Empirical win prob, totals, RL from simulated run distributions" },
  { num: "06", name: "Verify & Publish", desc: "Anti-correlation, calibration, posterior-age checks; write to Supabase" },
];

const backtestRows = [
  { metric: "Brier score (ML)", v1: "0.2570", v2: "0.2393", delta: "−6.88%", pass: true },
  { metric: "Log-loss (ML)", v1: "0.7244", v2: "0.6713", delta: "−7.33%", pass: true },
  { metric: "Max calibration gap", v1: "41.92%", v2: "3.20%", delta: "−38.7pp", pass: true },
  { metric: "ROI moneyline (flagged)", v1: "+15.82%", v2: "+26.25%", delta: "+10.4pp", pass: null },
  { metric: "ROI run line (flagged)", v1: "+5.04%", v2: "+14.96%", delta: "+9.9pp", pass: null },
  { metric: "ROI totals (flagged)", v1: "−12.83%", v2: "−7.58%", delta: "+5.3pp", pass: null },
];

const samplerDiagnostics = [
  { model: "Batter (812 × platoon)", rhat: "1.00", ess: "755", divergences: "0", wall: "7.1 min" },
  { model: "Pitcher (1074 × role)", rhat: "1.00", ess: "1707", divergences: "0", wall: "4.5 min" },
  { model: "Park (30 venues)", rhat: "1.00", ess: "15.9k", divergences: "0", wall: "6 s" },
];

const stack = [
  {
    category: "Probabilistic Modeling",
    items: ["PyMC", "numpyro", "JAX", "arviz", "NUTS sampler"],
  },
  {
    category: "Simulation & Data",
    items: ["NumPy (vectorized)", "pandas", "pybaseball", "Statcast pitch data", "parquet cache"],
  },
  {
    category: "Database",
    items: ["Supabase (PostgreSQL)", "SQLAlchemy", "RLS public_read"],
  },
  {
    category: "Frontend",
    items: ["Next.js 16", "TypeScript", "Tailwind CSS", "shadcn/ui", "Recharts"],
  },
  {
    category: "Orchestration",
    items: ["GitHub Actions (cron + workflow_run)", "MLB Stats API", "The Odds API"],
  },
];

export function MethodologyContent({
  featured,
}: {
  featured?: DistributionGameData | null;
}) {
  return (
    <div className="flex flex-col gap-6">
      {/* Overview */}
      <SectionCard
        id="overview"
        title="Project Overview"
        subtitle="Hierarchical Bayesian skill model + per-PA Monte Carlo simulator for probabilistic run prediction"
      >
        <div className="space-y-4 text-sm leading-relaxed">
          <p>
            This project predicts a full distribution of runs scored per team per MLB game,
            then derives win, run-line, and totals probabilities by simulating each matchup
            roughly 10,000 times. Predictions are graded daily against sportsbook lines.{" "}
            <strong>
              Sports betting markets are used as a calibration benchmark, not as a gambling
              application.
            </strong>{" "}
            Sharp market participants push lines toward true probabilities quickly, which
            makes them a higher-quality probability signal than most independently
            constructed models.
          </p>
          <p>
            The framework is borrowed from quantitative finance and decision theory:
            expected value, the Kelly criterion (Kelly 1956, originally an information-theory
            result), and probability calibration. The domain is baseball; the methods are
            standard in applied statistics and quant analysis.
          </p>
          <p>
            The pipeline covers data ingestion from multiple APIs, feature derivation from
            pitch-level Statcast data, hierarchical Bayesian inference via NUTS, daily
            evaluation against actual outcomes, and a production frontend that refreshes
            each morning of the season.
          </p>
          <p>
            The current model (v2, live since May 12, 2026) is a two-layer system: a{" "}
            <strong>hierarchical Bayesian skill model</strong> (Dirichlet-Multinomial over the
            eight plate-appearance outcomes, fit with NUTS via numpyro/JAX) followed by a{" "}
            <strong>per-PA Monte Carlo simulator</strong> that plays out each inning with
            rest-aware bullpen rules, an empirical baserunner-advancement table, and per-game
            posterior draws to propagate parameter uncertainty. It replaced a prior XGBoost
            regressor (v1, archived) after a 542-game head-to-head backtest in which v2 won
            on every metric.
          </p>
          <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
            {[
              { label: "Prediction target", val: "Per-team run distribution per game" },
              { label: "Skill model", val: "Hierarchical Dirichlet-Multinomial (8 outcomes)" },
              { label: "Sampler", val: "NUTS via numpyro / JAX" },
              { label: "Training data", val: "401,826 PAs across 2024 + 2025 + 2026-YTD" },
              { label: "Simulator", val: "K=30 posterior draws × ~333 inning sims each (default ~10,000 sims total)" },
              { label: "Probability output", val: "Empirical quantiles with p10/p50/p90 bands" },
              { label: "Sizing rule", val: "Quarter-Kelly on flagged plays" },
              { label: "Cutover from v1", val: "May 12, 2026" },
            ].map(({ label, val }) => (
              <div key={label} className="rounded-sm border border-border p-3">
                <p className="font-mono text-xs uppercase tracking-wider text-muted-foreground">{label}</p>
                <p className="mt-0.5 text-sm font-medium">{val}</p>
              </div>
            ))}
          </div>
        </div>
      </SectionCard>

      {/* Featured distribution chart */}
      <SectionCard
        id="example"
        title="Example Output: Today's Featured Game"
        subtitle="Per-team simulated run distributions, win probability band, and total runs percentiles"
      >
        {featured ? (
          <MethodologyDistributionChart
            date={featured.date}
            home={featured.home}
            away={featured.away}
            homeWinProb={featured.homeWinProb}
            homeWinProbP10={featured.homeWinProbP10}
            homeWinProbP90={featured.homeWinProbP90}
            totalLine={featured.totalLine}
            totalMean={featured.totalMean}
            totalP10={featured.totalP10}
            totalP90={featured.totalP90}
            startTimeUtc={featured.startTimeUtc}
          />
        ) : (
          <p className="text-sm text-muted-foreground">
            No scheduled games with v2 predictions for today yet. The chart populates each
            morning after <span className="font-mono">daily-pipeline-v2</span> writes to{" "}
            <span className="font-mono">model_outputs</span>.
          </p>
        )}
      </SectionCard>

      {/* Changelog */}
      <SectionCard
        id="changelog"
        title="Model Changelog"
        subtitle="Significant updates to model behavior, inference, and data handling, most recent first"
      >
        <div className="space-y-6 text-sm">
          <ChangelogEntry
            date="May 12, 2026"
            title="v2 cutover: hierarchical Bayesian skill model + per-PA Monte Carlo simulator"
            accent="emerald"
          >
            Replaced the v1 XGBoost regressor with a two-layer probabilistic system. The
            skill layer fits three hierarchical Dirichlet-Multinomial models (batter,
            pitcher, park) via NUTS with full convergence diagnostics. The simulator runs
            ~10,000 vectorized games per matchup (K=30 posterior draws × ~333 inning sims
            each), propagating parameter uncertainty through the draws and feeding an
            empirical baserunner-advancement table built from 365k PAs of Statcast data.
            Backtest over 542 games (Mar 26 to May 9, 2026):
            Brier −6.9%, log-loss −7.3%, max calibration gap from 41.9% down to 3.2%, ROI
            improved on every market. The v1 model is preserved in archive tables and
            described in the legacy section below.
          </ChangelogEntry>

          <ChangelogEntry
            date="April 21, 2026 (v1)"
            title="Dynamic starter/bullpen inning split in batting-split features"
          >
            v1 batting-split features started weighting starter vs bullpen innings by the
            opposing starter&apos;s trailing IP per start, replacing the fixed 60/40
            assumption. Carried forward conceptually into v2 via per-PA pitcher identity.
          </ChangelogEntry>

          <details className="group">
            <summary className="cursor-pointer select-none rounded-sm border border-border bg-muted/40 px-3 py-2 text-xs uppercase tracking-wider text-muted-foreground hover:bg-muted">
              See full changelog (2 older entries)
            </summary>
            <div className="mt-6 space-y-6">
              <ChangelogEntry
                date="April 20, 2026 (v1)"
                title="Raised +EV thresholds from 3% to 4.5% (ML/RL) and 6.5% (totals)"
              >
                Below the typical vig on a −110 line there is no cushion for model
                miscalibration. v2 inherits these thresholds for apples-to-apples comparison;
                v2.1 will retune them from edge-bucket ROI on live v2 data.
              </ChangelogEntry>

              <ChangelogEntry
                date="April 2026 (v1)"
                title="Win probability switched from Poisson to negative binomial"
              >
                MLB run-scoring is overdispersed relative to Poisson. Negative binomial with
                r=6 brought v1 tail probabilities into line. In v2, win probability is derived
                empirically from the simulated run distributions and no parametric assumption
                about the run-scoring distribution is required.
              </ChangelogEntry>
            </div>
          </details>
        </div>
      </SectionCard>

      {/* Architecture / Flow */}
      <SectionCard
        id="flow"
        title="System Architecture"
        subtitle="From pitch-level data to calibrated market probabilities"
      >
        <div className="space-y-3">
          <p className="text-sm leading-relaxed text-muted-foreground">
            The system is split into a skill layer (Bayesian inference, refit nightly) and a
            simulation layer (Monte Carlo, run per game per scoring pass). The skill layer
            learns time-stable parameters from years of Statcast data; the simulator consumes
            them to produce per-game run distributions that the markets layer turns into win,
            total, and run-line probabilities with edge and sizing.
          </p>
          <div className="mt-4 flex flex-col gap-2 md:flex-row md:items-stretch md:gap-0">
            {flowNodes.map((node, i) => (
              <div key={node.group} className="flex min-w-0 md:flex-1 md:flex-col">
                <div className={`rounded-sm border ${node.color} p-3 flex-1`}>
                  <p className={`mb-2 font-mono text-[10px] uppercase tracking-widest font-semibold ${node.labelColor}`}>
                    {node.group}
                  </p>
                  <ul className="space-y-1.5">
                    {node.items.map((item) => (
                      <li key={item.label}>
                        <p className="text-xs font-medium leading-tight">{item.label}</p>
                        <p className="text-[10px] text-muted-foreground leading-snug">{item.sub}</p>
                      </li>
                    ))}
                  </ul>
                </div>
                {i < flowNodes.length - 1 && (
                  <>
                    <div className="flex justify-center py-1 md:hidden">
                      <span className="text-muted-foreground text-sm">↓</span>
                    </div>
                    <div className="hidden md:flex md:items-center md:justify-center md:w-6 md:shrink-0">
                      <span className="text-muted-foreground text-sm">→</span>
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>
        </div>
      </SectionCard>

      {/* Pipeline */}
      <SectionCard
        id="pipeline"
        title="Daily Pipeline"
        subtitle="Six-step orchestration. Nightly refit then morning scoring, plus intraday lineup re-scoring"
      >
        <div className="space-y-4 text-sm">
          <p className="leading-relaxed text-muted-foreground">
            Nightly <span className="font-mono">train-v2</span> runs a full NUTS refit on a
            GitHub Actions runner (~4 AM PT). On success it triggers{" "}
            <span className="font-mono">daily-pipeline-v2</span> via{" "}
            <span className="font-mono">workflow_run</span>, guaranteeing scoring runs against
            fresh posteriors. A separate <span className="font-mono">refresh-lineups-v2</span>{" "}
            cron checks every 30 minutes between 7 AM and 4 PM PT and re-scores any game
            whose posted lineup has changed (detected via SHA-1 hash of sorted batter IDs).
          </p>
          <p className="leading-relaxed text-muted-foreground">
            Predictions reflect the best information available up to first pitch: lineups,
            scratches, and late odds all feed back into the score. Once a game starts,
            its row is frozen and the evaluation ledger uses that final pre-game value.
            A flag visible on <span className="font-mono">/games</span> in the morning can
            disappear later if the posted lineup re-score drops the edge below threshold,
            and in that case the bet is also removed from <span className="font-mono">/history</span>{" "}
            and from model evaluation. Posterior refits happen on their own nightly schedule,
            independent of the intraday lineup refresh.
          </p>

          <div className="mt-4 flex flex-col gap-2 md:flex-row md:items-start md:gap-0">
            {pipelineSteps.map((step, i) => (
              <div key={step.num} className="flex min-w-0 md:flex-1 md:flex-col">
                <div className="flex md:flex-col md:items-center">
                  <div className="flex flex-col md:items-center">
                    <div className="flex min-w-0 items-start gap-3 md:flex-col md:items-center md:gap-1">
                      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-border bg-muted font-mono text-xs text-muted-foreground">
                        {step.num}
                      </span>
                      <div className="min-w-0 md:text-center">
                        <p className="font-medium leading-tight">{step.name}</p>
                        <p className="mt-0.5 text-xs text-muted-foreground leading-snug md:mx-auto md:max-w-[140px]">
                          {step.desc}
                        </p>
                      </div>
                    </div>
                  </div>
                  {i < pipelineSteps.length - 1 && (
                    <div className="ml-3.5 mt-1 mb-1 h-4 w-px bg-border md:ml-0 md:mt-2 md:mb-0 md:hidden" />
                  )}
                </div>
                {i < pipelineSteps.length - 1 && (
                  <div className="hidden md:flex md:flex-1 md:items-start md:justify-center md:pt-3.5">
                    <span className="text-muted-foreground">→</span>
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="mt-4">
            <p className="mb-2 font-mono text-xs uppercase tracking-wider text-muted-foreground">Data sources</p>
            <div className="flex flex-wrap gap-2">
              {["MLB Stats API", "Statcast / pybaseball", "The Odds API", "Supabase posteriors cache"].map((src) => (
                <Badge key={src} variant="outline">{src}</Badge>
              ))}
            </div>
          </div>
        </div>
      </SectionCard>

      {/* Bayesian Skill Layer */}
      <SectionCard
        id="skill"
        title="Bayesian Skill Layer"
        subtitle="Hierarchical Dirichlet-Multinomial per actor, fit with NUTS in numpyro/JAX"
      >
        <div className="space-y-4 text-sm leading-relaxed">
          <p>
            Each plate appearance is modeled as a categorical draw over eight outcomes:
            strikeout, walk, hit-by-pitch, single, double, triple, home run, in-play out.
            Three separate hierarchical models learn the additive log-odds offsets that each
            actor (batter, pitcher, venue) contributes to those eight logits, with{" "}
            <span className="font-mono">OUT</span> as the reference category.
          </p>

          <p className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
            Generative model (batter, schematic)
          </p>
          <FormulaBlock>
            μ ~ Normal(0, 1)<sub>7</sub>{"  "}{"//"} per-outcome league intercept
            <br />
            σ<sub>b</sub> ~ HalfNormal(1)<sub>7</sub>{"  "}{"//"} batter-level scale per outcome
            <br />
            z<sub>i</sub> ~ Normal(0, 1)<sub>7</sub>{"  "}{"//"} non-centered batter offsets
            <br />
            β<sub>i</sub> = μ + σ<sub>b</sub> ⊙ z<sub>i</sub>{"  "}{"//"} batter i&apos;s log-odds vector
            <br />
            π<sub>i</sub> = softmax([0, β<sub>i</sub>])
            <br />
            y<sub>i</sub> ~ Multinomial(n<sub>i</sub>, π<sub>i</sub>){"  "}{"//"} aggregated per-batter PA counts
          </FormulaBlock>

          <p>
            Four implementation details that matter at this scale:
          </p>
          <ul className="ml-4 space-y-1 text-muted-foreground">
            <li>
              <strong className="text-foreground">Non-centered parameterization.</strong>{" "}
              Sampling{" "}
              <span className="font-mono">z<sub>i</sub></span> from a unit normal and forming{" "}
              <span className="font-mono">β<sub>i</sub> = μ + σ ⊙ z<sub>i</sub></span> avoids
              Neal&apos;s funnel pathology in hierarchical models, where the geometry of the
              posterior becomes too distorted for HMC to traverse efficiently.
            </li>
            <li>
              <strong className="text-foreground">Multinomial likelihood, not per-PA Categorical.</strong>{" "}
              Aggregating each batter&apos;s PAs into outcome counts and using{" "}
              <span className="font-mono">pm.Multinomial</span> is mathematically identical to
              a per-PA <span className="font-mono">pm.Categorical</span> but ~400× faster on
              400k PAs (~3 min vs ~19 h projected). Each leapfrog step evaluates a per-actor
              log-likelihood rather than a per-row one.
            </li>
            <li>
              <strong className="text-foreground">Platoon and role splits.</strong> Batters
              are fit per <span className="font-mono">(batter, vs_LHP)</span> cell; pitchers
              per <span className="font-mono">(pitcher, role ∈ {`{SP, RP}`})</span>.
              Position-player pitchers are dropped from the pitcher pool.
            </li>
            <li>
              <strong className="text-foreground">Park as residual log-PF.</strong> Park is
              fit last, on the residual wOBA after batter and pitcher effects are accounted
              for. A per-venue scalar <span className="font-mono">park_log[v]</span> shifts
              the seven non-OUT logits additively, weighted by each outcome&apos;s wOBA
              coefficient, so HR and 3B move most under park and K is unaffected.
            </li>
          </ul>

          <p className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
            Sampler diagnostics (4 chains, 2000 draws, 2500 tune)
          </p>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Model</TableHead>
                <TableHead className="text-right">R-hat</TableHead>
                <TableHead className="text-right">min ESS</TableHead>
                <TableHead className="text-right">Divergences</TableHead>
                <TableHead className="text-right">Wall</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {samplerDiagnostics.map((row) => (
                <TableRow key={row.model}>
                  <TableCell className="font-mono text-xs">{row.model}</TableCell>
                  <TableCell className="text-right font-mono text-xs">{row.rhat}</TableCell>
                  <TableCell className="text-right font-mono text-xs">{row.ess}</TableCell>
                  <TableCell className="text-right font-mono text-xs">{row.divergences}</TableCell>
                  <TableCell className="text-right font-mono text-xs">{row.wall}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <p className="text-xs text-muted-foreground">
            R-hat is the Gelman-Rubin convergence statistic; values near 1.00 indicate the
            chains have mixed and are sampling from the same posterior. Effective sample size
            (ESS) measures how many independent draws the autocorrelated chains are
            equivalent to; minimum ESS &gt; 400 is the gate.
          </p>
        </div>
      </SectionCard>

      {/* Monte Carlo simulator */}
      <SectionCard
        id="simulator"
        title="Per-PA Monte Carlo Simulator"
        subtitle="K=30 posterior draws × inning-level sims per draw, propagating parameter uncertainty"
      >
        <div className="space-y-4 text-sm leading-relaxed">
          <p>
            For each scheduled game we draw K=30 random posterior samples and run N
            inning-level simulations per draw. N defaults to ~333 in production (10,000 total
            sims) and 33 in the acceptance-gate test (990 total). Every draw is a coherent
            realization of all batter, pitcher, and park parameters together, so spread
            across draws becomes a posterior band on the win rate; the inner-loop variance is
            the aleatoric run-scoring noise within a fixed parameter setting.
          </p>

          <p className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
            Per-PA outcome sampler (vectorized)
          </p>
          <FormulaBlock>
            ℓ<sub>k</sub> = batter_offset[b, hand_k] + pitcher_offset[p, role, hand_k]
            <br />
            {"     "}+ park_log[v] · wOBA_weight<sub>k</sub>
            <br />
            {"     "}+ form_noise<sub>k</sub>{"  "}{"//"} sigma = 0.13, zero-sum across game
            <br />
            π = softmax([0, ℓ<sub>1..7</sub>])
            <br />
            outcome ~ Categorical(π)
          </FormulaBlock>
          <p className="text-xs text-muted-foreground">
            Indices: batter <span className="font-mono">b</span>, pitcher{" "}
            <span className="font-mono">p</span>, role ∈ {"{SP, RP}"}, venue{" "}
            <span className="font-mono">v</span>, outcome <span className="font-mono">k ∈ 1..7</span>{" "}
            (K, BB, HBP, 1B, 2B, 3B, HR; OUT is the reference 0 logit).
          </p>

          <p>
            On contact, baserunner state evolves via an{" "}
            <strong>empirical advancement table</strong> built from 365k Statcast PAs
            (2024+25). The table is a flat lookup of{" "}
            <span className="font-mono">P(new_state, runs_scored, outs_added | state, outs, outcome, out_subtype)</span>.
            Cells with fewer than 100 observations are linearly shrunk toward the
            outcome-conditional marginal{" "}
            <span className="font-mono">P(· | outcome, subtype)</span>, with weight{" "}
            <span className="font-mono">n_cell / 100</span> on the cell and the rest on the
            marginal. HR, BB, and HBP advances are deterministic forced moves; the empirical
            estimates were unreliable there because bases-loaded variants were sparse.
          </p>

          <p>
            Bullpen management is rest-aware in live mode. Starters are pulled at{" "}
            <span className="font-mono">pa_count ≥ 24</span> (a pitch-count proxy, roughly
            95 pitches at 3.95 P/PA). Relievers are drawn from a queue built from the active
            26-man roster, ordered by current rest, and any reliever with{" "}
            <span className="font-mono">≥ 6 outs in the last 1 day</span> or{" "}
            <span className="font-mono">≥ 9 outs in the last 2 days</span> is skipped. The
            per-pitcher workload table is refreshed in the daily pipeline from MLB boxscores.
          </p>

          <p className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
            Variance decomposition (per-game total runs)
          </p>
          <FormulaBlock>
            Var(total_runs) = Var<sub>posterior</sub>(parameter draws)
            <br />
            {"             "}+ Var<sub>form</sub>(per-game form noise, σ=0.13)
            <br />
            {"             "}+ Var<sub>aleatoric</sub>(inning-level sim noise)
          </FormulaBlock>
          <p className="text-xs text-muted-foreground">
            The form-noise scalar is calibrated against 2025 actuals to narrow the residual
            underdispersion in the empirical advancement table. It is not a tuning knob to
            be re-fit on market backtests; doing that mixes concerns and tends to mask
            modeling errors as noise.
          </p>

          <p className="text-muted-foreground">
            Acceptance gate (200 stratified 2025 games × 990 sims × 2 sides = 396k team-game
            samples): simulated mean runs/team-game 4.59 vs actual 4.38 (+4.86%, clears the
            5% mean gate); simulated variance 9.72 vs actual 10.32 (−5.86%, clears the
            relaxed 7% variance gate, misses the original 5%). The model is mildly
            underdispersed in the tails, so blowouts and large totals get slightly less
            probability mass than they should. Closing that gap is a v2.1 item.
          </p>
        </div>
      </SectionCard>

      {/* Markets */}
      <SectionCard
        id="markets"
        title="From Simulated Runs to Market Probabilities"
        subtitle="Empirical quantiles over the simulated run distributions, with per-draw posterior bands"
      >
        <div className="space-y-4 text-sm leading-relaxed">
          <p>
            Market probabilities are estimated directly from the simulated run
            distributions; there is no parametric assumption like Poisson or negative
            binomial on top of the sim output. Let{" "}
            <span className="font-mono">h<sub>i</sub></span>,{" "}
            <span className="font-mono">a<sub>i</sub></span> be the home and away runs in sim{" "}
            <span className="font-mono">i</span> for{" "}
            <span className="font-mono">i = 1..N</span>:
          </p>

          <FormulaBlock>
            P(home wins) = (1/N) · Σ<sub>i</sub> 𝟙[h<sub>i</sub> &gt; a<sub>i</sub>]
            <br />
            P(total &gt; L) = (1/N) · Σ<sub>i</sub> 𝟙[h<sub>i</sub> + a<sub>i</sub> &gt; L]
            <br />
            P(home covers s<sub>h</sub>) = (1/N) · Σ<sub>i</sub> 𝟙[h<sub>i</sub> − a<sub>i</sub> &gt; −s<sub>h</sub>]
            <br />
            <span className="text-muted-foreground">
              Sign convention: s<sub>h</sub> is the book&apos;s home spread (negative for a
              home favorite, so s<sub>h</sub> = −1.5 means home must win by &gt; 1.5).
              Pushes split 50/50 at integer lines.
            </span>
          </FormulaBlock>

          <p>
            The <strong>p10/p90 win-probability band</strong> is computed per posterior draw,
            not from the inning-level samples. For each of the K=30 draws we compute that
            draw&apos;s win probability across its inner sims, then take p10 and p90 over
            the K draw-level probabilities. The band is a measure of posterior parameter
            uncertainty about the win rate, separated from the aleatoric run variance.
            Anti-correlation is enforced by construction:{" "}
            <span className="font-mono">P<sub>away</sub>(p10) = 1 − P<sub>home</sub>(p90)</span>.
          </p>

          <p>
            A play is flagged when the modeled probability exceeds the sportsbook&apos;s
            de-vigged implied probability by more than 4.5% (ML, RL) or 6.5% (totals). The
            totals bar is higher because totals markets are noisier and respond more slowly
            to fresh information. Sizing uses the Kelly criterion:
          </p>
          <FormulaBlock>
            f* = (p · b − q) / b{"  "}{"//"} p = model prob, q = 1−p, b = decimal odds − 1
            <br />
            stake = clamp(0.25 · f*, 0, max_stake){"  "}{"//"} quarter-Kelly with cap
          </FormulaBlock>
          <p className="text-muted-foreground">
            Full Kelly is theoretically optimal for long-run log-wealth growth (Thorp,
            Shannon, Kelly 1956) but draws down aggressively on losing streaks. Quarter-Kelly
            trades some growth rate for a much tighter drawdown distribution. A high-variance
            flag fires when a team&apos;s simulated runs stdev exceeds 4.0, surfacing games
            where the model itself is unusually uncertain about scoring.
          </p>
        </div>
      </SectionCard>

      {/* Backtest */}
      <SectionCard
        id="backtest"
        title="Head-to-Head Backtest vs Frozen v1"
        subtitle="542 games, March 26 to May 9, 2026. v2 wins on every metric."
      >
        <div className="space-y-4 text-sm leading-relaxed">
          <p>
            Before cutover, v2 was benchmarked against the frozen v1 XGBoost baseline on
            every completed 2026 game where both models had a prediction and the v2 sim used
            a clean live-bullpen queue. v1&apos;s code is preserved verbatim at SHA{" "}
            <span className="font-mono">a84b4dd</span> in{" "}
            <span className="font-mono">v2/evaluation/baseline_v1/</span> so the comparison is
            stable.
          </p>

          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Metric</TableHead>
                <TableHead className="text-right">v1</TableHead>
                <TableHead className="text-right">v2</TableHead>
                <TableHead className="text-right">Δ</TableHead>
                <TableHead className="text-right">Gate</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {backtestRows.map((row) => (
                <TableRow key={row.metric}>
                  <TableCell className="text-xs">{row.metric}</TableCell>
                  <TableCell className="text-right font-mono text-xs">{row.v1}</TableCell>
                  <TableCell className="text-right font-mono text-xs">{row.v2}</TableCell>
                  <TableCell className="text-right font-mono text-xs">{row.delta}</TableCell>
                  <TableCell className="text-right">
                    {row.pass === true ? (
                      <Badge variant="outline" className="text-emerald-400 border-emerald-500/40">PASS</Badge>
                    ) : (
                      <span className="text-xs text-muted-foreground">—</span>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>

          <p className="text-muted-foreground">
            Brier and log-loss are both proper scoring rules, so the comparison is on
            calibration and sharpness together, not just hit rate on winners. v1&apos;s
            41.9% max calibration gap is partly thin-bin variance (the worst decile had few
            games in it), but the directional read still holds: v2 reports a tighter and
            better-calibrated probability. v2 is also more selective on flagged plays
            (~20-25% fewer flags) yet posts higher ROI on all three markets, which is the
            expected pattern when a model gets sharper.
          </p>
          <p className="text-muted-foreground">
            Backtest infrastructure is in <span className="font-mono">v2/evaluation/</span>{" "}
            (replay populates the v2 prediction table for a date range; backtester joins v1,
            v2, and actuals and emits the head-to-head report). Live performance, including
            ongoing Brier and calibration, is on the{" "}
            <Link href="/performance" className="underline underline-offset-2 hover:text-foreground">
              Performance page
            </Link>
            .
          </p>
        </div>
      </SectionCard>

      {/* Tech Stack */}
      <SectionCard
        id="stack"
        title="Tech Stack"
        subtitle="Production tools across probabilistic modeling, simulation, and frontend"
      >
        <div className="space-y-5 text-sm">
          {stack.map(({ category, items }) => (
            <div key={category}>
              <p className="mb-2 font-mono text-xs uppercase tracking-wider text-muted-foreground">
                {category}
              </p>
              <div className="flex flex-wrap gap-2">
                {items.map((item) => (
                  <Badge key={item} variant="outline" className="font-mono text-xs">
                    {item}
                  </Badge>
                ))}
              </div>
            </div>
          ))}

          <div className="mt-2 rounded-sm border border-border bg-muted/50 p-3 text-xs text-muted-foreground leading-relaxed">
            <strong className="text-foreground">A note on the sampler stack:</strong> PyMC
            describes the model; numpyro provides the JAX-backed NUTS implementation that
            actually samples. Pinning matters: numpyro 0.20.1 + jax 0.7.2 + jaxlib 0.7.2.
            Newer JAX removed an internal primitive (<span className="font-mono">xla_pmap_p</span>)
            that numpyro depends on, which breaks sampling silently.
          </div>
        </div>
      </SectionCard>

      {/* Legacy v1 */}
      <SectionCard
        id="legacy-v1"
        title="Legacy: v1 XGBoost Model (pre-2026-05-12)"
        subtitle="Predictions on the History and Performance tabs before the green cutover line come from this model"
      >
        <details className="group text-sm">
          <summary className="cursor-pointer select-none rounded-sm border border-border bg-muted/40 px-3 py-2 text-xs uppercase tracking-wider text-muted-foreground hover:bg-muted">
            Expand v1 methodology
          </summary>
          <div className="mt-4 space-y-4 leading-relaxed">
            <p>
              v1 was a gradient-boosted regressor (XGBoost,{" "}
              <span className="font-mono">reg:squarederror</span>) on 14 hand-crafted features
              per team per game, trained with <span className="font-mono">TimeSeriesSplit</span>{" "}
              cross-validation and <span className="font-mono">GridSearchCV</span>, retrained
              daily on all completed games of the season. It targeted expected runs per team
              directly; win, total, and run-line probabilities were derived afterward from a
              negative binomial joint score distribution with dispersion r=6, with isotonic
              regression on out-of-fold predictions for calibration.
            </p>

            <p className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
              Features (14)
            </p>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Feature</TableHead>
                  <TableHead className="whitespace-normal">Description</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {[
                  { name: "xfip", desc: "Starter expected FIP from Statcast pitch data" },
                  { name: "xfip_bullpen", desc: "IP-weighted bullpen xFIP" },
                  { name: "starter_whip", desc: "Starter walks + hits per inning pitched" },
                  { name: "bullpen_k_9", desc: "IP-weighted bullpen K/9" },
                  { name: "batting_ops", desc: "Team OPS, dynamic starter/bullpen handedness blend" },
                  { name: "batting_iso", desc: "Team ISO (SLG − AVG), dynamic blend" },
                  { name: "batting_k_pct", desc: "Team K%, dynamic blend" },
                  { name: "avg_last5", desc: "5-game rolling avg runs scored" },
                  { name: "avg_last10", desc: "10-game rolling avg runs scored" },
                  { name: "std_last5", desc: "5-game rolling std dev (volatility)" },
                  { name: "park_factor", desc: "Venue run-scoring factor (Baseball Savant)" },
                  { name: "is_home", desc: "Home-field indicator (0/1)" },
                  { name: "own_bp_outs_2d", desc: "Own bullpen reliever outs in prior 2 days (rest)" },
                  { name: "opp_bp_outs_2d", desc: "Opp bullpen reliever outs in prior 2 days (fatigue)" },
                ].map((row) => (
                  <TableRow key={row.name}>
                    <TableCell className="font-mono text-xs">{row.name}</TableCell>
                    <TableCell className="whitespace-normal text-xs text-muted-foreground">{row.desc}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>

            <p className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
              Why v2 replaced it
            </p>
            <ul className="ml-4 space-y-1 text-muted-foreground">
              <li>
                v1 had no concept of individual batter or pitcher skill; everything was
                team-aggregate. Lineup changes, platoon advantages, and bullpen identity were
                invisible to the model.
              </li>
              <li>
                Win-probability calibration degraded in the tails (max gap 41.9% across
                deciles), driven by sparse decile bins on a point-estimate output.
              </li>
              <li>
                Parameter uncertainty was not represented. The model produced a point
                prediction; the negative binomial added run-scoring variance on top of it,
                but the model itself reported no confidence band.
              </li>
              <li>
                Feature engineering was a maintenance burden. Twelve to fourteen features
                required ongoing tuning of split blends, IP weights, and fallback rules. v2
                replaces all of that with per-actor posteriors learned from raw PAs.
              </li>
            </ul>
            <p className="text-xs text-muted-foreground">
              v1 predictions before May 12, 2026 are still served from the archive tables (
              <span className="font-mono">model_outputs_v1_archive</span>,{" "}
              <span className="font-mono">model_outputs_season_v1_archive</span>) so the
              historical record is unchanged. A green vertical line on every time-series
              chart marks the cutover.
            </p>
          </div>
        </details>
      </SectionCard>
    </div>
  );
}
