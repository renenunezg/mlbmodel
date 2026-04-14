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

const features = [
  {
    category: "Pitching",
    rows: [
      { name: "xfip", desc: "Starter Expected FIP — park-neutral pitching quality", source: "Statcast" },
      { name: "starter_whip", desc: "Starter walks + hits per inning pitched", source: "Statcast" },
      { name: "xfip_bullpen", desc: "IP-weighted bullpen xFIP", source: "Statcast" },
      { name: "bullpen_k_9", desc: "IP-weighted bullpen strikeouts per 9 innings", source: "Statcast" },
    ],
  },
  {
    category: "Batting Splits",
    rows: [
      { name: "batting_ops", desc: "Team OPS vs opponent pitcher handedness (60/40 blend)", source: "Statcast" },
      { name: "batting_iso", desc: "Team isolated power (SLG − AVG) — pure extra-base hit rate", source: "Statcast" },
      { name: "batting_k_pct", desc: "Team strikeout percentage vs pitcher handedness", source: "Statcast" },
    ],
  },
  {
    category: "Rolling Performance",
    rows: [
      { name: "avg_last5", desc: "5-game rolling average runs scored", source: "DB" },
      { name: "avg_last10", desc: "10-game rolling average runs scored", source: "DB" },
      { name: "std_last5", desc: "5-game rolling standard deviation — scoring volatility", source: "DB" },
    ],
  },
  {
    category: "Context",
    rows: [
      { name: "park_factor", desc: "Venue run-scoring environment (100 = neutral)", source: "Savant" },
      { name: "is_home", desc: "Home field advantage indicator (0/1)", source: "MLB API" },
    ],
  },
];

const pipelineSteps = [
  { num: "01", name: "Schedule & Scores", desc: "Fetch 3-day window of schedules, upsert games, finalize scores" },
  { num: "02", name: "Statcast Stats", desc: "Compute pitcher, bullpen, and batting stats from pitch-level data" },
  { num: "03", name: "Park Factors", desc: "Load ballpark run environment factors (cached per season)" },
  { num: "04", name: "Odds", desc: "Fetch moneyline, run line, and totals; match to games by team + time" },
  { num: "05", name: "Model", desc: "Train XGBoost, predict xR per team, write to model_outputs" },
  { num: "06", name: "Evaluation", desc: "Score predictions vs actuals, update accuracy + calibration tables" },
];

const stack = [
  {
    category: "ML / Data",
    items: ["Python 3.13", "XGBoost", "scikit-learn", "pybaseball", "pandas", "NumPy", "SciPy"],
  },
  {
    category: "Database",
    items: ["Supabase (PostgreSQL)", "SQLAlchemy"],
  },
  {
    category: "Frontend",
    items: ["Next.js 16", "TypeScript", "Tailwind CSS", "shadcn/ui", "Recharts"],
  },
  {
    category: "Infrastructure",
    items: ["GitHub Actions (daily cron)", "MLB Stats API", "The Odds API", "Baseball Savant"],
  },
];

export function MethodologyContent() {
  return (
    <div className="flex flex-col gap-6">
      {/* Overview */}
      <SectionCard
        id="overview"
        title="Project Overview"
        subtitle="End-to-end ML system for probabilistic run prediction and model evaluation against efficient markets"
      >
        <div className="space-y-4 text-sm leading-relaxed">
          <p>
            This project builds a complete machine learning pipeline that predicts expected runs
            scored per team per MLB game and evaluates those predictions against sportsbook-implied
            probabilities — not as a gambling exercise, but because sports betting markets function
            as one of the most efficient real-time probability benchmarks available. Sharp
            market participants force lines to near-true probabilities quickly, making them
            a higher-quality calibration target than any single statistical model alone.
          </p>
          <p>
            The betting-adjacent metrics here — expected value, Kelly criterion, probability
            calibration — are applied entirely as <strong>quantitative finance and decision-theory
            concepts</strong>. Kelly criterion is a portfolio optimization formula from information
            theory. Expected value is the foundation of rational decision-making under uncertainty.
            Probability calibration is a core model evaluation technique. The sports context is the
            domain; the underlying methods are those of applied statistics and quantitative analysis.
          </p>
          <p>
            The goal was to build something genuinely end-to-end: automated data ingestion from
            multiple APIs, feature engineering from pitch-level Statcast data, probabilistic
            inference with calibrated outputs, daily evaluation against actual results, and a
            production frontend that displays everything in real time.
          </p>
          <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
            {[
              { label: "Prediction target", val: "Expected runs per team per game" },
              { label: "Model", val: "XGBoost regressor, 12 features" },
              { label: "Win probability", val: "Negative binomial distribution (r = 6)" },
              { label: "Calibration", val: "Isotonic regression on OOF predictions" },
              { label: "Market comparison", val: "Quarter-Kelly criterion (portfolio theory)" },
              { label: "Pipeline cadence", val: "Daily cron at 6 AM PST via GitHub Actions" },
            ].map(({ label, val }) => (
              <div key={label} className="rounded-sm border border-border p-3">
                <p className="font-mono text-xs uppercase tracking-wider text-muted-foreground">{label}</p>
                <p className="mt-0.5 text-sm font-medium">{val}</p>
              </div>
            ))}
          </div>
        </div>
      </SectionCard>

      {/* Pipeline */}
      <SectionCard
        id="pipeline"
        title="Daily Pipeline"
        subtitle="Six-step automated pipeline running every morning before the first pitch"
      >
        <div className="space-y-4 text-sm">
          <p className="leading-relaxed text-muted-foreground">
            The pipeline runs as a GitHub Actions cron job at 14:00 UTC (6 AM PST), before
            early games tip off. Each step is timed and logged; failures are captured with
            full tracebacks but don&apos;t abort downstream steps.
          </p>

          {/* Step flow — vertical on mobile, horizontal on desktop */}
          <div className="mt-4 flex flex-col gap-2 md:flex-row md:items-start md:gap-0">
            {pipelineSteps.map((step, i) => (
              <div key={step.num} className="flex md:flex-1 md:flex-col">
                <div className="flex md:flex-col md:items-center">
                  <div className="flex flex-col md:items-center">
                    <div className="flex items-start gap-3 md:flex-col md:items-center md:gap-1">
                      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-border bg-muted font-mono text-xs text-muted-foreground">
                        {step.num}
                      </span>
                      <div className="md:text-center">
                        <p className="font-medium leading-tight">{step.name}</p>
                        <p className="mt-0.5 text-xs text-muted-foreground leading-snug md:mx-auto md:max-w-[120px]">
                          {step.desc}
                        </p>
                      </div>
                    </div>
                  </div>
                  {/* Arrow */}
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
            <p className="mb-2 font-mono text-xs uppercase tracking-wider text-muted-foreground">Data Sources</p>
            <div className="flex flex-wrap gap-2">
              {["MLB Stats API", "Statcast / pybaseball", "Baseball Savant", "The Odds API"].map((src) => (
                <Badge key={src} variant="outline">{src}</Badge>
              ))}
            </div>
          </div>
        </div>
      </SectionCard>

      {/* Feature Engineering */}
      <SectionCard
        id="features"
        title="Feature Engineering"
        subtitle="12 features across four categories, all computed from raw pitch-level Statcast data"
      >
        <div className="space-y-4 text-sm">
          <p className="leading-relaxed text-muted-foreground">
            FanGraphs was the original data source for advanced pitching stats, but it blocks
            automated requests via Cloudflare. All features are now computed directly from
            Statcast pitch-level data using{" "}
            <span className="font-mono">pybaseball</span> — giving full control over the
            formulas and removing a brittle external dependency.
          </p>
          <p className="leading-relaxed text-muted-foreground">
            Batting split features use a <strong>60/40 handedness blend</strong>: 60% weight
            on the team&apos;s splits vs the starting pitcher&apos;s hand, 40% on bullpen (assumed
            60% RHP league-wide). This approximates real plate appearance distribution across
            a full game. Early-season fallbacks use league-average values when fewer than 10
            games of data exist.
          </p>

          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Feature</TableHead>
                <TableHead className="hidden sm:table-cell">Category</TableHead>
                <TableHead className="whitespace-normal">Description</TableHead>
                <TableHead className="hidden md:table-cell">Source</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {features.map((group) =>
                group.rows.map((row, i) => (
                  <TableRow key={row.name}>
                    <TableCell className="font-mono text-xs">{row.name}</TableCell>
                    <TableCell className="hidden sm:table-cell text-muted-foreground">
                      {i === 0 ? group.category : ""}
                    </TableCell>
                    <TableCell className="whitespace-normal font-sans text-xs text-muted-foreground">
                      {row.desc}
                    </TableCell>
                    <TableCell className="hidden md:table-cell text-muted-foreground">{row.source}</TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>
      </SectionCard>

      {/* Model Architecture */}
      <SectionCard
        id="model"
        title="Model Architecture"
        subtitle="XGBoost regressor trained with temporally-aware cross-validation"
      >
        <div className="space-y-4 text-sm leading-relaxed">
          <p>
            The model predicts expected runs as a regression target (
            <span className="font-mono">reg:squarederror</span>) rather than a
            classification, which lets downstream logic derive win probabilities via
            distributional assumptions rather than baking them into the model directly.
          </p>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div>
              <p className="mb-2 font-mono text-xs uppercase tracking-wider text-muted-foreground">
                Hyperparameter Search Space
              </p>
              <div className="rounded-sm border border-border">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="px-3 py-2 text-left font-mono text-xs uppercase tracking-wider text-muted-foreground">Parameter</th>
                      <th className="px-3 py-2 text-left font-mono text-xs uppercase tracking-wider text-muted-foreground">Values</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { param: "n_estimators", vals: "100, 200, 300" },
                      { param: "max_depth", vals: "3, 4, 5" },
                      { param: "learning_rate", vals: "0.05, 0.10" },
                      { param: "min_child_weight", vals: "3, 5" },
                    ].map(({ param, vals }) => (
                      <tr key={param} className="border-b border-border last:border-0">
                        <td className="px-3 py-1.5 font-mono">{param}</td>
                        <td className="px-3 py-1.5 text-muted-foreground">{vals}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="mt-1.5 text-xs text-muted-foreground">36 combinations evaluated per run</p>
            </div>

            <div className="space-y-3">
              <div>
                <p className="mb-1.5 font-mono text-xs uppercase tracking-wider text-muted-foreground">
                  Cross-Validation
                </p>
                <p className="text-muted-foreground">
                  <strong className="text-foreground">TimeSeriesSplit</strong> with 5 folds,
                  games sorted chronologically. Training windows only ever see past data —
                  no future leakage. Minimum 60 samples required; folds reduced
                  dynamically for early-season sparsity.
                </p>
              </div>
              <div>
                <p className="mb-1.5 font-mono text-xs uppercase tracking-wider text-muted-foreground">
                  Optimization Metric
                </p>
                <p className="text-muted-foreground">
                  Negative mean absolute error (MAE). Chosen over RMSE because run-scoring
                  outliers (blowout games) shouldn&apos;t dominate gradient updates.
                </p>
              </div>
            </div>
          </div>
        </div>
      </SectionCard>

      {/* Win Probability */}
      <SectionCard
        id="probability"
        title="Win Probability"
        subtitle="Negative binomial joint distribution over all possible final scores"
      >
        <div className="space-y-4 text-sm leading-relaxed">
          <p>
            Given expected runs λ for each team, win probability is derived from a{" "}
            <strong>negative binomial distribution</strong> rather than the simpler Poisson.
            Baseball run-scoring exhibits overdispersion — the variance in runs scored exceeds
            the mean across games. Poisson forces variance = mean, producing probabilities
            that are systematically overconfident. The negative binomial adds a dispersion
            parameter <span className="font-mono">r</span> that relaxes this constraint.
          </p>

          <FormulaBlock>
            P(X = k) = C(k+r−1, k) · p^r · (1−p)^k<br />
            <span className="text-muted-foreground">where  p = r / (r + λ),  r = 6  (calibrated to MLB run distributions)</span>
          </FormulaBlock>

          <p>
            A joint probability matrix is computed for all combinations of home/away scores
            from 0–25 runs. From this matrix, three probabilities are derived:
          </p>
          <ul className="ml-4 space-y-1 text-muted-foreground">
            <li>
              <strong className="text-foreground">Win probability</strong> — P(home &gt; away) +
              P(tie) × λ_home / (λ_home + λ_away)
            </li>
            <li>
              <strong className="text-foreground">Cover probability</strong> — P(home margin &gt; spread), pushes split 50/50
            </li>
            <li>
              <strong className="text-foreground">Over/under probability</strong> — P(total &gt; line), pushes split 50/50
            </li>
          </ul>
          <p className="text-muted-foreground">
            All outputs are clipped to [0.05, 0.95] to prevent degenerate Kelly fractions
            from extreme probability estimates.
          </p>
        </div>
      </SectionCard>

      {/* Calibration */}
      <SectionCard
        id="calibration"
        title="Probability Calibration"
        subtitle="Isotonic regression on out-of-fold predictions eliminates in-sample leakage"
      >
        <div className="space-y-4 text-sm leading-relaxed">
          <p>
            Raw model win probabilities are calibrated using{" "}
            <strong>isotonic regression</strong> — a non-parametric, monotone-constrained
            method that maps predicted probabilities to empirical win rates without
            assuming any functional form.
          </p>
          <p>
            Crucially, the calibrator is fit only on{" "}
            <strong>out-of-fold (OOF) predictions</strong> from the TimeSeriesSplit CV folds.
            This means the calibration mapping never sees the same data the model was trained
            on, preventing the calibrated outputs from being optimistically biased.
          </p>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            {[
              { label: "Method", val: "Isotonic regression" },
              { label: "Training data", val: "OOF predictions only" },
              { label: "Threshold", val: "≥ 400 outcomes to activate" },
            ].map(({ label, val }) => (
              <div key={label} className="rounded-sm border border-border p-3">
                <p className="font-mono text-xs uppercase tracking-wider text-muted-foreground">{label}</p>
                <p className="mt-0.5 text-sm">{val}</p>
              </div>
            ))}
          </div>
          <p className="text-muted-foreground">
            After calibration, complementary probabilities are renormalized so that
            P(home win) + P(away win) = 1 per game. The calibration curve is tracked
            live on the{" "}
            <Link href="/performance" className="underline underline-offset-2 hover:text-foreground">
              Performance page
            </Link>
            .
          </p>
        </div>
      </SectionCard>

      {/* Betting */}
      <SectionCard
        id="betting"
        title="Quantitative Evaluation & Market Comparison"
        subtitle="Kelly criterion as a portfolio theory benchmark; walk-forward backtesting on historical seasons"
      >
        <div className="space-y-4 text-sm leading-relaxed">
          <p>
            When the model&apos;s implied probability diverges from the sportsbook&apos;s implied
            probability, that gap is the <strong>edge</strong> — a measure of how much the
            model disagrees with the market consensus. Tracking edge and its realized accuracy
            over time is how model quality is evaluated in practice: not just &ldquo;was the
            prediction close?&rdquo; but &ldquo;did the model identify cases where the market
            was systematically wrong?&rdquo;
          </p>
          <p>
            Position sizing uses the{" "}
            <strong>Kelly criterion</strong> — a formula from information theory and portfolio
            optimization that determines the theoretically optimal allocation to maximize
            long-run logarithmic growth. It is widely used in quantitative finance (Thorp,
            Shannon) and serves here as a principled framework for weighting predictions by
            confidence, not as a gambling strategy:
          </p>
          <FormulaBlock>
            f* = (p · b − q) / b<br />
            <span className="text-muted-foreground">
              where  p = model win prob,  q = 1 − p,  b = decimal odds − 1
            </span>
          </FormulaBlock>
          <p>
            Full Kelly is mathematically optimal but practically aggressive — a short losing
            streak can draw down significantly. The implementation uses{" "}
            <strong>quarter-Kelly</strong> (f* × 0.25), a standard industry convention that
            trades some long-run growth for substantially reduced variance.
          </p>

          <div>
            <p className="mb-2 font-mono text-xs uppercase tracking-wider text-muted-foreground">
              Evaluation Metrics
            </p>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
              {[
                "Win accuracy (ML, RL, O/U)",
                "Mean absolute error (xR vs actual)",
                "Brier score",
                "Log-loss",
                "ROI by edge bucket",
                "Equity curve (unit P&L)",
              ].map((m) => (
                <div key={m} className="flex items-start gap-1.5 text-xs text-muted-foreground">
                  <span className="mt-0.5 shrink-0 text-foreground">·</span>
                  {m}
                </div>
              ))}
            </div>
          </div>

          <p className="text-muted-foreground">
            Historical validation uses a{" "}
            <strong className="text-foreground">walk-forward backtest</strong> over full
            MLB seasons: the model trains on all games before a 7-day window, predicts
            that window, advances, and repeats — mimicking live deployment. Results are
            visible on the{" "}
            <Link href="/performance" className="underline underline-offset-2 hover:text-foreground">
              Performance dashboard
            </Link>
            .
          </p>
        </div>
      </SectionCard>

      {/* Tech Stack */}
      <SectionCard
        id="stack"
        title="Tech Stack"
        subtitle="Production tools across data science, backend, and frontend"
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
            <strong className="text-foreground">Note on data sourcing:</strong> FanGraphs
            was the original source for advanced pitching stats but blocks automated requests
            via Cloudflare. All stats (xFIP, WHIP, K/9, batting splits) are now computed
            directly from Statcast pitch-level data via <span className="font-mono">pybaseball</span>,
            giving full formula control and eliminating a brittle scraping dependency.
            Prior-season stats are cached to{" "}
            <span className="font-mono">cache/</span> on first run (~30 min) and reused on
            subsequent days.
          </div>
        </div>
      </SectionCard>
    </div>
  );
}
