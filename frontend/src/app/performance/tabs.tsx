"use client";

import type {
  ModelEvaluation,
  CalibrationBin,
  FeatureImportance,
  EdgeBucket,
} from "@/lib/types";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { KpiCard } from "@/components/kpi-card";
import { AccuracyChart } from "@/components/accuracy-chart";
import { MetricLineChart } from "@/components/metric-line-chart";
import { CalibrationChart } from "@/components/calibration-chart";
import { EquityCurveChart } from "@/components/equity-curve-chart";
import { FeatureImportanceChart } from "@/components/feature-importance-chart";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface PerformanceTabsProps {
  evaluations: ModelEvaluation[];
  calibration: CalibrationBin[];
  featureImportance: FeatureImportance[];
  edgeBuckets: EdgeBucket[];
}

function pct(value: number | null | undefined): string {
  if (value == null) return "\u2014";
  return `${(value * 100).toFixed(1)}%`;
}

function fmt(value: number | null | undefined, decimals = 3): string {
  if (value == null) return "\u2014";
  return value.toFixed(decimals);
}

function fmtSigned(value: number | null | undefined, decimals = 2): string {
  if (value == null) return "\u2014";
  const s = value.toFixed(decimals);
  return value >= 0 ? `+${s}` : s;
}

export function PerformanceTabs({
  evaluations,
  calibration,
  featureImportance,
  edgeBuckets,
}: PerformanceTabsProps) {
  // Split evaluations by window type
  const dailyEvals = evaluations.filter(
    (e) => !e.eval_window || e.eval_window === "day"
  );
  const seasonEvals = evaluations.filter((e) => e.eval_window === "season");

  // Latest season row for headline KPIs
  const latest =
    seasonEvals.length > 0
      ? seasonEvals[seasonEvals.length - 1]
      : dailyEvals[dailyEvals.length - 1];

  // Latest day row for daily detail
  const latestDay =
    dailyEvals.length > 0 ? dailyEvals[dailyEvals.length - 1] : latest;

  // Latest calibration data (most recent date)
  const latestCalDate =
    calibration.length > 0 ? calibration[0].date : null;
  const latestCalibration = latestCalDate
    ? calibration
        .filter((c) => c.date === latestCalDate)
        .sort((a, b) => a.bin_mid - b.bin_mid)
    : [];

  // Latest feature importance (most recent date)
  const latestFeatDate =
    featureImportance.length > 0 ? featureImportance[0].date : null;
  const latestFeatures = latestFeatDate
    ? featureImportance.filter((f) => f.date === latestFeatDate)
    : [];

  // Latest edge buckets (season window, most recent date)
  const seasonBuckets = edgeBuckets.filter((b) => b.eval_window === "season");
  const latestBucketDate =
    seasonBuckets.length > 0 ? seasonBuckets[0].date : null;
  const latestBuckets = latestBucketDate
    ? seasonBuckets.filter((b) => b.date === latestBucketDate)
    : [];

  return (
    <Tabs defaultValue="overview">
      <TabsList className="mb-6">
        <TabsTrigger value="overview">Overview</TabsTrigger>
        <TabsTrigger value="regression">Regression</TabsTrigger>
        <TabsTrigger value="probabilistic">Probabilistic</TabsTrigger>
        <TabsTrigger value="betting">Betting</TabsTrigger>
        <TabsTrigger value="diagnostics">Diagnostics</TabsTrigger>
      </TabsList>

      {/* ============================================================ */}
      {/* OVERVIEW TAB */}
      {/* ============================================================ */}
      <TabsContent value="overview" className="space-y-8">
        <div className="flex flex-wrap items-baseline gap-x-5 gap-y-3 font-mono text-sm">
          <KpiCard label="ROI" value={pct(latest?.roi)} />
          <KpiCard label="Sharpe" value={fmt(latest?.sharpe, 2)} />
          <KpiCard label="Max DD" value={pct(latest?.max_drawdown)} />
          <KpiCard label="Brier" value={fmt(latest?.brier_score)} />
          <KpiCard label="MAE" value={fmt(latest?.mae)} />
          <KpiCard
            label="Overall"
            value={pct(latest?.total_accuracy)}
            sub={`${latest?.total_correct ?? 0}/${latest?.total_predictions ?? 0}`}
          />
        </div>

        <div className="border-t border-border pt-6">
          <h2 className="font-heading text-lg mb-4">Accuracy Over Time</h2>
          <AccuracyChart data={dailyEvals} />
        </div>

        <div className="border-t border-border pt-6">
          <h2 className="font-heading text-lg mb-4">Evaluation History</h2>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Date</TableHead>
                <TableHead>Overall</TableHead>
                <TableHead>ML</TableHead>
                <TableHead>Run Line</TableHead>
                <TableHead className="text-right">MAE</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {[...dailyEvals].reverse().map((row) => (
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
                  <TableCell className="text-right font-mono tabular-nums">
                    {fmt(row.mae)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </TabsContent>

      {/* ============================================================ */}
      {/* REGRESSION TAB */}
      {/* ============================================================ */}
      <TabsContent value="regression" className="space-y-8">
        <div className="flex flex-wrap items-baseline gap-x-5 gap-y-3 font-mono text-sm">
          <KpiCard label="MAE" value={fmt(latest?.mae)} />
          <KpiCard label="RMSE" value={fmt(latest?.rmse)} />
          <KpiCard label="R&#178;" value={fmt(latest?.r2)} />
          <KpiCard label="MAPE" value={latest?.mape != null ? `${latest.mape.toFixed(1)}%` : "\u2014"} />
        </div>

        <div className="border-t border-border pt-6">
          <h2 className="font-heading text-lg mb-4">MAE Over Time</h2>
          <MetricLineChart
            data={dailyEvals}
            dataKey="mae"
            name="MAE"
            color="#b08a30"
          />
        </div>

        <div className="border-t border-border pt-6">
          <h2 className="font-heading text-lg mb-4">RMSE Over Time</h2>
          <MetricLineChart
            data={dailyEvals}
            dataKey="rmse"
            name="RMSE"
            color="#4a6fa5"
          />
        </div>
      </TabsContent>

      {/* ============================================================ */}
      {/* PROBABILISTIC TAB */}
      {/* ============================================================ */}
      <TabsContent value="probabilistic" className="space-y-8">
        <div className="flex flex-wrap items-baseline gap-x-5 gap-y-3 font-mono text-sm">
          <KpiCard label="Brier Score" value={fmt(latest?.brier_score)} sub="Lower is better (baseline: 0.250)" />
          <KpiCard label="Log Loss" value={fmt(latest?.log_loss)} />
          <KpiCard label="Sharpness" value={fmt(latest?.sharpness, 4)} sub="Higher = more decisive" />
          <KpiCard label="80% Coverage" value={pct(latest?.interval_coverage_80)} sub="Target: 80%" />
        </div>

        <div className="border-t border-border pt-6">
          <h2 className="font-heading text-lg mb-4">Calibration Curve</h2>
          {latestCalibration.length > 0 ? (
            <>
              <p className="text-xs text-muted-foreground mb-2">
                Season-to-date as of {latestCalDate}. Dashed = perfect calibration.
              </p>
              <CalibrationChart data={latestCalibration} />
            </>
          ) : (
            <p className="text-muted-foreground text-sm">
              No calibration data yet.
            </p>
          )}
        </div>

        <div className="border-t border-border pt-6">
          <h2 className="font-heading text-lg mb-4">Brier Score Over Time</h2>
          <MetricLineChart
            data={dailyEvals}
            dataKey="brier_score"
            name="Brier Score"
            color="#2d7a4f"
          />
        </div>
      </TabsContent>

      {/* ============================================================ */}
      {/* BETTING TAB */}
      {/* ============================================================ */}
      <TabsContent value="betting" className="space-y-8">
        <div className="flex flex-wrap items-baseline gap-x-5 gap-y-3 font-mono text-sm">
          <KpiCard label="ROI" value={pct(latest?.roi)} />
          <KpiCard label="Sharpe" value={fmt(latest?.sharpe, 2)} />
          <KpiCard label="Sortino" value={fmt(latest?.sortino, 2)} />
          <KpiCard label="Max Drawdown" value={pct(latest?.max_drawdown)} />
          <KpiCard
            label="P&L"
            value={`${fmtSigned(latest?.net_profit_units)}u`}
            sub={`${fmt(latest?.total_staked_units, 2)}u staked`}
          />
        </div>

        <div className="border-t border-border pt-6">
          <h2 className="font-heading text-lg mb-4">Equity Curve</h2>
          {dailyEvals.some((d) => d.equity_end_units != null) ? (
            <EquityCurveChart data={dailyEvals.filter((d) => d.equity_end_units != null)} />
          ) : (
            <p className="text-muted-foreground text-sm">
              No equity data yet.
            </p>
          )}
        </div>

        <div className="border-t border-border pt-6">
          <h2 className="font-heading text-lg mb-4">Hit Rate by Edge Bucket</h2>
          {latestBuckets.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Edge Bucket</TableHead>
                  <TableHead className="text-right">Bets</TableHead>
                  <TableHead className="text-right">Hit Rate</TableHead>
                  <TableHead className="text-right">ROI</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {latestBuckets.map((b) => (
                  <TableRow key={b.bucket_label}>
                    <TableCell className="font-medium">{b.bucket_label}</TableCell>
                    <TableCell className="text-right font-mono tabular-nums">
                      {b.n_bets}
                    </TableCell>
                    <TableCell className="text-right font-mono tabular-nums">
                      {pct(b.hit_rate)}
                    </TableCell>
                    <TableCell className="text-right font-mono tabular-nums">
                      {pct(b.roi)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <p className="text-muted-foreground text-sm">
              No edge bucket data yet.
            </p>
          )}
        </div>
      </TabsContent>

      {/* ============================================================ */}
      {/* DIAGNOSTICS TAB */}
      {/* ============================================================ */}
      <TabsContent value="diagnostics" className="space-y-8">
        <div className="border-t border-border pt-6">
          <h2 className="font-heading text-lg mb-4">Feature Importance</h2>
          {latestFeatures.length > 0 ? (
            <>
              <p className="text-xs text-muted-foreground mb-2">
                XGBoost gain-based importance as of {latestFeatDate}.
              </p>
              <FeatureImportanceChart data={latestFeatures} />
            </>
          ) : (
            <p className="text-muted-foreground text-sm">
              No feature importance data yet.
            </p>
          )}
        </div>

        <div className="border-t border-border pt-6">
          <h2 className="font-heading text-lg mb-4">Prediction Interval Coverage</h2>
          <div className="flex flex-wrap items-baseline gap-x-5 gap-y-3 font-mono text-sm">
            <KpiCard
              label="80% Interval"
              value={pct(latest?.interval_coverage_80)}
              sub="% of actual outcomes inside the predicted 80% NB interval"
            />
          </div>
        </div>

        <div className="border-t border-border pt-6">
          <h2 className="font-heading text-lg mb-4">Daily Metric Stability</h2>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <h3 className="text-sm font-medium mb-2">MAE (daily)</h3>
              <MetricLineChart data={dailyEvals} dataKey="mae" name="MAE" color="#b08a30" />
            </div>
            <div>
              <h3 className="text-sm font-medium mb-2">Brier Score (daily)</h3>
              <MetricLineChart data={dailyEvals} dataKey="brier_score" name="Brier" color="#2d7a4f" />
            </div>
          </div>
        </div>
      </TabsContent>
    </Tabs>
  );
}
