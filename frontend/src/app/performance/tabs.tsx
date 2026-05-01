"use client";

import { useMemo, useState } from "react";
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
import { ResidualsChart } from "@/components/residuals-chart";
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
  residuals: number[];
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

function fmtAmerican(value: number | null | undefined): string {
  if (value == null) return "\u2014";
  const r = Math.round(value);
  return r > 0 ? `+${r}` : `${r}`;
}

export function PerformanceTabs({
  evaluations,
  calibration,
  featureImportance,
  edgeBuckets,
  residuals,
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
      <div className="mb-6 -mx-4 overflow-x-auto px-4 sm:mx-0 sm:overflow-visible sm:px-0">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="regression">Regression</TabsTrigger>
          <TabsTrigger value="probabilistic">Probabilistic</TabsTrigger>
          <TabsTrigger value="betting">Betting</TabsTrigger>
          <TabsTrigger value="diagnostics">Diagnostics</TabsTrigger>
        </TabsList>
      </div>

      {/* ============================================================ */}
      {/* OVERVIEW TAB */}
      {/* ============================================================ */}
      <TabsContent value="overview" className="space-y-8">
        <div className="grid grid-cols-3 sm:flex sm:flex-wrap items-baseline gap-x-4 gap-y-3 font-mono text-sm">
          <KpiCard
            label="ROI"
            value={pct(latest?.roi)}
            tooltip="Profit per dollar risked. ROI = total P&L ÷ total stakes. 13% ROI means 13¢ profit per $1 staked, on average — not 13% of your bankroll."
          />
          <KpiCard
            label="Sharpe"
            value={fmt(latest?.sharpe, 2)}
            tooltip="Risk-adjusted return: mean daily P&L ÷ std dev of daily P&L. >1 is good, >2 excellent."
          />
          <KpiCard
            label="Max DD"
            value={pct(latest?.max_drawdown)}
            tooltip="Maximum drawdown: largest peak-to-trough decline in cumulative units. The deepest hole the model dug during a losing stretch."
          />
          <KpiCard
            label="Brier"
            value={fmt(latest?.brier_score)}
            tooltip="Mean squared error of probabilistic predictions vs binary outcomes. Lower is better; 0.25 is the coin-flip baseline."
          />
          <KpiCard
            label="MAE"
            value={fmt(latest?.mae)}
            tooltip="Mean absolute error of expected runs vs actual runs. Lower is better; ~2.5 is typical for MLB run prediction."
          />
          <KpiCard
            label="Pick Acc"
            value={pct(latest?.total_accuracy)}
            sub={`model picks winner (${latest?.total_correct ?? 0}/${latest?.total_predictions ?? 0})`}
          />
        </div>

        <div className="border-t border-border pt-6">
          <h2 className="font-heading text-lg mb-4">Accuracy Over Time</h2>
          <AccuracyChart data={dailyEvals} />
        </div>

        <div className="border-t border-border pt-6">
          <h2 className="font-heading text-lg mb-4">Evaluation History</h2>
          <EvalHistoryTable rows={dailyEvals} />
        </div>
      </TabsContent>

      {/* ============================================================ */}
      {/* REGRESSION TAB */}
      {/* ============================================================ */}
      <TabsContent value="regression" className="space-y-8">
        <div className="grid grid-cols-3 sm:flex sm:flex-wrap items-baseline gap-x-4 gap-y-3 font-mono text-sm">
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
        <div className="grid grid-cols-3 sm:flex sm:flex-wrap items-baseline gap-x-4 gap-y-3 font-mono text-sm">
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
        <div className="grid grid-cols-3 sm:flex sm:flex-wrap items-baseline gap-x-4 gap-y-3 font-mono text-sm">
          <KpiCard
            label="ROI"
            value={pct(latest?.roi)}
            tooltip="Profit per dollar risked. ROI = total P&L ÷ total stakes. 13% ROI means the model returns 13¢ profit on every $1 staked, on average. Independent of bankroll size."
          />
          <KpiCard
            label="Sharpe"
            value={fmt(latest?.sharpe, 2)}
            tooltip="Risk-adjusted return: mean daily P&L ÷ std dev of daily P&L. Higher is better. >1 is good, >2 is excellent."
          />
          <KpiCard
            label="Sortino"
            value={fmt(latest?.sortino, 2)}
            tooltip="Like Sharpe but penalizes only downside volatility. Better metric for asymmetric strategies (gambling, where upside variance is fine)."
          />
          <KpiCard
            label="Max Drawdown"
            value={pct(latest?.max_drawdown)}
            tooltip="Largest peak-to-trough decline in cumulative units. A 20% drawdown means at the worst stretch you were down 20% from your previous high."
          />
          <KpiCard
            label="P&L"
            value={`${fmtSigned(latest?.net_profit_units)}u`}
            sub={`${fmt(latest?.total_staked_units, 2)}u staked`}
            tooltip="Net profit in units. 1 unit = your chosen bankroll size — if your bankroll is $100, 1u = $100. Stakes shown below are TOTAL summed across all bets in the window, not a single bet. Bankroll never compounds; each bet is sized as a fraction of a fixed 1u."
          />
          <KpiCard
            label="Favorites"
            value={pct(latest?.roi_favorites)}
            sub={`(${latest?.favorites_correct ?? 0}-${(latest?.n_favorites ?? 0) - (latest?.favorites_correct ?? 0)})`}
          />
          <KpiCard
            label="Underdogs"
            value={pct(latest?.roi_underdogs)}
            sub={`(${latest?.underdogs_correct ?? 0}-${(latest?.n_underdogs ?? 0) - (latest?.underdogs_correct ?? 0)})`}
          />
          <KpiCard label="Avg Line" value={fmtAmerican(latest?.avg_ml_line)} />
          <KpiCard
            label="Overs"
            value={pct(latest?.overs_roi)}
            sub={`(${latest?.overs_correct ?? 0}-${(latest?.overs_predictions ?? 0) - (latest?.overs_correct ?? 0)})`}
          />
          <KpiCard
            label="Unders"
            value={pct(latest?.unders_roi)}
            sub={`(${latest?.unders_correct ?? 0}-${(latest?.unders_predictions ?? 0) - (latest?.unders_correct ?? 0)})`}
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
          <h2 className="font-heading text-lg mb-4">Daily History</h2>
          {dailyEvals.length > 0 ? (
            <DailyBettingHistory rows={dailyEvals} />
          ) : (
            <p className="text-muted-foreground text-sm">
              No daily history yet.
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
          <div className="grid grid-cols-3 sm:flex sm:flex-wrap items-baseline gap-x-4 gap-y-3 font-mono text-sm">
            <KpiCard
              label="50% Interval"
              value={pct(latest?.interval_coverage_50)}
              sub="Target: 50%"
            />
            <KpiCard
              label="80% Interval"
              value={pct(latest?.interval_coverage_80)}
              sub="Target: 80%"
            />
            <KpiCard
              label="90% Interval"
              value={pct(latest?.interval_coverage_90)}
              sub="Target: 90%"
            />
          </div>
        </div>

        <div className="border-t border-border pt-6">
          <h2 className="font-heading text-lg mb-4">MAE Over Time</h2>
          <MetricLineChart data={dailyEvals} dataKey="mae" name="MAE" color="#b08a30" />
        </div>

        <div className="border-t border-border pt-6">
          <h2 className="font-heading text-lg mb-4">Residual Distribution</h2>
          {residuals.length > 0 ? (
            <ResidualsChart residuals={residuals} />
          ) : (
            <p className="text-muted-foreground text-sm">
              No graded games yet.
            </p>
          )}
        </div>
      </TabsContent>
    </Tabs>
  );
}

const WINDOW_LABELS = [
  { key: "7", label: "Last 7" },
  { key: "30", label: "Last 30" },
  { key: "season", label: "Season" },
] as const;

const PAGE_SIZE = 25;

function EvalHistoryTable({ rows }: { rows: ModelEvaluation[] }) {
  const [windowKey, setWindowKey] =
    useState<(typeof WINDOW_LABELS)[number]["key"]>("30");
  const [page, setPage] = useState(0);

  const filtered = useMemo(() => {
    const sorted = [...rows].sort((a, b) => (a.date < b.date ? 1 : -1));
    if (windowKey === "season") return sorted;
    const days = Number(windowKey);
    return sorted.slice(0, days);
  }, [rows, windowKey]);

  const totalPages =
    windowKey === "season"
      ? Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))
      : 1;
  const visible =
    windowKey === "season"
      ? filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)
      : filtered;

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="inline-flex rounded-sm border border-border p-0.5 text-xs font-mono">
          {WINDOW_LABELS.map((w) => (
            <button
              key={w.key}
              type="button"
              onClick={() => {
                setWindowKey(w.key);
                setPage(0);
              }}
              className={`px-3 py-1 transition-colors ${
                windowKey === w.key
                  ? "bg-foreground text-background"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {w.label}
            </button>
          ))}
        </div>
        {windowKey === "season" && totalPages > 1 ? (
          <div className="flex items-center gap-2 font-mono text-xs">
            <button
              type="button"
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
              className="px-2 py-1 text-muted-foreground hover:text-foreground disabled:opacity-30"
            >
              {"<"}
            </button>
            <span className="text-muted-foreground">
              {page + 1} / {totalPages}
            </span>
            <button
              type="button"
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1}
              className="px-2 py-1 text-muted-foreground hover:text-foreground disabled:opacity-30"
            >
              {">"}
            </button>
          </div>
        ) : null}
      </div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Date</TableHead>
            <TableHead>Pick Acc</TableHead>
            <TableHead>ML</TableHead>
            <TableHead>Run Line</TableHead>
            <TableHead>Totals</TableHead>
            <TableHead className="text-right">MAE</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {visible.map((row) => (
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
              <TableCell>
                {row.totals_correct ?? "—"}/
                {row.totals_predictions ?? "—"}{" "}
                <span className="text-muted-foreground">
                  ({pct(row.totals_accuracy)})
                </span>
              </TableCell>
              <TableCell className="text-right font-mono tabular-nums">
                {fmt(row.mae ?? row.average_total_diff)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

function DailyBettingHistory({ rows }: { rows: ModelEvaluation[] }) {
  const [windowKey, setWindowKey] =
    useState<(typeof WINDOW_LABELS)[number]["key"]>("30");
  const [page, setPage] = useState(0);

  const filtered = useMemo(() => {
    const sorted = [...rows].sort((a, b) => (a.date < b.date ? 1 : -1));
    if (windowKey === "season") return sorted;
    return sorted.slice(0, Number(windowKey));
  }, [rows, windowKey]);

  const totalPages =
    windowKey === "season"
      ? Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))
      : 1;
  const visible =
    windowKey === "season"
      ? filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)
      : filtered;

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="inline-flex rounded-sm border border-border p-0.5 text-xs font-mono">
          {WINDOW_LABELS.map((w) => (
            <button
              key={w.key}
              type="button"
              onClick={() => {
                setWindowKey(w.key);
                setPage(0);
              }}
              className={`px-3 py-1 transition-colors ${
                windowKey === w.key
                  ? "bg-foreground text-background"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {w.label}
            </button>
          ))}
        </div>
        {windowKey === "season" && totalPages > 1 ? (
          <div className="flex items-center gap-2 font-mono text-xs">
            <button
              type="button"
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
              className="px-2 py-1 text-muted-foreground hover:text-foreground disabled:opacity-30"
            >
              {"<"}
            </button>
            <span className="text-muted-foreground">
              {page + 1} / {totalPages}
            </span>
            <button
              type="button"
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1}
              className="px-2 py-1 text-muted-foreground hover:text-foreground disabled:opacity-30"
            >
              {">"}
            </button>
          </div>
        ) : null}
      </div>
      <p className="text-xs text-muted-foreground mb-2 font-mono">
        Stakes = total units risked summed across all bets that day (not per-bet). 1u = your bankroll. ROI = P&amp;L ÷ Stakes.
      </p>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Date</TableHead>
            <TableHead className="text-right">Bets</TableHead>
            <TableHead className="text-right">Stakes</TableHead>
            <TableHead className="text-right">P&amp;L</TableHead>
            <TableHead className="text-right">ROI</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {visible.map((d) => {
            const bets =
              (d.ml_predictions ?? 0) +
              (d.run_line_predictions ?? 0) +
              (d.totals_predictions ?? 0);
            const roiClass =
              d.roi == null
                ? ""
                : d.roi > 0
                ? "text-emerald-500"
                : d.roi < 0
                ? "text-rose-500"
                : "";
            return (
              <TableRow key={d.date}>
                <TableCell className="font-medium">{d.date}</TableCell>
                <TableCell className="text-right font-mono tabular-nums">
                  {bets}
                </TableCell>
                <TableCell className="text-right font-mono tabular-nums">
                  {fmt(d.total_staked_units, 2)}u
                </TableCell>
                <TableCell className="text-right font-mono tabular-nums">
                  {fmtSigned(d.net_profit_units)}u
                </TableCell>
                <TableCell className={`text-right font-mono tabular-nums ${roiClass}`}>
                  {pct(d.roi)}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
