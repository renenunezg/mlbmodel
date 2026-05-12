import { supabase } from "@/lib/supabase";
import { ModelOutput, GameInfo } from "@/lib/types";
import { cn, formatDate, formatOdds, formatRuns, formatPct } from "@/lib/utils";
import { getFirstV2Date } from "@/lib/constants";
import { V2Badge } from "@/components/v2-badge";
import Filters from "@/components/filters";
import { LastUpdated } from "@/components/last-updated";
import { RealtimeRefresh } from "@/components/realtime-refresh";
import {
  Table,
  TableHeader,
  TableBody,
  TableHead,
  TableRow,
  TableCell,
} from "@/components/ui/table";
import Link from "next/link";

export const revalidate = 300;

const PAGE_SIZE = 50;

export default async function HistoryPage({
  searchParams,
}: {
  searchParams: Promise<{ team?: string; from?: string; to?: string; page?: string; period?: string }>;
}) {
  const params = await searchParams;

  const team = params.team ?? "";
  const from = params.from ?? "";
  const to = params.to ?? "";
  const period = params.period ?? "";
  const page = Math.max(1, parseInt(params.page ?? "1", 10));
  const offset = (page - 1) * PAGE_SIZE;

  // Build query for model_outputs_season
  let query = supabase
    .from("model_outputs_season")
    .select("*", { count: "exact" })
    .order("date", { ascending: false })
    .order("game_pk", { ascending: true })
    .range(offset, offset + PAGE_SIZE - 1);

  if (team) {
    query = query.eq("team", team);
  }
  if (from) {
    query = query.gte("date", from);
  }
  if (to) {
    query = query.lte("date", to);
  }

  // Compute record period date up front
  const periodDate = period === "7"
    ? new Date(Date.now() - 7 * 86400000).toISOString().split("T")[0]
    : period === "30"
      ? new Date(Date.now() - 30 * 86400000).toISOString().split("T")[0]
      : "";

  // Build record query (runs in parallel with main pagination query).
  // Pull odds + market probabilities so we can revalidate the +EV threshold
  // here - ev_flag values stored in model_outputs_season can be sub-threshold
  // due to win_prob rounding or stale odds snapshots.
  let rq = supabase
    .from("model_outputs_season")
    .select(
      "game_pk, team, win_prob, ev_flag, run_line_ev_flag, total_play, our_total, total, " +
      "moneyline, spread, spread_odds, p_cover, p_over, p_under, total_over_odds, total_under_odds, " +
      "kelly_quarter_ml, kelly_quarter_rl, kelly_quarter_total"
    );
  if (periodDate) rq = rq.gte("date", periodDate);
  if (team) rq = rq.eq("team", team);

  // Run both main queries in parallel
  const [{ data: rows, count, error }, { data: rRows }, { data: latest }] = await Promise.all([
    query,
    rq,
    supabase
      .from("games")
      .select("updated_at")
      .order("updated_at", { ascending: false })
      .limit(1),
  ]);

  const lastUpdated: string | null = latest?.[0]?.updated_at ?? null;

  const predictions: ModelOutput[] = rows ?? [];
  const totalRows = count ?? 0;
  const totalPages = Math.max(1, Math.ceil(totalRows / PAGE_SIZE));
  const recordPredictions: ModelOutput[] = (rRows ?? []) as unknown as ModelOutput[];

  // Fetch games for both sets of pks in parallel
  const gamePks = [...new Set(predictions.map((p) => p.game_pk))];
  const rPks = [...new Set(recordPredictions.map((p) => p.game_pk))];

  const [gamesRes, rGamesRes] = await Promise.all([
    gamePks.length > 0
      ? supabase.from("games").select("*").in("game_pk", gamePks)
      : Promise.resolve({ data: [] }),
    rPks.length > 0
      ? supabase.from("games").select("*").eq("status", "Final").in("game_pk", rPks)
      : Promise.resolve({ data: [] }),
  ]);

  const gamesMap: Record<number, GameInfo> = {};
  for (const g of (gamesRes.data ?? []) as GameInfo[]) {
    gamesMap[g.game_pk] = g;
  }

  const recordGamesMap: Record<number, GameInfo> = {};
  for (const g of (rGamesRes.data ?? []) as GameInfo[]) {
    recordGamesMap[g.game_pk] = g;
  }

  // Tally records. Mirrors backend/strategy.py thresholds and
  // backend/evaluate_model.py's _calc_run_line_pick / _calc_total_pick logic
  // so the History records reconcile with the Performance eval.
  const ML_THRESHOLD = 0.045;
  const RL_THRESHOLD = 0.045;
  const TOT_THRESHOLD = 0.065;

  function americanToImplied(odds: number | null): number | null {
    if (odds == null) return null;
    return odds > 0 ? 100 / (odds + 100) : -odds / (-odds + 100);
  }

  let mlWins = 0, mlLosses = 0;
  let rlWins = 0, rlLosses = 0;
  let totalsWins = 0, totalsLosses = 0;
  const seenTotalsPks = new Set<number>();

  for (const row of recordPredictions) {
    const game = recordGamesMap[row.game_pk];
    if (!game) continue;

    const isHome = game.home_team === row.team;
    const teamScore = isHome ? game.home_score : game.away_score;
    const oppScore = isHome ? game.away_score : game.home_score;
    if (teamScore == null || oppScore == null) continue;

    const won = teamScore > oppScore;
    const margin = teamScore - oppScore;

    if (row.ev_flag === row.team && (row.kelly_quarter_ml ?? 0) > 0) {
      const book = americanToImplied(row.moneyline);
      if (book != null && row.win_prob - book >= ML_THRESHOLD) {
        if (won) mlWins++; else mlLosses++;
      }
    }

    if (row.run_line_ev_flag === row.team && row.spread != null && (row.kelly_quarter_rl ?? 0) > 0) {
      const book = americanToImplied(row.spread_odds);
      const pCover = (row as ModelOutput & { p_cover: number | null }).p_cover;
      if (book != null && pCover != null && pCover - book >= RL_THRESHOLD) {
        const covered = row.spread < 0
          ? margin >= -row.spread
          : (won || margin >= -row.spread);
        if (covered) rlWins++; else rlLosses++;
      }
    }

    if (
      (row.total_play === "Over" || row.total_play === "Under")
      && !seenTotalsPks.has(row.game_pk)
      && (row.kelly_quarter_total ?? 0) > 0
    ) {
      const r = row as ModelOutput & {
        p_over: number | null; p_under: number | null;
        total_over_odds: number | null; total_under_odds: number | null;
      };
      const isOver = row.total_play === "Over";
      const modelP = isOver ? r.p_over : r.p_under;
      const book = americanToImplied(isOver ? r.total_over_odds : r.total_under_odds);
      if (modelP != null && book != null && modelP - book >= TOT_THRESHOLD) {
        seenTotalsPks.add(row.game_pk);
        const actualTotal = (game.home_score ?? 0) + (game.away_score ?? 0);
        const bookTotal = row.total ?? 0;
        if (isOver) {
          if (actualTotal > bookTotal) totalsWins++; else if (actualTotal < bookTotal) totalsLosses++;
        } else {
          if (actualTotal < bookTotal) totalsWins++; else if (actualTotal > bookTotal) totalsLosses++;
        }
      }
    }
  }

  function fmtRecord(w: number, l: number) {
    const total = w + l;
    if (total === 0) return "0-0";
    const pct = ((w / total) * 100).toFixed(0);
    return `${w}-${l} (${pct}%)`;
  }

  // Build current search params string for pagination links
  function pageUrl(p: number) {
    const sp = new URLSearchParams();
    if (team) sp.set("team", team);
    if (from) sp.set("from", from);
    if (to) sp.set("to", to);
    if (period) sp.set("period", period);
    sp.set("page", String(p));
    return `/history?${sp.toString()}`;
  }

  return (
    <main className="mx-auto w-full max-w-6xl min-w-0 px-4 py-8 space-y-6">
      <RealtimeRefresh tables={["games", "model_outputs_season"]} />
      <div className="flex items-start justify-between gap-4">
        <h1 className="font-heading text-2xl tracking-tight">Season History</h1>
        <LastUpdated
          timestamp={lastUpdated}
          schedule="Predictions ~5 AM PT • Results scored overnight"
        />
      </div>

      <Filters />

      {/* Record Summary */}
      <div className="flex flex-wrap items-center gap-4 font-mono text-sm">
        <div className="flex items-center gap-2">
          <span className="text-muted-foreground">ML:</span>
          <span className={cn("font-semibold", mlWins + mlLosses > 0 && mlWins > mlLosses ? "text-positive" : mlWins < mlLosses ? "text-negative" : "")}>
            {fmtRecord(mlWins, mlLosses)}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-muted-foreground">RL:</span>
          <span className={cn("font-semibold", rlWins + rlLosses > 0 && rlWins > rlLosses ? "text-positive" : rlWins < rlLosses ? "text-negative" : "")}>
            {fmtRecord(rlWins, rlLosses)}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-muted-foreground">Totals:</span>
          <span className={cn("font-semibold", totalsWins + totalsLosses > 0 && totalsWins > totalsLosses ? "text-positive" : totalsWins < totalsLosses ? "text-negative" : "")}>
            {fmtRecord(totalsWins, totalsLosses)}
          </span>
        </div>
      </div>

      {error ? (
        <p className="text-sm text-destructive">
          Error loading data: {error.message}
        </p>
      ) : predictions.length === 0 ? (
        <p className="text-sm text-muted-foreground py-12 text-center">
          No predictions found for the selected filters.
        </p>
      ) : (
        <>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Date</TableHead>
                <TableHead>Team</TableHead>
                <TableHead>Starter</TableHead>
                <TableHead className="text-right">xR</TableHead>
                <TableHead className="text-right">Win Prob</TableHead>
                <TableHead className="text-right">Model Odds</TableHead>
                <TableHead className="text-right">Book ML</TableHead>
                <TableHead className="text-right">Score</TableHead>
                <TableHead className="text-center">Result</TableHead>
                <TableHead className="text-center">+EV</TableHead>
                <TableHead className="text-center">RL +EV</TableHead>
                <TableHead className="text-center">Total Play</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {(() => {
                const firstV2Date = getFirstV2Date(predictions);
                const firstV2Idx = firstV2Date
                  ? predictions.findIndex(
                      (r) => r.date && r.date.slice(0, 10) === firstV2Date,
                    )
                  : -1;
                return predictions.map((row, i) => {
                const nextRow = predictions[i + 1];
                const showV2Badge = i === firstV2Idx;
                const game = gamesMap[row.game_pk];
                const isFinal = game?.status === "Final";
                const isHome = game?.home_team === row.team;
                const teamScore = isHome ? game?.home_score : game?.away_score;
                const oppScore = isHome ? game?.away_score : game?.home_score;
                const won = isFinal && teamScore != null && oppScore != null
                  ? teamScore > oppScore
                  : null;
                const hasPlay =
                  row.ev_flag !== "No Play" || row.run_line_ev_flag !== "No Play";

                const mlIsPlay = row.ev_flag !== "No Play";
                const rlIsPlay = row.run_line_ev_flag !== "No Play";
                const totalsIsPlay =
                  row.total_play === "Over" || row.total_play === "Under";

                const mlWon: boolean | null =
                  mlIsPlay && won !== null ? won : null;

                const rlWon: boolean | null =
                  rlIsPlay &&
                  isFinal &&
                  teamScore != null &&
                  oppScore != null
                    ? teamScore - oppScore >= 2
                    : null;

                const totalsWon: boolean | null = (() => {
                  if (!totalsIsPlay) return null;
                  if (
                    !isFinal ||
                    game?.home_score == null ||
                    game?.away_score == null
                  )
                    return null;
                  if (row.total == null) return null;
                  const actual = game.home_score + game.away_score;
                  const book = row.total;
                  if (actual === book) return null; // push
                  return row.total_play === "Over"
                    ? actual > book
                    : actual < book;
                })();

                const cellClass = (outcome: boolean | null, isPlay: boolean) => {
                  if (!isPlay) return "text-muted-foreground";
                  if (outcome === true) return "text-positive font-semibold";
                  if (outcome === false) return "text-negative font-semibold";
                  return "font-semibold";
                };

                return (
                <TableRow
                  key={`${row.game_pk}-${row.team}`}
                  className={cn(
                    // Suppress the divider between the two rows of the same game
                    nextRow?.game_pk === row.game_pk && "border-b-0",
                    hasPlay && won === true && "text-positive",
                    hasPlay && won === false && "text-negative"
                  )}
                >
                  <TableCell>
                    {formatDate(row.date)}
                    {showV2Badge ? <V2Badge /> : null}
                  </TableCell>
                  <TableCell className="font-medium">{row.team}</TableCell>
                  <TableCell>{row.starter ?? "-"}</TableCell>
                  <TableCell className="text-right">{formatRuns(row.expected_runs)}</TableCell>
                  <TableCell className="text-right">{formatPct(row.win_prob)}</TableCell>
                  <TableCell className="text-right">{formatOdds(row.our_odds)}</TableCell>
                  <TableCell className="text-right">{formatOdds(row.moneyline)}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    {isFinal && teamScore != null ? `${teamScore}` : "-"}
                  </TableCell>
                  <TableCell className="text-center">
                    {won != null ? (
                      <span
                        className={cn(
                          "font-semibold",
                          won ? "text-positive" : "text-negative"
                        )}
                      >
                        {won ? "W" : "L"}
                      </span>
                    ) : (
                      "-"
                    )}
                  </TableCell>
                  <TableCell className="text-center">
                    <span className={cellClass(mlWon, mlIsPlay)}>
                      {mlIsPlay ? row.ev_flag : "-"}
                    </span>
                  </TableCell>
                  <TableCell className="text-center">
                    <span className={cellClass(rlWon, rlIsPlay)}>
                      {rlIsPlay ? row.run_line_ev_flag : "-"}
                    </span>
                  </TableCell>
                  <TableCell className="text-center">
                    <span className={cellClass(totalsWon, totalsIsPlay)}>
                      {totalsIsPlay ? row.total_play : "-"}
                    </span>
                  </TableCell>
                </TableRow>
                );
              });
              })()}
            </TableBody>
          </Table>

          {/* Pagination */}
          <div className="flex items-center justify-between pt-2">
            <p className="text-sm text-muted-foreground">
              Showing {offset + 1}-{Math.min(offset + PAGE_SIZE, totalRows)} of{" "}
              {totalRows} rows
            </p>
            <div className="flex items-center gap-2">
              {page > 1 ? (
                <Link
                  href={pageUrl(page - 1)}
                  className="inline-flex h-9 items-center justify-center border border-input bg-background px-3 font-mono text-xs hover:bg-muted transition-colors"
                >
                  Previous
                </Link>
              ) : (
                <span className="inline-flex h-9 items-center justify-center border border-input bg-background px-3 font-mono text-xs text-muted-foreground opacity-50">
                  Previous
                </span>
              )}
              <span className="text-sm text-muted-foreground">
                Page {page} of {totalPages}
              </span>
              {page < totalPages ? (
                <Link
                  href={pageUrl(page + 1)}
                  className="inline-flex h-9 items-center justify-center border border-input bg-background px-3 font-mono text-xs hover:bg-muted transition-colors"
                >
                  Next
                </Link>
              ) : (
                <span className="inline-flex h-9 items-center justify-center border border-input bg-background px-3 font-mono text-xs text-muted-foreground opacity-50">
                  Next
                </span>
              )}
            </div>
          </div>
        </>
      )}
    </main>
  );
}
