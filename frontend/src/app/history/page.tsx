import { supabase } from "@/lib/supabase";
import { ModelOutput, GameInfo } from "@/lib/types";
import { cn, formatDate, formatOdds, formatRuns, formatPct } from "@/lib/utils";
import { V2_CUTOVER_DATE } from "@/lib/constants";
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
  const period = params.period ?? "7";
  const page = Math.max(1, parseInt(params.page ?? "1", 10));
  const offset = (page - 1) * PAGE_SIZE;

  // 7D / 30D quick-filter applies a date floor to both the table and the
  // records widget. Explicit from/to in the URL overrides it for the table.
  const periodFloor =
    period === "7"
      ? new Date(Date.now() - 7 * 86400000).toISOString().split("T")[0]
      : period === "30"
        ? new Date(Date.now() - 30 * 86400000).toISOString().split("T")[0]
        : "";
  const effectiveFrom = from || periodFloor;

  // Read from the unified view so v1's pre-cutover history shows alongside
  // v2's post-cutover picks. start_time is the true chronological order;
  // date alone has no within-day granularity, and game_pk is unrelated to
  // first-pitch time, so sorting by it would scramble the daily schedule.
  let query = supabase
    .from("model_outputs_season_unified")
    .select("*", { count: "exact" })
    .order("start_time", { ascending: false })
    .order("game_pk", { ascending: true })  // groups the two rows of a game adjacent
    .order("team", { ascending: true })     // deterministic home/away order within a game
    .range(offset, offset + PAGE_SIZE - 1);

  if (team) {
    query = query.eq("team", team);
  }
  if (effectiveFrom) {
    query = query.gte("date", effectiveFrom);
  }
  if (to) {
    query = query.lte("date", to);
  }

  // Records widget reads the canonical bet_ledger_agg_v Postgres view. The
  // view does the same filter + grading work that this page used to do in
  // TS, so History and Performance can't drift. Range capped at 10k to
  // defeat the Supabase JS default 1000-row limit.
  let aq = supabase
    .from("bet_ledger_agg_v")
    .select("bet_type, won")
    .range(0, 9999);
  if (periodFloor) aq = aq.gte("date", periodFloor);
  if (team) aq = aq.eq("team", team);

  const [{ data: rows, count, error }, { data: ledgerRows }, { data: latest }, { data: firstV2GameRows }] = await Promise.all([
    query,
    aq,
    supabase
      .from("games")
      .select("updated_at")
      .order("updated_at", { ascending: false })
      .limit(1),
    supabase
      .from("games")
      .select("game_pk")
      .gte("game_date", V2_CUTOVER_DATE)
      .order("start_time", { ascending: true })
      .limit(1),
  ]);

  const firstV2GamePk: number | null = firstV2GameRows?.[0]?.game_pk ?? null;

  const lastUpdated: string | null = latest?.[0]?.updated_at ?? null;

  const predictions: ModelOutput[] = rows ?? [];
  const totalRows = count ?? 0;
  const totalPages = Math.max(1, Math.ceil(totalRows / PAGE_SIZE));

  // Fetch games for current page's pks (for the row-level result columns).
  const gamePks = [...new Set(predictions.map((p) => p.game_pk))];
  const gamesRes = gamePks.length > 0
    ? await supabase.from("games").select("*").in("game_pk", gamePks)
    : { data: [] };

  const gamesMap: Record<number, GameInfo> = {};
  for (const g of (gamesRes.data ?? []) as GameInfo[]) {
    gamesMap[g.game_pk] = g;
  }

  let mlWins = 0, mlLosses = 0;
  let rlWins = 0, rlLosses = 0;
  let totalsWins = 0, totalsLosses = 0;
  for (const r of (ledgerRows ?? []) as { bet_type: string; won: boolean }[]) {
    if (r.bet_type === "ml") { r.won ? mlWins++ : mlLosses++; }
    else if (r.bet_type === "rl") { r.won ? rlWins++ : rlLosses++; }
    else if (r.bet_type === "total") { r.won ? totalsWins++ : totalsLosses++; }
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
                // Badge appears once globally, on the first row of the
                // chronologically-first v2 game across the entire dataset.
                // If that game isn't on the current page, no badge here.
                const firstV2Idx = firstV2GamePk !== null
                  ? predictions.findIndex((r) => r.game_pk === firstV2GamePk)
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
