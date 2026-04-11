import { supabase } from "@/lib/supabase";
import { ModelOutput, GameInfo } from "@/lib/types";
import { cn, formatDate, formatOdds, formatRuns, formatPct } from "@/lib/utils";
import Filters from "@/components/filters";
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

  const { data: rows, count, error } = await query;
  const predictions: ModelOutput[] = rows ?? [];
  const totalRows = count ?? 0;
  const totalPages = Math.max(1, Math.ceil(totalRows / PAGE_SIZE));

  // Fetch actual scores for the game_pks in this page
  const gamePks = [...new Set(predictions.map((p) => p.game_pk))];
  let gamesMap: Record<number, GameInfo> = {};

  if (gamePks.length > 0) {
    const { data: games } = await supabase
      .from("games")
      .select("*")
      .in("game_pk", gamePks);

    if (games) {
      for (const g of games as GameInfo[]) {
        gamesMap[g.game_pk] = g;
      }
    }
  }

  // Compute record summary for the period filter
  const periodDate = period === "7"
    ? new Date(Date.now() - 7 * 86400000).toISOString().split("T")[0]
    : period === "30"
      ? new Date(Date.now() - 30 * 86400000).toISOString().split("T")[0]
      : "";

  let recordPredictions: ModelOutput[] = [];
  let recordGamesMap: Record<number, GameInfo> = {};

  {
    let rq = supabase
      .from("model_outputs_season")
      .select("game_pk, team, win_prob, ev_flag, run_line_ev_flag, total_play, our_total, total");
    if (periodDate) rq = rq.gte("date", periodDate);
    if (team) rq = rq.eq("team", team);

    const { data: rRows } = await rq;
    recordPredictions = (rRows ?? []) as ModelOutput[];

    const rPks = [...new Set(recordPredictions.map((p) => p.game_pk))];
    if (rPks.length > 0) {
      const { data: rGames } = await supabase
        .from("games")
        .select("*")
        .eq("status", "Final")
        .in("game_pk", rPks);
      if (rGames) {
        for (const g of rGames as GameInfo[]) {
          recordGamesMap[g.game_pk] = g;
        }
      }
    }
  }

  // Tally records
  let mlWins = 0, mlLosses = 0;
  let rlWins = 0, rlLosses = 0;
  let totalsWins = 0, totalsLosses = 0;

  for (const row of recordPredictions) {
    const game = recordGamesMap[row.game_pk];
    if (!game) continue;

    const isHome = game.home_team === row.team;
    const teamScore = isHome ? game.home_score : game.away_score;
    const oppScore = isHome ? game.away_score : game.home_score;
    if (teamScore == null || oppScore == null) continue;

    const won = teamScore > oppScore;
    const margin = teamScore - oppScore;

    // ML record: only +EV picks
    if (row.ev_flag !== "No Play") {
      if (won) mlWins++; else mlLosses++;
    }

    // RL record: +EV run line picks, W = won by 2+ (covers -1.5)
    if (row.run_line_ev_flag !== "No Play") {
      if (margin >= 2) rlWins++; else rlLosses++;
    }

    // Totals record: Over/Under plays
    if (row.total_play === "Over" || row.total_play === "Under") {
      const actualTotal = (game.home_score ?? 0) + (game.away_score ?? 0);
      const bookTotal = row.total ?? 0;
      if (row.total_play === "Over") {
        if (actualTotal > bookTotal) totalsWins++; else if (actualTotal < bookTotal) totalsLosses++;
      } else {
        if (actualTotal < bookTotal) totalsWins++; else if (actualTotal > bookTotal) totalsLosses++;
      }
    }
  }

  function fmtRecord(w: number, l: number) {
    const total = w + l;
    if (total === 0) return "0-0";
    const pct = ((w / total) * 100).toFixed(0);
    return `${w}-${l} (${pct}%)`;
  }

  // Map game_pk → alternating group index (0, 1, 2…) in display order
  const gameGroupIndex: Record<number, number> = {};
  let groupCounter = 0;
  for (const row of predictions) {
    if (!(row.game_pk in gameGroupIndex)) {
      gameGroupIndex[row.game_pk] = groupCounter++;
    }
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
    <main className="mx-auto max-w-6xl px-4 py-8 space-y-6">
      <h1 className="font-heading text-2xl tracking-tight">Season History</h1>

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
              {predictions.map((row) => {
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
                    gameGroupIndex[row.game_pk] % 2 === 0 ? "bg-muted/30" : "",
                    hasPlay && won === true && "text-positive",
                    hasPlay && won === false && "text-negative"
                  )}
                >
                  <TableCell>{formatDate(row.date)}</TableCell>
                  <TableCell className="font-medium">{row.team}</TableCell>
                  <TableCell>{row.starter ?? "—"}</TableCell>
                  <TableCell className="text-right">{formatRuns(row.expected_runs)}</TableCell>
                  <TableCell className="text-right">{formatPct(row.win_prob)}</TableCell>
                  <TableCell className="text-right">{formatOdds(row.our_odds)}</TableCell>
                  <TableCell className="text-right">{formatOdds(row.moneyline)}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    {isFinal && teamScore != null ? `${teamScore}` : "—"}
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
                      "—"
                    )}
                  </TableCell>
                  <TableCell className="text-center">
                    <span className={cellClass(mlWon, mlIsPlay)}>
                      {mlIsPlay ? row.ev_flag : "—"}
                    </span>
                  </TableCell>
                  <TableCell className="text-center">
                    <span className={cellClass(rlWon, rlIsPlay)}>
                      {rlIsPlay ? row.run_line_ev_flag : "—"}
                    </span>
                  </TableCell>
                  <TableCell className="text-center">
                    <span className={cellClass(totalsWon, totalsIsPlay)}>
                      {totalsIsPlay ? row.total_play : "—"}
                    </span>
                  </TableCell>
                </TableRow>
                );
              })}
            </TableBody>
          </Table>

          {/* Pagination */}
          <div className="flex items-center justify-between pt-2">
            <p className="text-sm text-muted-foreground">
              Showing {offset + 1}–{Math.min(offset + PAGE_SIZE, totalRows)} of{" "}
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
