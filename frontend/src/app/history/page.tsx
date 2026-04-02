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
  searchParams: Promise<{ team?: string; from?: string; to?: string; page?: string }>;
}) {
  const params = await searchParams;

  const team = params.team ?? "";
  const from = params.from ?? "";
  const to = params.to ?? "";
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

  // Build current search params string for pagination links
  function pageUrl(p: number) {
    const sp = new URLSearchParams();
    if (team) sp.set("team", team);
    if (from) sp.set("from", from);
    if (to) sp.set("to", to);
    sp.set("page", String(p));
    return `/history?${sp.toString()}`;
  }

  return (
    <main className="mx-auto max-w-6xl px-4 py-8 space-y-6">
      <h1 className="font-heading text-2xl tracking-tight">Season History</h1>

      <Filters />

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
              {predictions.map((row) => (
                <TableRow key={`${row.game_pk}-${row.team}`}>
                  <TableCell>{formatDate(row.date)}</TableCell>
                  <TableCell className="font-medium">{row.team}</TableCell>
                  <TableCell>{row.starter ?? "—"}</TableCell>
                  <TableCell className="text-right">{formatRuns(row.expected_runs)}</TableCell>
                  <TableCell className="text-right">{formatPct(row.win_prob)}</TableCell>
                  <TableCell className="text-right">{formatOdds(row.our_odds)}</TableCell>
                  <TableCell className="text-right">{formatOdds(row.moneyline)}</TableCell>
                  {(() => {
                    const game = gamesMap[row.game_pk];
                    const isFinal = game?.status === "Final";
                    const isHome = game?.home_team === row.team;
                    const teamScore = isHome ? game?.home_score : game?.away_score;
                    const oppScore = isHome ? game?.away_score : game?.home_score;
                    const won = teamScore != null && oppScore != null ? teamScore > oppScore : null;
                    const predictedWin = row.win_prob > 0.5;
                    const correctPick = won != null ? won === predictedWin : null;
                    return (
                      <>
                        <TableCell className="text-right tabular-nums">
                          {isFinal && teamScore != null && oppScore != null
                            ? `${teamScore}-${oppScore}`
                            : "—"}
                        </TableCell>
                        <TableCell className="text-center">
                          {won != null ? (
                            <span
                              className={cn(
                                "font-semibold",
                                correctPick
                                  ? "text-positive"
                                  : "text-negative"
                              )}
                            >
                              {won ? "W" : "L"}
                            </span>
                          ) : (
                            "—"
                          )}
                        </TableCell>
                      </>
                    );
                  })()}
                  <TableCell className="text-center">
                    <span
                      className={
                        row.ev_flag !== "No Play"
                          ? "text-positive font-semibold"
                          : "text-muted-foreground"
                      }
                    >
                      {row.ev_flag !== "No Play" ? row.ev_flag : "—"}
                    </span>
                  </TableCell>
                  <TableCell className="text-center">
                    <span
                      className={
                        row.run_line_ev_flag !== "No Play"
                          ? "text-accent-blue font-semibold"
                          : "text-muted-foreground"
                      }
                    >
                      {row.run_line_ev_flag !== "No Play" ? row.run_line_ev_flag : "—"}
                    </span>
                  </TableCell>
                  <TableCell className="text-center">{row.total_play || "—"}</TableCell>
                </TableRow>
              ))}
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
