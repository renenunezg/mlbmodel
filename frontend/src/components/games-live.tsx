"use client";

import { useEffect, useRef, useState } from "react";
import type { GameMatchup } from "@/lib/types";
import type { LiveScore } from "@/app/api/live-scores/route";
import { GameCard } from "@/components/game-card";

const POLL_MS = 60_000;

function mergeScores(
  matchups: GameMatchup[],
  scores: LiveScore[]
): GameMatchup[] {
  if (scores.length === 0) return matchups;
  const byPk = new Map(scores.map((s) => [s.game_pk, s]));
  return matchups.map((m) => {
    const s = byPk.get(m.game_pk);
    if (!s) return m;
    return {
      ...m,
      home_score: s.home_score ?? m.home_score,
      away_score: s.away_score ?? m.away_score,
      status: s.status ?? m.status,
      current_inning: s.current_inning,
      inning_state: s.inning_state,
    };
  });
}

function allFinal(matchups: GameMatchup[]): boolean {
  if (matchups.length === 0) return true;
  return matchups.every((m) => m.status === "Final");
}

export function GamesLive({ initial }: { initial: GameMatchup[] }) {
  const [matchups, setMatchups] = useState<GameMatchup[]>(initial);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function fetchOnce() {
      try {
        const res = await fetch("/api/live-scores", { cache: "no-store" });
        if (!res.ok) return;
        const data = (await res.json()) as { scores?: LiveScore[] };
        if (cancelled || !data.scores) return;
        setMatchups((prev) => mergeScores(prev, data.scores!));
      } catch {
        // network blip — just wait for next tick
      }
    }

    function schedule() {
      if (timerRef.current) clearTimeout(timerRef.current);
      if (document.visibilityState !== "visible") return;
      if (allFinal(matchups)) return;
      timerRef.current = setTimeout(async () => {
        await fetchOnce();
        schedule();
      }, POLL_MS);
    }

    // Kick off immediately on mount so users see live data without waiting 60s
    fetchOnce().then(() => {
      if (!cancelled) schedule();
    });

    function onVisibility() {
      if (document.visibilityState === "visible") {
        fetchOnce().then(() => {
          if (!cancelled) schedule();
        });
      } else if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    }
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      cancelled = true;
      if (timerRef.current) clearTimeout(timerRef.current);
      document.removeEventListener("visibilitychange", onVisibility);
    };
    // Re-evaluate when all games become Final so we can stop polling
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [allFinal(matchups)]);

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
      {matchups.map((matchup) => (
        <GameCard key={matchup.game_pk} matchup={matchup} />
      ))}
    </div>
  );
}
