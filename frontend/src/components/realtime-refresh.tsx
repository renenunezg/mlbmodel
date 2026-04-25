"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabase";

/**
 * Subscribes to Postgres changes on the given tables and triggers
 * router.refresh() (re-runs the server component) on any event.
 *
 * Realtime fires only when the pipeline writes to the DB (a few times/day),
 * so this is cheap. In-game live scores still come from /api/live-scores.
 */
export function RealtimeRefresh({ tables }: { tables: string[] }) {
  const router = useRouter();

  useEffect(() => {
    const channel = supabase.channel("page-refresh");
    for (const table of tables) {
      channel.on(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        "postgres_changes" as any,
        { event: "*", schema: "public", table },
        () => router.refresh()
      );
    }
    channel.subscribe();
    return () => {
      supabase.removeChannel(channel);
    };
  }, [router, tables]);

  return null;
}
