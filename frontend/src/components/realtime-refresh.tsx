"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabase";

// Cheap by design: the pipeline writes to these tables a few times/day, so
// the realtime channel fires that few times. In-game live scores still come
// from /api/live-scores.
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
