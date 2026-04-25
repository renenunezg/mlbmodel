"use client";

import { useEffect, useState } from "react";

function formatRelative(ts: string | null): string {
  if (!ts) return "unknown";
  const then = new Date(ts).getTime();
  const now = Date.now();
  const diffSec = Math.max(0, Math.floor((now - then) / 1000));
  if (diffSec < 60) return `${diffSec}s ago`;
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin} min ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr} hr ago`;
  const diffDay = Math.floor(diffHr / 24);
  return `${diffDay}d ago`;
}

export function LastUpdated({
  timestamp,
  schedule,
}: {
  timestamp: string | null;
  schedule: string;
}) {
  const [, setTick] = useState(0);

  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 30_000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="text-xs text-muted-foreground">
      <div>{schedule}</div>
      <div>Last updated: {formatRelative(timestamp)}</div>
    </div>
  );
}
