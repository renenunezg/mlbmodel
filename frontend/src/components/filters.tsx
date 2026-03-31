"use client";

import { useRouter, usePathname, useSearchParams } from "next/navigation";
import { useCallback } from "react";

const MLB_TEAMS = [
  "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL", "DET",
  "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
  "PHI", "PIT", "SD", "SF", "SEA", "STL", "TB", "TEX", "TOR", "WSH",
];

export default function Filters() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const team = searchParams.get("team") ?? "";
  const from = searchParams.get("from") ?? "";
  const to = searchParams.get("to") ?? "";

  const updateParam = useCallback(
    (key: string, value: string) => {
      const params = new URLSearchParams(searchParams.toString());
      if (value) {
        params.set(key, value);
      } else {
        params.delete(key);
      }
      // Reset to page 1 when filters change
      params.delete("page");
      router.push(`${pathname}?${params.toString()}`);
    },
    [router, pathname, searchParams]
  );

  const clearAll = useCallback(() => {
    router.push(pathname);
  }, [router, pathname]);

  return (
    <div className="flex flex-wrap items-center gap-3">
      <select
        value={team}
        onChange={(e) => updateParam("team", e.target.value)}
        className="h-9 rounded-md border border-input bg-background px-3 text-sm"
      >
        <option value="">All Teams</option>
        {MLB_TEAMS.map((t) => (
          <option key={t} value={t}>
            {t}
          </option>
        ))}
      </select>

      <label className="flex items-center gap-1.5 text-sm text-muted-foreground">
        From
        <input
          type="date"
          value={from}
          onChange={(e) => updateParam("from", e.target.value)}
          className="h-9 rounded-md border border-input bg-background px-2 text-sm"
        />
      </label>

      <label className="flex items-center gap-1.5 text-sm text-muted-foreground">
        To
        <input
          type="date"
          value={to}
          onChange={(e) => updateParam("to", e.target.value)}
          className="h-9 rounded-md border border-input bg-background px-2 text-sm"
        />
      </label>

      {(team || from || to) && (
        <button
          onClick={clearAll}
          className="h-9 rounded-md border border-input bg-background px-3 text-sm text-muted-foreground hover:bg-muted transition-colors"
        >
          Clear
        </button>
      )}
    </div>
  );
}
