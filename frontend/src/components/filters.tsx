"use client";

import { useRouter, usePathname, useSearchParams } from "next/navigation";
import { useCallback } from "react";

const MLB_TEAMS = [
  "ARI", "ATL", "ATH", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL",
  "DET", "HOU", "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY",
  "PHI", "PIT", "SDP", "SEA", "SFG", "STL", "TBR", "TEX", "TOR", "WSN",
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
    <div className="flex flex-wrap items-center gap-3 font-mono text-xs">
      <select
        value={team}
        onChange={(e) => updateParam("team", e.target.value)}
        className="h-8 border border-input bg-transparent px-2"
      >
        <option value="">All Teams</option>
        {MLB_TEAMS.map((t) => (
          <option key={t} value={t}>
            {t}
          </option>
        ))}
      </select>

      <label className="flex items-center gap-1.5 text-muted-foreground">
        From
        <input
          type="date"
          value={from}
          onChange={(e) => updateParam("from", e.target.value)}
          className="h-8 border border-input bg-transparent px-2"
        />
      </label>

      <label className="flex items-center gap-1.5 text-muted-foreground">
        To
        <input
          type="date"
          value={to}
          onChange={(e) => updateParam("to", e.target.value)}
          className="h-8 border border-input bg-transparent px-2"
        />
      </label>

      {(team || from || to) && (
        <button
          onClick={clearAll}
          className="text-muted-foreground underline hover:text-foreground transition-colors"
        >
          Clear
        </button>
      )}
    </div>
  );
}
