"use client";

interface KpiCardProps {
  label: string;
  value: string;
  sub?: string;
  tooltip?: string;
}

export function KpiCard({ label, value, sub, tooltip }: KpiCardProps) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-xs uppercase tracking-wider text-muted-foreground inline-flex items-center gap-1">
        {label}
        {tooltip && (
          <span className="relative group">
            <span
              aria-label={tooltip}
              className="inline-flex h-3.5 w-3.5 items-center justify-center rounded-full border border-muted-foreground/40 text-[9px] leading-none text-muted-foreground/70 cursor-help select-none"
            >
              ?
            </span>
            <span
              role="tooltip"
              className="pointer-events-none absolute left-1/2 top-full z-50 mt-1.5 -translate-x-1/2 w-64 max-w-[80vw] rounded-md border border-border bg-popover px-3 py-2 text-[11px] font-normal normal-case leading-snug text-popover-foreground tracking-normal opacity-0 shadow-md transition-opacity duration-150 group-hover:opacity-100 group-focus-within:opacity-100"
            >
              {tooltip}
            </span>
          </span>
        )}
      </span>
      <span className="font-mono font-bold tabular-nums text-base">
        {value}
      </span>
      {sub && (
        <span className="font-mono text-xs text-muted-foreground tabular-nums">
          {sub}
        </span>
      )}
    </div>
  );
}
