"use client";

import { useEffect, useRef, useState } from "react";

interface KpiCardProps {
  label: string;
  value: string;
  sub?: string;
  tooltip?: string;
}

export function KpiCard({ label, value, sub, tooltip }: KpiCardProps) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDocPointer = (e: PointerEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", onDocPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onDocPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-xs uppercase tracking-wider text-muted-foreground inline-flex items-center gap-1">
        {label}
        {tooltip && (
          <span ref={wrapRef} className="relative inline-flex">
            <button
              type="button"
              aria-label={tooltip}
              aria-expanded={open}
              onClick={(e) => {
                e.stopPropagation();
                setOpen((v) => !v);
              }}
              className="inline-flex h-3.5 w-3.5 items-center justify-center rounded-full border border-muted-foreground/40 text-[9px] leading-none text-muted-foreground/70 cursor-pointer select-none hover:text-foreground hover:border-foreground/60 focus:outline-none focus-visible:ring-1 focus-visible:ring-foreground/60"
            >
              ?
            </button>
            {open && (
              <span
                role="tooltip"
                className="absolute left-1/2 top-full z-50 mt-1.5 -translate-x-1/2 w-64 max-w-[80vw] rounded-md border border-border bg-popover px-3 py-2 text-[11px] font-normal normal-case leading-snug text-popover-foreground tracking-normal shadow-md"
              >
                {tooltip}
              </span>
            )}
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
