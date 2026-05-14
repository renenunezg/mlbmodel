"use client";

import { useState, type ReactNode } from "react";

interface Props {
  date: string;
  title: string;
  accent?: "emerald" | "default";
  children: ReactNode;
}

export function ChangelogEntry({ date, title, accent = "default", children }: Props) {
  const [open, setOpen] = useState(false);

  const borderClass =
    accent === "emerald" ? "border-emerald-500/50" : "border-border";

  return (
    <div className={`border-l-2 ${borderClass} pl-4`}>
      <p className="font-mono text-xs uppercase tracking-wider text-muted-foreground mb-1">
        {date}
      </p>
      <p className="font-medium mb-1.5">{title}</p>
      <div
        className={`text-muted-foreground leading-relaxed transition-[max-height] duration-200 ${
          open ? "" : "line-clamp-5"
        }`}
      >
        {children}
      </div>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="mt-1.5 font-mono text-xs uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors"
      >
        {open ? "− Read less" : "+ Read more"}
      </button>
    </div>
  );
}
