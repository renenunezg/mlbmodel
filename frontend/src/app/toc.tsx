"use client";

import { useEffect, useState } from "react";

const sections = [
  { id: "overview", label: "Overview" },
  { id: "example", label: "Example" },
  { id: "changelog", label: "Changelog" },
  { id: "flow", label: "Architecture" },
  { id: "pipeline", label: "Pipeline" },
  { id: "skill", label: "Skill Layer" },
  { id: "simulator", label: "Simulator" },
  { id: "markets", label: "Markets" },
  { id: "backtest", label: "Backtest" },
  { id: "stack", label: "Tech Stack" },
  { id: "legacy-v1", label: "Legacy v1" },
];

export function TableOfContents() {
  const [activeId, setActiveId] = useState("overview");

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActiveId(entry.target.id);
          }
        }
      },
      { rootMargin: "-10% 0px -75% 0px" }
    );

    sections.forEach(({ id }) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, []);

  return (
    <>
      {/* Desktop sidebar */}
      <aside className="hidden lg:block">
        <nav className="sticky top-6 flex flex-col gap-0.5">
          <p className="mb-2 font-mono text-xs uppercase tracking-wider text-muted-foreground">
            Contents
          </p>
          {sections.map(({ id, label }) => (
            <a
              key={id}
              href={`#${id}`}
              className={`border-l-2 py-1 pl-3 font-mono text-xs uppercase tracking-wider transition-colors ${
                activeId === id
                  ? "border-foreground text-foreground"
                  : "border-transparent text-muted-foreground hover:text-foreground"
              }`}
            >
              {label}
            </a>
          ))}
        </nav>
      </aside>

      {/* Mobile horizontal strip */}
      <div className="lg:hidden mb-6 overflow-x-auto border-b border-border pb-3">
        <nav className="flex gap-0 min-w-max">
          {sections.map(({ id, label }) => (
            <a
              key={id}
              href={`#${id}`}
              className={`border-b-2 px-3 py-1.5 font-mono text-xs uppercase tracking-wider whitespace-nowrap transition-colors ${
                activeId === id
                  ? "border-foreground text-foreground"
                  : "border-transparent text-muted-foreground hover:text-foreground"
              }`}
            >
              {label}
            </a>
          ))}
        </nav>
      </div>
    </>
  );
}
