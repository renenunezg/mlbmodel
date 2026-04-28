"use client";

import { useTheme } from "next-themes";
import { Sun, Moon, Monitor } from "lucide-react";
import { useEffect, useState } from "react";

const order = ["system", "light", "dark"] as const;

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const current = mounted ? theme ?? "system" : "system";
  const Icon =
    current === "dark" ? Moon : current === "light" ? Sun : Monitor;

  const cycle = () => {
    const i = order.indexOf(current as (typeof order)[number]);
    setTheme(order[(i + 1) % order.length]);
  };

  return (
    <button
      type="button"
      onClick={cycle}
      aria-label={`Theme: ${current}. Click to cycle.`}
      title={`Theme: ${current}`}
      className="inline-flex h-7 w-7 items-center justify-center rounded-sm border border-transparent text-muted-foreground transition-colors hover:border-border hover:text-foreground"
    >
      <Icon className="h-4 w-4" suppressHydrationWarning />
    </button>
  );
}
