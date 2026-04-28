"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ThemeToggle } from "@/components/theme-toggle";

const links = [
  { href: "/", label: "Methodology" },
  { href: "/games", label: "Games" },
  { href: "/history", label: "History" },
  { href: "/performance", label: "Performance" },
  { href: "/about", label: "About" },
];

export function Nav() {
  const pathname = usePathname();

  return (
    <nav className="relative z-10 border-b border-border bg-background">
      <div className="mx-auto flex w-full max-w-7xl min-w-0 flex-col items-start gap-1 px-4 pt-3 sm:flex-row sm:items-center sm:justify-between sm:gap-0 sm:pb-0">
        <div className="flex items-baseline gap-3 pb-2 sm:pb-3">
          <Link href="/" className="font-heading text-lg tracking-tight">
            MLB Predictions
          </Link>
          <span className="font-mono text-xs text-muted-foreground tabular-nums">
            {new Date().toLocaleDateString("en-US", {
              month: "short",
              day: "numeric",
              year: "numeric",
            })}
          </span>
        </div>
        <div className="flex w-full items-center gap-0 overflow-x-auto pb-0 sm:w-auto sm:overflow-visible">
          {links.map((link) => {
            const isActive =
              link.href === "/"
                ? pathname === "/"
                : pathname.startsWith(link.href);

            return (
              <Link
                key={link.href}
                href={link.href}
                className={`px-3 py-3 font-mono text-xs uppercase tracking-wider transition-colors ${
                  isActive
                    ? "border-b-2 border-foreground text-foreground"
                    : "border-b-2 border-transparent text-muted-foreground hover:text-foreground"
                }`}
              >
                {link.label}
              </Link>
            );
          })}
          <div className="ml-2 flex items-center">
            <ThemeToggle />
          </div>
        </div>
      </div>
    </nav>
  );
}
