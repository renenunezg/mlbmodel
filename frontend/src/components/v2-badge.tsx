export function V2Badge({ className = "" }: { className?: string }) {
  return (
    <span
      className={`ml-2 inline-block rounded-sm border border-emerald-500/40 bg-emerald-500/10 px-1.5 py-0 text-[10px] font-mono uppercase tracking-wider text-emerald-600 dark:text-emerald-400 ${className}`}
      title="v2 model went live 2026-05-12. Earlier predictions are v1 XGBoost; later predictions are v2 Bayesian + Monte Carlo."
    >
      v2 starts
    </span>
  );
}
