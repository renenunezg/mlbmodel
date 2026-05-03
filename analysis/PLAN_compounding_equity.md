# Plan: Compounded Equity Curve in Production

## Why
The current evaluator (`backend/metrics.py::equity_curve_from_ledger`) is non-compounding:
`equity = 1.0 + cumsum(pnl)`. Stakes are always sized as fractions of a fixed 1u
bankroll. This is fine for *comparing model versions* but misleads users who want
to know "what would my $X bankroll actually have grown to?"

The displayed `kelly_quarter_*` values in `model_outputs_season` are already
fractions. To compound, just multiply them by *current* equity instead of by 1.0.

## Goal
Surface a second equity curve on the Performance → Betting tab labeled
"Compounded equity" alongside the existing flat curve, plus an
`equity_compounded_end` KPI. Keep the flat curve so we can still compare
model versions cleanly.

## Files to change

### 1. `backend/metrics.py`
Add a new function (do **not** modify the existing one - it's used for the
flat-bankroll measurement):

```python
def equity_curve_compounded(ledger: pd.DataFrame, initial: float = 1.0,
                             min_stake: float = 0.0) -> pd.DataFrame:
    """Compounded equity: each bet sized as kelly_fraction * current_equity.

    The ledger already stores `stake` as the kelly fraction (since it was
    computed assuming bankroll=1.0). To compound, treat that stored value
    as a fraction-of-current-equity instead.

    Per bet: dE = stake_frac * equity * (decimal_odds - 1)  if won
                = -stake_frac * equity                       if lost
    Equivalently: equity *= (1 + stake_frac * (decimal_odds - 1))  if won
                  equity *= (1 - stake_frac)                       if lost

    `min_stake` is a dollar floor expressed as a fraction of `initial`
    (e.g. 0.01 means skip any bet smaller than 1% of starting bankroll).
    Default 0.0 = no floor.

    Returns DataFrame[date, equity, daily_pnl, daily_stake].
    """
```

Iterate the ledger sorted by `(date, game_pk)`. Maintain `equity = initial`.
Per row: compute `actual_stake = stake_frac * equity`, skip if below floor,
otherwise apply the multiplicative update. Aggregate to daily granularity
for charting.

### 2. `backend/evaluate_model.py`

Around line 404 where `equity_end` is computed, also compute the compounded
end equity:

```python
eq_flat = equity_curve_from_ledger(window_ledger)
eq_comp = equity_curve_compounded(window_ledger)
equity_end_flat = float(eq_flat["equity"].iloc[-1]) if not eq_flat.empty else 1.0
equity_end_compounded = float(eq_comp["equity"].iloc[-1]) if not eq_comp.empty else 1.0
```

Add to the row dict written to `model_evaluation`:
```python
"equity_end_units": round(equity_end_flat, 4),
"equity_end_compounded": round(equity_end_compounded, 4),
```

### 3. Database migration
Add column via Supabase MCP (`apply_migration`):
```sql
ALTER TABLE model_evaluation
  ADD COLUMN IF NOT EXISTS equity_end_compounded NUMERIC;
```

### 4. `frontend/src/lib/types.ts`
Add `equity_end_compounded: number | null` to `ModelEvaluation` interface.

### 5. `frontend/src/components/equity-curve-chart.tsx`
Currently plots `equity_end_units`. Update to plot **two lines**:
- "Flat (1u bankroll, no compound)" - existing series
- "Compounded (re-Kelly)" - new series

Use distinct colors and a legend. Same y-axis (units). Both start at 1.0.

To get the compounded curve daily, we need either:
- (a) read `equity_end_compounded` from `model_evaluation` (one point per day, simplest)
- (b) recompute on the client from a per-bet ledger endpoint (overkill)

Go with (a): the existing chart already plots one daily point per series.

### 6. `frontend/src/app/performance/tabs.tsx`

In the Betting tab KPI grid, add a new card:
```tsx
<KpiCard
  label="Compounded P&L"
  value={`${fmtSigned((latest?.equity_end_compounded ?? 1) - 1)}u`}
  sub={`$100 → $${(((latest?.equity_end_compounded ?? 1)) * 100).toFixed(0)}`}
  tooltip="Equity curve where each bet is sized as Kelly × CURRENT equity (compounding wins, scaling down after losses). Shows what an actual bankroll would have done."
/>
```

### 7. Verification
- Run `python pipeline.py` locally (or just the eval step). Confirm new column populates.
- Visual check: compounded curve should generally *outpace* flat over winning windows and *fall slower* after losses (because stakes shrink as equity shrinks).
- Sanity invariants: at the very first bet, `equity_end_flat == equity_end_compounded` (within float epsilon). If they diverge on bet #1, there's a bug.

### 8. Not in scope (intentional)
- **Don't** change `kelly_quarter_*` or any other production sizing field.
- **Don't** rip out the flat curve - it's still the right metric for cross-model comparison.
- **Don't** model sportsbook minimum bets unless we see the floor matter empirically (it shouldn't above ~$500 starting bankroll).

## Estimated effort
1–2 hours total: 30 min backend + migration, 30 min frontend, rest verification.
