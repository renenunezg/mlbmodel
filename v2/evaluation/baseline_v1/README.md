# Frozen v1 Baseline

Snapshot of the production XGBoost model, captured for Phase 6 head-to-head
benchmarking against v2 (hierarchical Bayesian + Monte Carlo).

## Pin

| | |
|---|---|
| Captured | 2026-05-02 |
| Git SHA | `a84b4ddd3b4913dcdda0fb69b35dab4421a7442b` |
| Source | `backend/model.py`, `backend/simulation.py` (HEAD at capture time) |

## Files

- `model.py` - XGBoost training + NB win-prob conversion + isotonic calibration. **Frozen copy.**
- `simulation.py` - Negative-binomial (r=6) win-prob math used by `model.py`. **Frozen copy.**

The XGB weights (`models/xgb_model.json`) and isotonic calibrator pickle are
regenerated each fit, so they are **not** snapshotted here. Phase 6 backtester
re-trains v1 fresh on each walk-forward fold using this frozen code.

## Rules

- **Do not modify these files.** They define the v1 baseline that v2 must beat.
- If a bug is found in production `backend/model.py` and gets patched, the
  patched behavior is the *new* v1 - re-snapshot here and update the SHA above.
- Imports from `backend.data.*`, `backend.features`, `backend.strategy`,
  `backend.team_mappings`, `backend.db` are **not** snapshotted (they're
  data/utility code, not modeling logic). The frozen baseline assumes those
  modules are stable enough that v1's behavior is reproducible from the
  current repo state.

## How v2 evaluation uses this

```python
# v2/evaluation/backtester.py (Phase 6)
from v2.evaluation.baseline_v1 import model as v1_model
from v2.evaluation.baseline_v1 import simulation as v1_sim
# ... run walk-forward, train v1 on each fold, compare to v2 outputs
```
