"""Read-only analysis scripts. Never write to the production DB.

Each top-level script (e.g. `01_calibration.py`) writes outputs under
`analysis/outputs/<script_name>/`. Shared helpers live in `_common.py`,
`_linescores.py`, and `_boxscores.py`.
"""
