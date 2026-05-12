"""Pretty-print Phase 6 head-to-head results."""
from __future__ import annotations


def _pct(x, digits=2):
    if x is None or (isinstance(x, float) and (x != x)):
        return "  --  "
    return f"{x*100:+.{digits}f}%"


def _abs(x, digits=4):
    if x is None or (isinstance(x, float) and (x != x)):
        return "  --  "
    return f"{x:.{digits}f}"


def _gate_mark(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def format_head_to_head(v1: dict, v2: dict, gates: dict, window: str) -> str:
    rel_brier = (v2["brier"] - v1["brier"]) / v1["brier"] if v1.get("brier") else float("nan")
    rel_ll = (v2["log_loss"] - v1["log_loss"]) / v1["log_loss"] if v1.get("log_loss") else float("nan")

    lines = []
    lines.append(f"=== Phase 6 head-to-head ({window}) ===")
    lines.append(f"games compared: {min(v1['n_games'], v2['n_games'])}")
    lines.append(f"team-rows: v1={v1['n_team_rows']}, v2={v2['n_team_rows']}")
    lines.append("")
    lines.append(f"{'metric':24}{'v1':>12}{'v2':>12}{'delta':>14}  gate")
    lines.append("-" * 74)
    lines.append(
        f"{'Brier (ML)':24}{_abs(v1['brier'], 4):>12}{_abs(v2['brier'], 4):>12}"
        f"{_pct(rel_brier):>14}  {_gate_mark(gates['brier_within_1pct'])}"
    )
    lines.append(
        f"{'Log-loss (ML)':24}{_abs(v1['log_loss'], 4):>12}{_abs(v2['log_loss'], 4):>12}"
        f"{_pct(rel_ll):>14}  {_gate_mark(gates['logloss_within_1pct'])}"
    )
    lines.append(
        f"{'Max calibration gap':24}{_pct(v1['max_calibration_gap']):>12}"
        f"{_pct(v2['max_calibration_gap']):>12}{'':>14}  {_gate_mark(gates['calibration_max_5pp'])}"
    )

    for key, label in [("roi_ml", "ROI (ML, flag)"),
                       ("roi_rl", "ROI (RL, flag)"),
                       ("roi_total", "ROI (Totals)")]:
        r1 = v1[key]["roi"]
        r2 = v2[key]["roi"]
        delta = (r2 - r1) if (r1 == r1 and r2 == r2) else float("nan")  # NaN-safe
        lines.append(
            f"{label:24}{_pct(r1):>12}{_pct(r2):>12}{_pct(delta):>14}  "
        )

    lines.append("")
    lines.append(f"{'bets placed ML':24}{v1['roi_ml']['n_bets']:>12}{v2['roi_ml']['n_bets']:>12}")
    lines.append(f"{'bets placed RL':24}{v1['roi_rl']['n_bets']:>12}{v2['roi_rl']['n_bets']:>12}")
    lines.append(f"{'bets placed TOT':24}{v1['roi_total']['n_bets']:>12}{v2['roi_total']['n_bets']:>12}")
    lines.append("")
    verdict = "GREEN-LIGHT for cutover" if gates["all_pass"] else "GATES FAILED: diagnose before cutover"
    lines.append(f"verdict: {verdict}")
    for k, v in gates.items():
        if k == "all_pass":
            continue
        lines.append(f"  - {k}: {_gate_mark(v)}")
    return "\n".join(lines)
