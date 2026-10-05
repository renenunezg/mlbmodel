"""Shared production market switches and model-policy constants."""
import os
from datetime import date

# Totals bar is higher; total-runs markets are noisier than sides.
EV_THRESHOLDS = {
    "ml": 0.045,
    "rl": 0.045,
    "totals": 0.065,
}

# Moneyline recommendation switch.
MONEYLINE_ENABLED = True

# Run-line recommendation switch.
RUNLINE_ENABLED = True

# Totals are off: every totals edge bucket lost money on the 2026 backtest and
# model edge had no relationship to outcome.
TOTALS_ENABLED = False

# Set to 0 for weather-off control runs.
WEATHER_ENABLED = os.getenv("MLBMODEL_WEATHER_ENABLED", "1") == "1"

# Weight of the sim in the logit-scale blend with the de-vigged market
# consensus. The raw sim prob flags big underdogs +EV systematically. 0.2 was
# the log-loss optimum on 2026 games (a 2026-09-09 sweep over 1272 v2 games
# put it at 0 and worsening monotonically); at 0.2 the 4.5% ML threshold
# needs a 20+ point sim/market disagreement and no ML pick surfaced for three
# weeks. Raised to 0.5 on 2026-09-09 by decision, accepting the calibration
# cost, so the moneyline market is not silently off. 0.5 is the highest
# weight at which the measured big-underdog failure (sim 40% vs market 28%)
# still stays No Play; see test_market_anchor_stops_flagging_big_dogs.
MARKET_ANCHOR_W_MODEL = 0.5

# Home-field shift on the sim's home logit, applied before market anchoring.
# The sim itself has none: batting last and walkoff logic net to ~zero.
HOME_FIELD_LOGIT = 0.09

# v1 → v2 model cutover. Eval / calibration / feature-importance writes are
# blocked before this date so the v1-archive history stays frozen.
V1_CUTOVER_DATE = date(2026, 5, 12)
