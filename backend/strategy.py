"""Shared production market switches and model-policy constants."""
import os
from datetime import date

# Grading thresholds for picks published before PICK_POLICY (mlb.grading_rules
# mirrors them). They no longer select picks: on the 2026 v2 games the sim
# edge they gated was noise (327 replayed moneyline picks hit 42.2% against a
# claimed 51.1% and a market-implied 42.3%).
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

# Daily pick policy: each game publishes at most one pick, the moneyline or
# run-line side with the highest expected value at its best executable price,
# priced from the paired de-vigged consensus. Every pick is graded at 1u.
PICK_POLICY = "mlb-picks-v3"

# Sim weight in the logit blend behind pick probabilities (moneyline only; run
# lines use the consensus alone). Fitted on the 2026 v2 games with a raw sim
# and a morning consensus (1,254, May 12 to Aug 17): 0.068 on all of them,
# 0.25 / 0.04 / 0.00 walking forward through June, July and August. The sim's
# coefficient beyond the market is 0.04 +- 0.30, so the lean stays small.
PICK_MODEL_WEIGHT = 0.07

# Lowest expected value per unit staked that still publishes a pick; at or
# above zero the pick is tier "value", below it "lean". A volume choice, not an
# edge: on the 2026 three-book games (Jul 20 on, 971) -0.0125 publishes 2.7
# picks a day with an expected -0.65% per pick, and only 31 games (3%) reached
# zero. The market is calibrated and the three books differ by 0.4 points, so
# the price, not the model, decides the expected value.
PICK_MIN_EV = -0.0125

# Home-field shift on the sim's home logit, applied before market anchoring.
# The sim itself has none: batting last and walkoff logic net to ~zero.
HOME_FIELD_LOGIT = 0.09

# v1 → v2 model cutover. Eval / calibration / feature-importance writes are
# blocked before this date so the v1-archive history stays frozen.
V1_CUTOVER_DATE = date(2026, 5, 12)
