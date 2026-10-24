"""
Market-agnostic building blocks.

- probability  group softmax, normalization, odds conversion
- metrics      event log-loss, calibration, block bootstrap CIs, drawdown, Sharpe
- kelly        single, fractional, mutually exclusive, scenario and risk-constrained Kelly (fixed odds)
- parimutuel   pool payouts with own-money impact, max-EV bet, breakage, dilution, place/show splits
- portfolio    concave expected-log solver across several pari-mutuel pools for one event
- backtest     ledgers, settlement summaries, grid search, bankroll path simulation
- validation   date splits, walk-forward windows, leakage assertions, pre-registered selection
- ratings      online Kalman entity ratings for multi-outcome events
- models       LightGBM conditional-logit (group softmax) training and prediction
"""

from lynk.core import probability, metrics, kelly, parimutuel, portfolio, backtest, validation, ratings  # noqa: F401
