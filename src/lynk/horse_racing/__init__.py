"""
The horse racing market.

- schema        the starter-level column contract
- features      leakage-safe horse history, connection form and field-relative features
- finish_order  discounted Harville pair / triple / top-k probabilities and finish-order scenarios
- bets          win strategy (forecast final odds + impact-aware Kelly) and the multi-pool race solver
- synthetic     a seeded synthetic racing circuit for examples and tests
"""

from lynk.horse_racing import schema, features, finish_order, bets, synthetic  # noqa: F401
