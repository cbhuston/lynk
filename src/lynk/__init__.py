"""
Lynk: a quantitative wagering framework.

The package is split into a market-agnostic core and one module per market:

- ``lynk.core``          probabilities, metrics, Kelly sizing, pari-mutuel pricing, the portfolio solver,
                         backtesting, leakage-safe validation and online entity ratings.
- ``lynk.horse_racing``  the horse racing market: race schema, handicapping features, finish-order models,
                         bet types (win, place, show, exacta, daily double) and a synthetic race generator.

New markets (sports exchanges, prediction markets, other pari-mutuel games) plug into the same core by adding
a sibling module next to ``horse_racing``.
"""

__version__ = "0.1.0"

from lynk import core  # noqa: F401
