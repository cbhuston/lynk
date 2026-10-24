# Lynk

Lynk is a Python framework for quantitative wagering. It takes the most reusable pieces of my horse racing research and packages them so they can be installed, understood and extended to new markets:

- **Probabilities that respect the event.** Group softmax (conditional logit) models where the outcomes of a race, match or election always sum to one, including a LightGBM objective that can start from the market's odds and learn corrections.
- **Bet sizing from Kelly to a portfolio solver.** Single, fractional, mutually exclusive, scenario and risk-constrained Kelly, plus a concave solver that sizes bets across several pari-mutuel pools while accounting for how your own money moves each price.
- **Leakage-safe validation and honest backtests.** Date splits, walk-forward windows, an assertion that fails when a feature arrives after the decision time, pre-registered strategy selection and results with day-block bootstrap confidence intervals.
- **A horse racing module** with handicapping features, discounted Harville finish-order models, win / place / show / exacta bets and a synthetic racing circuit, so every example runs without any proprietary data.

The research write-ups live at **[cbhuston.github.io](https://cbhuston.github.io/)**:

1. [Handicapping Without the Tote](https://cbhuston.github.io/reports/fundamental-model.html): what drives a fundamental horse racing model.
2. [Sizing the Bet](https://cbhuston.github.io/reports/bet-sizing.html): from the Kelly criterion to a pari-mutuel portfolio solver.
3. [The Lynk Framework](https://cbhuston.github.io/code/lynk-framework.html): design, architecture and a walkthrough of this package.

## Install

```bash
pip install "lynk[models] @ git+https://github.com/cbhuston/lynk"
```

The core needs only NumPy, pandas and SciPy. The `models` extra adds LightGBM for the conditional logit models.

## Quickstart

```bash
python examples/quickstart.py
```

The quickstart simulates a year of racing, builds leakage-safe features, rates trainers and jockeys, fits a fundamental model and a market model, chooses a win-pool strategy on a validation period with a pre-registered rule and scores the test period once. It runs in about 15 seconds.

A smaller taste of the core:

```python
from lynk.core import kelly, parimutuel

# Full Kelly for a 25% horse at 4-1 (decimal odds 5.0)
kelly.kelly_fraction(0.25, 5.0)                       # 0.0625

# The bet that maximizes expected dollars in a $100k pool with $15k already on the horse, 17% takeout
parimutuel.max_ev_bet(0.25, 100_000, 15_000, 0.17)    # about $3,271
```

## Layout

```
src/lynk/
  core/            market-agnostic: probability, models, ratings, kelly, parimutuel, portfolio,
                   backtest, validation, metrics
  horse_racing/    schema, features, finish_order, bets, synthetic
examples/          quickstart.py
examples/figures/  scripts that make the bet sizing and framework figures for the research site
tests/             unit tests for the math and the leakage guards
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Scope

Lynk ships no proprietary data. Racing data is licensed, so the horse racing module defines a column schema and includes a synthetic circuit instead of data parsers. Nothing in this repository is betting advice: a backtest whose confidence interval includes zero is not proven, and the framework is built to say so.

## License

MIT. See [LICENSE](LICENSE).

## Author

Cooper Huston
