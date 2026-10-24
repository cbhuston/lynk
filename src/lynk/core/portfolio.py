"""
The pari-mutuel portfolio solver.

One event (a race) can be bet through several pools at once: win, place, show, exacta, the first leg of a daily
double. Every pool has its own takeout, its own money and its own winners, and every dollar we add moves that pool's
prices. This module finds the stakes that maximize expected log wealth across all of them together.

Formulation (one event, scenarios s = possible results with probability pi_s):

    wealth_s = W - sum(x) + sum over pools k of payout_k,s(x) + sum(rebate_k * x_k)

    payout_k,s(x) = sum over winners w of x_w * (1 + profit_k,s / (n_k,s * (m_w + x_w)))
    profit_k,s    = (1 - takeout_k) * (T_k + X_k) - sum over winners u of (m_u + x_u)

where T_k is the pool total, m the money on each outcome, X_k our total stake in pool k and n_k,s the number of
winners (1 for win / exacta / daily double, 2 for place, 3 for show). The objective is concave in x, so the solution
is the global optimum. Gradients are exact.

Lesson built in from the research: optimize bankroll FRACTIONS, not dollars. In dollars the gradient of log wealth is
about 1e-6 per dollar and quasi-Newton solvers stop at the starting point.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class Pool:
    """
    One pari-mutuel pool for one event.

    Attributes:
    - name: Pool name, e.g. 'win', 'place', 'exacta'.
    - takeout: Operator cut as a fraction.
    - money: Dict mapping every outcome label to the money bet on it (our forecast of the FINAL pool, without us).
             For an exacta the labels are tuples like (3, 7).
    - rebate: Share of each stake returned win or lose.
    - total: Pool total. Defaults to the sum of ``money``.
    """
    name: str
    takeout: float
    money: dict
    rebate: float = 0.0
    total: float = None
    labels: list = field(init=False)

    def __post_init__(self):
        self.labels = list(self.money)
        if self.total is None:
            self.total = float(sum(self.money.values()))


@dataclass
class Scenario:
    """
    One possible result of the event.

    Attributes:
    - probability: Probability of this result.
    - winners: Dict mapping pool name to the tuple of winning outcome labels in that pool, e.g.
               {'win': (3,), 'place': (3, 7), 'exacta': ((3, 7),)}.
    """
    probability: float
    winners: dict


def _index_problem(pools, scenarios, allowed):
    """Flattens pools into one stake vector and precomputes winner positions for every scenario."""
    columns, offsets = [], {}
    for pool in pools:
        offsets[pool.name] = len(columns)
        for label in pool.labels:
            columns.append((pool.name, label))
    lookup = {col: i for i, col in enumerate(columns)}
    money = np.array([dict(p.money)[lab] for p in pools for lab in p.labels], dtype=float)
    pool_of = np.array([k for k, p in enumerate(pools) for _ in p.labels])
    bettable = np.array([allowed is None or col in allowed for col in columns])
    winners = []
    for s in scenarios:
        per_pool = []
        for k, pool in enumerate(pools):
            labels = s.winners.get(pool.name, ())
            per_pool.append(np.array([lookup[(pool.name, lab)] for lab in labels if (pool.name, lab) in lookup], dtype=int))
        winners.append(per_pool)
    return columns, money, pool_of, bettable, winners


def _wealth_and_gradient(f, pools, money_frac, pool_of, winners, probabilities):
    """Wealth per scenario (as a multiple of bankroll) and its gradient with respect to the fractions f."""
    n_bets = len(f)
    takeouts = np.array([p.takeout for p in pools])
    rebates = np.array([p.rebate for p in pools])
    totals = np.array([p.total for p in pools])  # already in bankroll units
    X = np.bincount(pool_of, weights=f, minlength=len(pools))
    base = 1.0 - f.sum() + np.dot(rebates[pool_of], f)
    base_grad = -1.0 + rebates[pool_of]
    wealth = np.empty(len(probabilities))
    grads = np.empty((len(probabilities), n_bets))
    for s, per_pool in enumerate(winners):
        w_total = base
        g = base_grad.copy()
        for k, idx in enumerate(per_pool):
            if len(idx) == 0:
                continue
            n_win = len(idx)
            stake_k = f[idx]
            held = money_frac[idx] + stake_k
            profit = (1 - takeouts[k]) * (totals[k] + X[k]) - held.sum()
            w_total = w_total + np.sum(stake_k * (1 + profit / (n_win * held)))
            share = stake_k / (n_win * held)                                   # d payout / d profit
            in_pool = pool_of == k
            g[in_pool] += share.sum() * (1 - takeouts[k])                      # every stake in pool k grows the pool
            g[idx] += -share.sum()                                             # winners' own money is netted out
            g[idx] += 1 + profit * money_frac[idx] / (n_win * held ** 2)       # direct effect on the winning stakes
        wealth[s] = w_total
        grads[s] = g
    return wealth, grads


def optimize_portfolio(pools, scenarios, bankroll, kelly_multiplier=1.0, allowed=None, max_fraction_per_bet=0.25,
                       min_stake=0.0):
    """
    Finds the stakes across pools that maximize expected log wealth for one event.

    Parameters:
    - pools: List of ``Pool`` objects (money in dollars).
    - scenarios: List of ``Scenario`` objects. Probabilities should sum to about one.
    - bankroll: Bankroll in dollars available for this event.
    - kelly_multiplier: Fractional Kelly applied to the solution (0.5 = half Kelly).
    - allowed: Optional set of (pool_name, label) pairs that may be bet. Everything else is fixed at zero.
    - max_fraction_per_bet: Upper bound on any single stake as a fraction of bankroll.
    - min_stake: Stakes below this many dollars are dropped (e.g. 2.0 for a $2 minimum ticket).

    Returns:
    - DataFrame with one row per non-zero bet: pool, outcome, stake, expected_payout_if_win.
    """
    # Step 1: Express everything in bankroll units so gradients are well scaled
    scaled = [Pool(p.name, p.takeout, {k: v / bankroll for k, v in p.money.items()}, p.rebate, p.total / bankroll)
              for p in pools]
    columns, money_frac, pool_of, bettable, winners = _index_problem(scaled, scenarios, allowed)
    probabilities = np.array([s.probability for s in scenarios], dtype=float)
    probabilities = probabilities / probabilities.sum()
    bounds = [(0.0, max_fraction_per_bet if ok else 0.0) for ok in bettable]

    # Step 2: Concave objective (negative expected log wealth) with its exact gradient
    def objective(f):
        wealth, grads = _wealth_and_gradient(f, scaled, money_frac, pool_of, winners, probabilities)
        wealth = np.clip(wealth, 1e-12, None)
        value = -np.dot(probabilities, np.log(wealth))
        gradient = -((probabilities / wealth) @ grads)
        return value, gradient

    # Step 3: Solve with box bounds; keep total exposure below the bankroll
    start = np.where(bettable, 1e-5, 0.0)
    result = minimize(objective, start, jac=True, method='L-BFGS-B', bounds=bounds,
                      options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 2000})
    stakes = np.clip(result.x, 0.0, None) * bankroll * kelly_multiplier
    if stakes.sum() > bankroll:
        stakes *= bankroll / stakes.sum()

    # Step 4: Report the bets, dropping anything below the minimum ticket
    rows = []
    for (pool_name, label), stake in zip(columns, stakes):
        if stake >= max(min_stake, 1e-9):
            rows.append(dict(pool=pool_name, outcome=label, stake=float(stake)))
    return pd.DataFrame(rows, columns=['pool', 'outcome', 'stake'])


def expected_log_growth(pools, scenarios, bankroll, bets):
    """
    Evaluates expected log growth of a given set of bets (DataFrame from ``optimize_portfolio``).

    Useful for comparing strategies, e.g. win-only versus win + exacta, on the same scenarios.
    """
    scaled = [Pool(p.name, p.takeout, {k: v / bankroll for k, v in p.money.items()}, p.rebate, p.total / bankroll)
              for p in pools]
    columns, money_frac, pool_of, _, winners = _index_problem(scaled, scenarios, None)
    f = np.zeros(len(columns))
    lookup = {c: i for i, c in enumerate(columns)}
    for row in bets.itertuples():
        f[lookup[(row.pool, row.outcome)]] = row.stake / bankroll
    probabilities = np.array([s.probability for s in scenarios], dtype=float)
    probabilities = probabilities / probabilities.sum()
    wealth, _ = _wealth_and_gradient(f, scaled, money_frac, pool_of, winners, probabilities)
    return float(np.dot(probabilities, np.log(np.clip(wealth, 1e-12, None))))


def cap_at_max_ev(stakes, max_ev_stakes, share=2.0 / 3.0):
    """
    Caps each stake at a share of its maximum-expected-value stake (Benter's practical rule).

    The Kelly solution can exceed the stake that maximizes expected dollars when the bankroll is large relative to
    the pool. Capping at two thirds of the max-EV stake keeps most of the expected profit and lowers variance.
    """
    return np.minimum(np.asarray(stakes, dtype=float), share * np.asarray(max_ev_stakes, dtype=float))
