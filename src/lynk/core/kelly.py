"""
Kelly sizing, from the single bet up to general scenario portfolios.

Everything here assumes FIXED odds (the price does not move when you bet). Pools where your own money changes the
price are handled in ``lynk.core.parimutuel`` and ``lynk.core.portfolio``, which build on these functions.

References:
- Kelly (1956), A New Interpretation of Information Rate.
- Smoczynski and Tomkins (2010), optimal allocations for mutually exclusive outcomes.
- Busseti, Ryu and Boyd (2016), Risk-Constrained Kelly Gambling.
"""

import numpy as np
from scipy.optimize import minimize


def kelly_fraction(p, decimal_odds):
    """
    Full Kelly stake as a fraction of bankroll for one bet with fixed odds.

    f* = (p * b - (1 - p)) / b, where b = decimal_odds - 1 is the net profit per $1 when the bet wins.

    Parameters:
    - p: Probability the bet wins.
    - decimal_odds: Payout per $1 including the stake (5.0 means a $1 bet returns $5).

    Returns:
    - Fraction of bankroll to bet (0 when the bet has no edge).
    """
    p = np.asarray(p, dtype=float)
    b = np.asarray(decimal_odds, dtype=float) - 1.0
    return np.clip((p * b - (1.0 - p)) / b, 0.0, None)


def fractional_kelly(p, decimal_odds, fraction=0.5):
    """
    Kelly stake scaled down by ``fraction``.

    Betting half Kelly gives up about a quarter of the growth rate but cuts the volatility of the bankroll in half,
    and it protects against probabilities that are a little too optimistic (Baker and McHale, 2013).
    """
    return fraction * kelly_fraction(p, decimal_odds)


def expected_log_growth(f, p, decimal_odds):
    """
    Expected log growth per bet of staking fraction ``f`` on a single fixed-odds bet.

    G(f) = p * log(1 + f * b) + (1 - p) * log(1 - f).
    """
    f = np.asarray(f, dtype=float)
    b = decimal_odds - 1.0
    return p * np.log1p(f * b) + (1 - p) * np.log1p(-f)


def kelly_mutually_exclusive(probabilities, decimal_odds):
    """
    Exact Kelly allocation across mutually exclusive outcomes (one horse wins, one team wins).

    Only one outcome can pay, so the bets hedge each other and the optimal set can include outcomes that look
    marginal on their own. The algorithm (Kelly 1956, Smoczynski and Tomkins 2010):

    1. Rank outcomes by expected return p_i * o_i, best first.
    2. Grow the betting set S while the next outcome's expected return beats the reserve rate
       R(S) = (1 - sum of p over S) / (1 - sum of 1/o over S).
    3. Bet f_i = p_i - R / o_i on every outcome in S, and keep the rest of the bankroll.

    Parameters:
    - probabilities: Array of win probabilities for every outcome of the event (should sum to about one).
    - decimal_odds: Array of fixed decimal odds for the same outcomes.

    Returns:
    - Array of bankroll fractions, one per outcome.
    """
    p = np.asarray(probabilities, dtype=float)
    o = np.asarray(decimal_odds, dtype=float)
    # Step 1: Order outcomes by expected return per $1
    order = np.argsort(-(p * o))
    in_set = []
    reserve = 1.0
    # Step 2: Add outcomes while they beat the current reserve rate
    for i in order:
        if p[i] * o[i] <= reserve:
            break
        in_set.append(i)
        prob_out = 1.0 - p[in_set].sum()
        price_out = 1.0 - (1.0 / o[in_set]).sum()
        if price_out <= 0:  # the set's prices cover the whole event: arbitrage, bet the set fully
            reserve = 0.0
            break
        reserve = prob_out / price_out
    # Step 3: Stake p_i - R / o_i on each outcome in the set
    fractions = np.zeros_like(p)
    if in_set:
        idx = np.array(in_set)
        fractions[idx] = np.clip(p[idx] - reserve / o[idx], 0.0, None)
    return fractions


def kelly_scenarios(gross_returns, scenario_probabilities, fraction=1.0, max_total=1.0):
    """
    Numerical Kelly for any set of bets described by scenarios.

    Each scenario s happens with probability pi_s and pays gross return R[s, j] per $1 on bet j (0 when the bet
    loses). The solver maximizes sum_s pi_s * log(1 - sum_j f_j + sum_j f_j * R[s, j]). This covers parlays,
    correlated bets across events and any mix of bet types as long as the odds are fixed.

    Parameters:
    - gross_returns: Array (n_scenarios, n_bets) of payouts per $1 staked, including the stake.
    - scenario_probabilities: Array (n_scenarios,) summing to one.
    - fraction: Fractional Kelly multiplier applied to the solution.
    - max_total: Maximum share of the bankroll that can be at risk at once.

    Returns:
    - Array (n_bets,) of bankroll fractions.
    """
    R = np.asarray(gross_returns, dtype=float)
    pi = np.asarray(scenario_probabilities, dtype=float)
    n_bets = R.shape[1]

    def negative_growth(f):
        wealth = 1.0 - f.sum() + R @ f
        return -np.dot(pi, np.log(np.clip(wealth, 1e-12, None)))

    def gradient(f):
        wealth = np.clip(1.0 - f.sum() + R @ f, 1e-12, None)
        return -((pi / wealth) @ (R - 1.0))

    result = minimize(negative_growth, np.full(n_bets, 1e-3), jac=gradient, method='SLSQP',
                      bounds=[(0.0, max_total)] * n_bets,
                      constraints=[{'type': 'ineq', 'fun': lambda f: max_total - f.sum()}],
                      options={'ftol': 1e-12, 'maxiter': 500})
    return fraction * np.clip(result.x, 0.0, None)


def risk_constrained_kelly(gross_returns, scenario_probabilities, alpha=0.7, beta=0.1):
    """
    Risk-constrained Kelly (Busseti, Ryu and Boyd, 2016).

    Maximizes expected log growth subject to sum_s pi_s * (wealth_s) ** (-lam) <= 1 with lam = log(beta) / log(alpha).
    That constraint bounds the probability that wealth ever falls below ``alpha`` times its starting value by
    ``beta``. For example alpha=0.7, beta=0.1 means "at most a 10% chance of ever losing 30% of the bankroll".
    Unlike fractional Kelly, the risk budget is stated directly in drawdown terms.

    Parameters:
    - gross_returns: Array (n_scenarios, n_bets) of payouts per $1, including the stake.
    - scenario_probabilities: Array (n_scenarios,).
    - alpha: Drawdown level as a fraction of starting wealth (0 < alpha < 1).
    - beta: Allowed probability of ever hitting that drawdown.

    Returns:
    - Array (n_bets,) of bankroll fractions.
    """
    R = np.asarray(gross_returns, dtype=float)
    pi = np.asarray(scenario_probabilities, dtype=float)
    lam = np.log(beta) / np.log(alpha)
    n_bets = R.shape[1]
    wealth = lambda f: np.clip(1.0 - f.sum() + R @ f, 1e-12, None)
    result = minimize(lambda f: -np.dot(pi, np.log(wealth(f))), np.full(n_bets, 1e-4), method='SLSQP',
                      bounds=[(0.0, 1.0)] * n_bets,
                      constraints=[{'type': 'ineq', 'fun': lambda f: 1.0 - f.sum()},
                                   {'type': 'ineq', 'fun': lambda f: -np.log(np.dot(pi, wealth(f) ** (-lam)))}],
                      options={'ftol': 1e-12, 'maxiter': 500})
    return np.clip(result.x, 0.0, None)
