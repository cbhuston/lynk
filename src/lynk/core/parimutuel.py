"""
Pari-mutuel pricing with your own money in the pool.

In a pari-mutuel pool every dollar bet on an outcome is pooled, the operator takes a cut (the takeout) and the rest is
split among the winning tickets. Unlike fixed odds, your own bet changes the price you get:

    payout per $1 on outcome i = (1 - takeout) * (pool_total + stake) / (money_on_i + stake)

So the first dollar you bet is worth more than the last one, and there is a bet size where expected profit in dollars
peaks (Isaacs, 1953). Everything in this module is vectorized over arrays of bets.
"""

import numpy as np


def payout_per_dollar(pool_total, money_on_outcome, takeout, stake=0.0):
    """
    Gross payout per $1 if the outcome wins, after adding ``stake`` to the pool.

    Parameters:
    - pool_total: Total money in the pool before your bet.
    - money_on_outcome: Money already bet on the outcome you are backing.
    - takeout: Operator cut as a fraction (0.17 = 17%).
    - stake: Your bet size, added to both the pool and the outcome.

    Returns:
    - Payout per $1 including the stake (decimal odds). Rebates are paid separately (see ``expected_profit``).
    """
    pool_total = np.asarray(pool_total, dtype=float)
    money_on_outcome = np.asarray(money_on_outcome, dtype=float)
    stake = np.asarray(stake, dtype=float)
    return (1.0 - takeout) * (pool_total + stake) / (money_on_outcome + stake)


def expected_profit(p, pool_total, money_on_outcome, takeout, stake, rebate=0.0):
    """
    Expected profit in dollars of a single pari-mutuel bet, including own-money impact and rebate.

    E[profit] = p * stake * payout(stake) + rebate * stake - stake
    """
    stake = np.asarray(stake, dtype=float)
    gross = payout_per_dollar(pool_total, money_on_outcome, takeout, stake)
    return p * stake * gross + rebate * stake - stake


def max_ev_bet(p, pool_total, money_on_outcome, takeout, rebate=0.0):
    """
    The stake that maximizes expected profit in dollars (Isaacs, 1953), extended with a rebate.

    Setting d/ds of E[profit] to zero gives a quadratic with the closed-form root

        s* = -m + sqrt(a * m * (T - m) / (1 - rebate - a)),   a = p * (1 - takeout)

    where T is the pool total and m the money already on the outcome. Betting more than s* loses expected dollars
    even on a good bet, because each extra dollar mostly buys back your own money.

    Parameters:
    - p: Win probability.
    - pool_total, money_on_outcome, takeout: As in ``payout_per_dollar``.
    - rebate: Share of each stake returned win or lose.

    Returns:
    - Optimal stake in dollars (0 when the bet has no edge at a $0 stake).
    """
    p = np.asarray(p, dtype=float)
    T = np.asarray(pool_total, dtype=float)
    m = np.asarray(money_on_outcome, dtype=float)
    a = p * (1.0 - takeout)
    denominator = np.clip(1.0 - rebate - a, 1e-12, None)
    root = np.sqrt(np.clip(a * m * (T - m) / denominator, 0.0, None))
    edge_at_zero = a * T / np.clip(m, 1e-12, None) + rebate > 1.0
    return np.where(edge_at_zero, np.clip(root - m, 0.0, None), 0.0)


def apply_breakage(gross_payout, base=2.0, increment=0.10, minimum=None):
    """
    Rounds payouts down the way US tracks do (to the dime per $2 ticket), then applies a minimum payout.

    Parameters:
    - gross_payout: Payout per $1 before rounding.
    - base: Ticket size the rounding is applied to ($2 in the US).
    - increment: Rounding unit for the ticket ($0.10, "dime breakage").
    - minimum: Optional minimum payout per $1 (1.05 for US place and show, the "$2.10 minimum").

    Returns:
    - Payout per $1 after breakage.
    """
    per_ticket = np.floor(np.asarray(gross_payout, dtype=float) * base / increment + 1e-9) * increment
    payout = per_ticket / base
    return payout if minimum is None else np.maximum(payout, minimum)


def diluted_actual_payout(actual_payout, money_on_outcome, stake, minimum=None):
    """
    Adjusts a historical payout for the dilution your own bet would have caused.

    When backtesting with real results we know the official payout, which was computed without our money. Adding a
    stake s to an outcome that had m dollars on it shrinks the profit part of the payout by m / (m + s).

    Parameters:
    - actual_payout: Official payout per $1 (decimal odds) from the results chart.
    - money_on_outcome: Estimated final money on the outcome (without us).
    - stake: Our stake.
    - minimum: Optional payout floor per $1.

    Returns:
    - Payout per $1 we would have received.
    """
    actual_payout = np.asarray(actual_payout, dtype=float)
    m = np.asarray(money_on_outcome, dtype=float)
    stake = np.asarray(stake, dtype=float)
    payout = 1.0 + (actual_payout - 1.0) * m / (m + stake)
    return payout if minimum is None else np.maximum(payout, minimum)


def place_payout(pool_total, money_on_horse, money_on_other_placer, takeout, stake=0.0):
    """
    US place-pool payout per $1 for a horse that finishes in the top two, given the other placing horse.

    The net pool minus the money on both placing horses is split in half between them; each half is shared by that
    horse's backers. Floor of 1.05 (the $2.10 minimum).
    """
    T = np.asarray(pool_total, dtype=float) + stake
    mi = np.asarray(money_on_horse, dtype=float) + stake
    profit = np.clip((1.0 - takeout) * T - mi - np.asarray(money_on_other_placer, dtype=float), 0.0, None) / 2.0
    return np.maximum(1.0 + profit / mi, 1.05)


def show_payout(pool_total, money_on_horse, money_on_other_two, takeout, stake=0.0):
    """
    US show-pool payout per $1 for a horse that finishes in the top three, given the money on the other two.

    The profit is split in thirds between the three placing horses. Floor of 1.05.
    """
    T = np.asarray(pool_total, dtype=float) + stake
    mi = np.asarray(money_on_horse, dtype=float) + stake
    profit = np.clip((1.0 - takeout) * T - mi - np.asarray(money_on_other_two, dtype=float), 0.0, None) / 3.0
    return np.maximum(1.0 + profit / mi, 1.05)
