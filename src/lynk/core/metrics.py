"""
Scoring and risk metrics.

Every result in Lynk is reported with a confidence interval from a block bootstrap over days. Wagering outcomes on the
same day are correlated (same tracks, same weather, same late money), so resampling single bets would make results
look more certain than they are.
"""

import numpy as np
import pandas as pd

from lynk.core.probability import group_index


def group_log_loss(probabilities, outcomes, group_ids):
    """
    Average negative log probability given to the realized outcome of each event (lower is better).

    Parameters:
    - probabilities: Model probabilities, one per outcome, summing to one inside each event.
    - outcomes: 1 for the outcome that happened, 0 otherwise. Ties may share the 1 (dead heats).
    - group_ids: Array of event ids (contiguous).

    Returns:
    - Float, mean log-loss per event.
    """
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    outcomes = np.asarray(outcomes, dtype=float)
    starts, gid = group_index(group_ids)
    # Step 1: Share each event's single unit of "winner" across dead-heated outcomes
    totals = np.add.reduceat(outcomes, starts)
    weights = outcomes / np.where(totals > 0, totals, 1.0)[gid]
    # Step 2: Sum -log(p) over winners per event, then average over events that had a winner
    per_event = np.add.reduceat(-weights * np.log(probabilities), starts)
    return float(per_event[totals > 0].mean())


def per_group_log_loss(probabilities, outcomes, group_ids):
    """Same as ``group_log_loss`` but returns one value per event (useful for paired comparisons)."""
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    outcomes = np.asarray(outcomes, dtype=float)
    starts, gid = group_index(group_ids)
    totals = np.add.reduceat(outcomes, starts)
    weights = outcomes / np.where(totals > 0, totals, 1.0)[gid]
    return np.add.reduceat(-weights * np.log(probabilities), starts)


def calibration_table(probabilities, outcomes, n_bins=10):
    """
    Groups predictions into equal-count bins and compares the average prediction with the realized rate.

    Parameters:
    - probabilities: Predicted probabilities.
    - outcomes: Realized 0/1 outcomes.
    - n_bins: Number of quantile bins.

    Returns:
    - DataFrame with columns bin, predicted, actual and n.
    """
    frame = pd.DataFrame({'p': np.asarray(probabilities, dtype=float), 'y': np.asarray(outcomes, dtype=float)})
    frame['bin'] = pd.qcut(frame.p.rank(method='first'), n_bins, labels=False)
    table = frame.groupby('bin').agg(predicted=('p', 'mean'), actual=('y', 'mean'), n=('y', 'size')).reset_index()
    return table


def block_bootstrap(values, blocks, statistic=np.mean, n_boot=2000, seed=0, ci=0.95):
    """
    Bootstrap confidence interval that resamples whole blocks (for example race days) instead of single rows.

    Parameters:
    - values: Array of per-row values (e.g. the log-loss difference between two models for each event).
    - blocks: Array of block labels, same length as ``values`` (e.g. the race date).
    - statistic: Function applied to the resampled values. Defaults to the mean.
    - n_boot: Number of bootstrap resamples.
    - seed: Random seed, so results are reproducible.
    - ci: Confidence level.

    Returns:
    - Tuple (point_estimate, lower, upper).
    """
    values = np.asarray(values, dtype=float)
    labels, block_codes = np.unique(np.asarray(blocks), return_inverse=True)
    members = [np.flatnonzero(block_codes == b) for b in range(len(labels))]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.integers(0, len(members), len(members))
        draws[i] = statistic(values[np.concatenate([members[b] for b in picked])])
    tail = (1 - ci) / 2 * 100
    return float(statistic(values)), float(np.percentile(draws, tail)), float(np.percentile(draws, 100 - tail))


def ratio_bootstrap(numerator, denominator, blocks, n_boot=2000, seed=0, ci=0.95):
    """
    Bootstrap CI for a ratio of sums, such as ROI = total profit / total staked, resampling whole blocks.

    Parameters:
    - numerator: Per-row values summed on top (e.g. profit per bet).
    - denominator: Per-row values summed on the bottom (e.g. stake per bet).
    - blocks: Block labels (e.g. race date).

    Returns:
    - Tuple (point_estimate, lower, upper).
    """
    table = pd.DataFrame({'n': numerator, 'd': denominator, 'b': blocks}).groupby('b')[['n', 'd']].sum()
    n_values, d_values = table.n.values, table.d.values
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.integers(0, len(table), len(table))
        draws[i] = n_values[picked].sum() / max(d_values[picked].sum(), 1e-12)
    tail = (1 - ci) / 2 * 100
    point = n_values.sum() / max(d_values.sum(), 1e-12)
    return float(point), float(np.percentile(draws, tail)), float(np.percentile(draws, 100 - tail))


def log_growth(returns_per_dollar_of_bankroll):
    """
    Sum of log bankroll growth, the quantity the Kelly criterion maximizes.

    Parameters:
    - returns_per_dollar_of_bankroll: Array of period returns as a fraction of bankroll (0.02 = +2%).

    Returns:
    - Float, total log growth. exp(total) is the final bankroll multiple.
    """
    r = np.asarray(returns_per_dollar_of_bankroll, dtype=float)
    return float(np.log1p(np.clip(r, -0.999999, None)).sum())


def max_drawdown(cumulative_pnl):
    """
    Largest peak-to-trough fall of a cumulative profit series, in the same units as the series.

    Parameters:
    - cumulative_pnl: Array of running profit (not returns).

    Returns:
    - Float, a negative number (or 0.0 when the series never falls).
    """
    series = np.asarray(cumulative_pnl, dtype=float)
    if len(series) == 0:
        return 0.0
    return float((series - np.maximum.accumulate(series)).min())


def annualized_sharpe(period_pnl, periods_per_year=252):
    """Mean over standard deviation of period profit, scaled to a year (zero when there is no variation)."""
    pnl = np.asarray(period_pnl, dtype=float)
    sd = pnl.std(ddof=1) if len(pnl) > 1 else 0.0
    return float(pnl.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0
