"""
Group-level probability utilities.

Most wagering markets are made of mutually exclusive outcomes grouped into events: horses in a race, teams in a
match, candidates in an election. These helpers turn per-outcome scores into probabilities that sum to one inside
each event, which is the single most important modeling decision in the framework.
"""

import numpy as np


def group_index(group_ids):
    """
    Builds start offsets and a per-row group number for rows sorted by group.

    Parameters:
    - group_ids: Array of event ids. Rows of the same event must be contiguous.

    Returns:
    - starts: Index of the first row of every event.
    - gid: Event number (0, 1, 2, ...) for every row.
    """
    group_ids = np.asarray(group_ids)
    if len(group_ids) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    is_new_group = np.r_[True, group_ids[1:] != group_ids[:-1]]
    starts = np.flatnonzero(is_new_group)
    gid = np.cumsum(is_new_group) - 1
    return starts, gid


def group_softmax(scores, group_ids):
    """
    Converts scores into probabilities that sum to one inside every event (the conditional logit).

    P(outcome i) = exp(score_i) / sum of exp(score_j) over outcomes j in the same event.

    Parameters:
    - scores: Array of real-valued strengths, one per outcome.
    - group_ids: Array of event ids (contiguous).

    Returns:
    - Array of probabilities with the same shape as ``scores``.
    """
    scores = np.asarray(scores, dtype=float)
    starts, gid = group_index(group_ids)
    # Step 1: Subtract each event's max score so the exponentials cannot overflow
    shifted = scores - np.maximum.reduceat(scores, starts)[gid]
    # Step 2: Exponentiate and normalize inside each event
    weights = np.exp(shifted)
    return weights / np.add.reduceat(weights, starts)[gid]


def normalize_within_groups(values, group_ids, floor=1e-12):
    """
    Rescales non-negative values so they sum to one inside each event.

    Parameters:
    - values: Non-negative array (for example raw money in a pool, or uncalibrated probabilities).
    - group_ids: Array of event ids (contiguous).
    - floor: Minimum value used before normalizing, so no outcome gets exactly zero.

    Returns:
    - Array of probabilities.
    """
    values = np.clip(np.asarray(values, dtype=float), floor, None)
    starts, gid = group_index(group_ids)
    return values / np.add.reduceat(values, starts)[gid]


def odds_to_probability(decimal_odds, group_ids=None):
    """
    Converts decimal odds (payout per $1 including the stake) to implied probabilities.

    Parameters:
    - decimal_odds: Array of decimal odds, e.g. 5.0 for a horse at 4-1.
    - group_ids: Optional event ids. When given, the implied probabilities are normalized inside each event, which
      removes the takeout (the house cut) from the prices.

    Returns:
    - Array of implied probabilities.
    """
    implied = 1.0 / np.asarray(decimal_odds, dtype=float)
    return implied if group_ids is None else normalize_within_groups(implied, group_ids)


def logit(p, eps=1e-9):
    """Log-odds of a probability, clipped away from 0 and 1."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def expit(x):
    """Inverse of ``logit``."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))
