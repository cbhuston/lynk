"""
From win probabilities to full finish orders.

Exotic and place/show bets need the probability of whole finishing orders, not just the winner. Harville (1973)
assumes the race after the winner is a new race among the remaining horses. That overrates favorites for second and
third, so the discounted version (Lo, Bacon-Shone and Busche, 1995) raises the win probabilities to a power below one
for the lower positions: gamma ~ 0.80 for second and ~ 0.60 for third worked well in the Lynk research
(Benter, 1994 reported 0.81 and 0.65).
"""

from itertools import permutations

import numpy as np


def _discount(p, gamma):
    q = np.asarray(p, dtype=float) ** gamma
    return q / q.sum()


def pair_probabilities(p, gamma_second=0.80):
    """
    Matrix P[i, j] = probability that horse i wins and horse j finishes second (discounted Harville).

    Parameters:
    - p: Win probabilities for one race (sum to one).
    - gamma_second: Discount exponent for second place (1.0 = plain Harville).
    """
    p = np.asarray(p, dtype=float)
    q = _discount(p, gamma_second)
    matrix = p[:, None] * q[None, :] / np.clip(1.0 - q[:, None], 1e-12, None)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def triple_probabilities(p, gamma_second=0.80, gamma_third=0.60):
    """
    Tensor P[i, j, k] = probability of the exact order i first, j second, k third (discounted Harville).
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    q3 = _discount(p, gamma_third)
    pairs = pair_probabilities(p, gamma_second)
    tensor = pairs[:, :, None] * q3[None, None, :] / np.clip(1.0 - q3[:, None, None] - q3[None, :, None], 1e-12, None)
    idx = np.arange(n)
    tensor[idx, :, idx] = 0.0
    tensor[:, idx, idx] = 0.0
    return tensor


def top_k_probabilities(p, k, gamma_second=0.80, gamma_third=0.60):
    """
    Probability that each horse finishes in the top k (k = 1, 2 or 3). Used to price place (k=2) and show (k=3).
    """
    p = np.asarray(p, dtype=float)
    if k == 1:
        return p.copy()
    pairs = pair_probabilities(p, gamma_second)
    if k == 2:
        return p + pairs.sum(axis=0)
    triples = triple_probabilities(p, gamma_second, gamma_third)
    return p + pairs.sum(axis=0) + triples.sum(axis=(0, 1))


def finish_order_scenarios(p, depth=3, gamma_second=0.80, gamma_third=0.60, min_probability=0.0):
    """
    Enumerates finishing orders down to ``depth`` places with their probabilities.

    Parameters:
    - p: Win probabilities for one race.
    - depth: 1 (winner only), 2 (first two) or 3 (first three).
    - min_probability: Drop orders less likely than this (renormalize afterwards if needed).

    Returns:
    - List of (order_tuple, probability) where order_tuple holds 0-based horse positions.
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    if depth == 1:
        table = {(i,): p[i] for i in range(n)}
    elif depth == 2:
        pairs = pair_probabilities(p, gamma_second)
        table = {(i, j): pairs[i, j] for i, j in permutations(range(n), 2)}
    else:
        triples = triple_probabilities(p, gamma_second, gamma_third)
        table = {(i, j, k): triples[i, j, k] for i, j, k in permutations(range(n), 3)}
    return [(order, float(prob)) for order, prob in table.items() if prob > min_probability]
