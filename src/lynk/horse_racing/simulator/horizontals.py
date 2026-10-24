"""
Horizontal pools across a card: daily doubles and Pick 3 / 4 / 5.

A horizontal bet is placed before the first leg, when later legs have no win pool yet, so the crowd allocates later
legs from its EARLY opinion of those races (the 30-minute view). The daily double publishes a full money matrix and,
once the first leg is official, will-pays for every second-leg runner. Pick-N pools pay the winning sequence or, when
nobody holds it, carry the net pool over.
"""

import numpy as np

from lynk.horse_racing.simulator.tote import _breakage


def _leg_view(race_state, bias):
    early = race_state['money']['win'][0]
    share = early / early.sum()
    share = share ** bias
    return share / share.sum()


def daily_doubles(world, card_states, rng):
    """
    Daily doubles on consecutive races of a card.

    Parameters:
    - card_states: List (in race order) of dicts with race_id, money, finish, pools.

    Returns:
    - (rows, willpays): rows describe each DD (legs, pool, winning pair, payout per $2), willpays list each second-leg
      runner's will-pay after leg 1.
    """
    E, T = world.params['exotics'], world.params['tote']
    rows, willpays = [], []
    for leg1, leg2 in zip(card_states[:-1], card_states[1:]):
        if rng.random() > E['offered']['daily_double']:
            continue
        pool = leg1['pools']['win'] * E['pool_over_win']['daily_double'] * float(np.exp(rng.normal(0, E['pool_over_win_log_sd'])))
        p1 = leg1['money']['win'][-1] / leg1['money']['win'][-1].sum()
        p2 = _leg_view(leg2, E['second_leg_longshot_bias'] ** 0.5)
        matrix = p1[:, None] * p2[None, :] * np.exp(rng.normal(0, E['combo_noise_sd'], (len(p1), len(p2))))
        matrix /= matrix.sum()
        w1 = int(np.argmin(leg1['finish'])); w2 = int(np.argmin(leg2['finish']))
        net = 2 * (1 - T['takeout']['daily_double'])
        for b in range(len(p2)):
            willpays.append(dict(race_id=leg1['race_id'], leg2_race_id=leg2['race_id'], leg2_post=b + 1,
                                 willpay=float(_breakage(net / max(matrix[w1, b], 1e-9), minimum=2.10))))
        rows.append(dict(bet='daily_double', race_id=leg1['race_id'], legs=f"{leg1['race_id']},{leg2['race_id']}", pool=pool,
                         winning=f'{w1 + 1}/{w2 + 1}', payout=float(_breakage(net / max(matrix[w1, w2], 1e-9), minimum=2.10)),
                         carryover=False))
    return rows, willpays


def pick_n(world, card_states, rng, carry):
    """
    Pick 3, Pick 4 and Pick 5 sequences starting on the card's races.

    Parameters:
    - carry: Dict bet -> carried-over amount from earlier days at this track (updated in place).

    Returns:
    - List of dicts (bet, first race, legs, pool incl. carryover, winning sequence, payout per $2 or NaN, carryover flag).
    """
    E, T = world.params['exotics'], world.params['tote']
    rows = []
    for bet, legs in [('pick3', 3), ('pick4', 4), ('pick5', 5)]:
        for start in range(0, len(card_states) - legs + 1):
            if rng.random() > E['offered'][bet] / max(1, len(card_states) - legs + 1) * (2 if bet == 'pick3' else 1):
                continue
            seq = card_states[start:start + legs]
            pool = seq[0]['pools']['win'] * E['pool_over_win'][bet] * float(np.exp(rng.normal(0, E['pool_over_win_log_sd'])))
            pool += carry.get(bet, 0.0)
            share = 1.0
            winners = []
            for k, st in enumerate(seq):
                view = (st['money']['win'][-1] if k == 0 else st['money']['win'][0])
                view = view / view.sum()
                view = view ** E['second_leg_longshot_bias'] if k > 0 else view
                view = view / view.sum()
                w = int(np.argmin(st['finish']))
                winners.append(w + 1)
                share *= view[w]
            share *= float(np.exp(rng.normal(0, E['combo_noise_sd'] * np.sqrt(legs)) - E['combo_noise_sd'] ** 2 * legs / 2))
            net = (1 - T['takeout'][bet]) * pool
            money_on_winner = share * (pool - carry.get(bet, 0.0))
            if money_on_winner < 0.5:            # nobody holds a 50-cent winning ticket: the pool carries over
                carry[bet] = net
                payout, carried = np.nan, True
            else:
                carry[bet] = 0.0
                payout, carried = float(_breakage(2 * net / money_on_winner, minimum=2.10)), False
            rows.append(dict(bet=bet, race_id=seq[0]['race_id'], legs=','.join(s['race_id'] for s in seq), pool=pool,
                             winning='/'.join(map(str, winners)), payout=payout, carryover=carried))
    return rows
