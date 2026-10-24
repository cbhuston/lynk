"""
The tote: pool-by-pool money for one race, cycle by cycle, and the official payoffs.

How money arrives:
- The final size of the win, place and show pools depends on purse, field size and the track. A share of that money
  is already in the pool at each time before post (30 minutes, 20 minutes, ... the post-time cycle, the final cycle).
- Each slice of money is split across horses by a mixture of two kinds of bettor. The public bets its own opinion,
  anchored on the morning line early, with a favorite-longshot bias (longshots get more than they deserve).
  Informed bettors bet something close to the true chances, and their share grows toward the close, which is why
  winners tend to shorten late.
- Splits are drawn from a Dirichlet distribution whose concentration grows with the dollars involved, so small pools
  are noisier than big ones.
- Place and show money follows the win money with their own slopes and horse-level noise. Exotic combinations are
  priced from the same crowd with a stronger longshot bias in the lower positions.
"""

import numpy as np

from lynk.horse_racing.simulator.racing import finish_probabilities


def _breakage(payout_per_2, increment=0.10, minimum=None):
    out = np.floor(np.asarray(payout_per_2, dtype=float) / increment + 1e-9) * increment
    return out if minimum is None else np.maximum(out, minimum)


def crowd_beliefs(world, comp, rng):
    """
    The three opinions that drive betting in one race.

    Returns:
    - Dict with p_public (the crowd's view), p_ml (the morning line), p_informed (sharp money's view) and the
      morning line odds as printed.
    """
    C, Rn = world.params['crowd'], world.params['running']
    noise = Rn['performance_noise_sd']
    visible = comp['total'] - comp['trainer_hidden'] - comp['jockey_hidden']
    visible = C['visible_share'] * visible + (1 - C['visible_share']) * visible.mean()
    public = visible + rng.normal(0, C['public_noise_sd'], len(visible))
    ml_view = visible + rng.normal(0, C['morning_line_noise_sd'], len(visible))
    informed = comp['total'] + rng.normal(0, C['informed_noise_sd'], len(visible))
    gm = Rn.get('gumbel_mix', 0.0)
    p_public = finish_probabilities(public, noise, rng, 2000, gm)['win']
    p_ml = finish_probabilities(ml_view, noise, rng, 2000, gm)['win']
    p_inf = finish_probabilities(informed, noise, rng, 2000, gm)['win']
    clip = lambda p: np.clip(p, 0.002, None) / np.clip(p, 0.002, None).sum()
    # The public's opinion evolves: early noise fades and part of its blind spot (truth - visible) is learned over time
    offsets = [str(o) for o in world.params['tote']['offsets']] + ['final']
    blind = comp['total'] - visible
    walk = np.cumsum(rng.normal(0, 1, (len(offsets), len(visible))), axis=0) / np.sqrt(np.arange(1, len(offsets) + 1))[:, None]
    path = {}
    for k, key in enumerate(offsets):
        fade = 1 - k / (len(offsets) - 1)
        learned = _interp_offset(C.get('info_learned', {}), key)
        score = public + learned * blind + C.get('early_noise_sd', 0.0) * fade * walk[k]
        path[key] = clip(finish_probabilities(score, noise, rng, 800, gm)['win'])
    ml_steps = np.array([0.5, 0.6, 0.8, 1, 1.2, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 8, 10, 12, 15, 20, 30, 50])
    ml_odds = ml_steps[np.argmin(np.abs(ml_steps[None, :] - (0.82 / clip(p_ml) - 1)[:, None]), axis=1)]
    return dict(p_public=clip(p_public), p_ml=clip(p_ml), p_informed=clip(p_inf), ml_odds=ml_odds, public_path=path)


def _interp_offset(table, key):
    if not table:
        return 0.0
    if key in table:
        return table[key]
    if key == 'final':
        return table.get('final', max(table.values()))
    xs = sorted(int(k) for k in table if k != 'final')
    return float(np.interp(-int(key), [-x for x in xs], [table[str(x)] for x in xs]))


def informed_weight(crowd_params, key):
    """Share of a money slice placed by informed bettors at a given offset, interpolated between the listed offsets."""
    table = crowd_params['informed_weight']
    if key in table:
        return table[key]
    xs = sorted(int(k) for k in table if k != 'final')
    return float(np.interp(int(key), xs, [table[str(x)] for x in xs]))


def _arrival_curve(world, rng):
    """Fraction of the FINAL pool already bet at each offset, with race-to-race variation (monotonic by construction)."""
    T = world.params['tote']
    offsets = T['offsets']
    base = np.array([T['arrival_at_offset'][str(o)] for o in offsets])
    at_post = np.clip(base[-1] * np.exp(rng.normal(0, 0.15)), 0.35, 0.97)
    shape = np.exp(rng.normal(0, T['arrival_log_sd']))
    return at_post * (base / base[-1]) ** shape


def pool_sizes(world, race, n, rng):
    """Final pool totals for every pool offered on this race."""
    T, E = world.params['tote'], world.params['exotics']
    track = race['track']
    log_wps = (T['wps_pool_intercept'] + T['wps_pool_log_purse'] * np.log(race['purse']) + T['wps_pool_field'] * n
               + world.tracks.pool_effect.values[track] + rng.normal(0, T['wps_pool_sd']))
    wps = float(np.exp(log_wps))
    win_share = float(np.clip(rng.normal(T['win_share_of_wps'], T.get('wps_share_sd', 0.0)), 0.5, 0.85))
    place_share = float(np.clip(rng.normal(T['place_share_of_wps'], T.get('wps_share_sd', 0.0) * 0.6), 0.1, 0.3))
    win = wps * win_share; place = wps * place_share; show = wps - win - place
    pools = dict(win=win, place=place, show=show)
    for name in ['exacta', 'trifecta', 'superfecta']:
        if rng.random() < E['offered'][name] and (name != 'superfecta' or n >= 5):
            pools[name] = win * E['pool_over_win'][name] * float(np.exp(rng.normal(0, E['pool_over_win_log_sd'])))
    return pools


def simulate_win_place_show(world, race, beliefs, pools, rng):
    """
    Money in the win, place and show pools at every offset and in the final cycle.

    Returns:
    - Dict with offsets, win / place / show money arrays of shape (n_offsets + 1, n) (last row = final cycle),
      and the arrival curve.
    """
    C, T = world.params['crowd'], world.params['tote']
    offsets = T['offsets']
    n = len(beliefs['p_public'])
    arrival = np.r_[_arrival_curve(world, rng), 1.0]
    keys = [str(o) for o in offsets] + ['final']
    win_money = np.zeros((len(keys), n))
    cumulative = np.zeros(n)
    previous = 0.0
    # Step 1: Win money arrives in slices; each slice is a public / informed mixture with Dirichlet noise
    for k, (key, frac) in enumerate(zip(keys, arrival)):
        amount = max(frac - previous, 1e-9) * pools['win']
        previous = frac
        offset = offsets[k] if k < len(offsets) else 0
        anchor = C['ml_anchor'] * offset / max(offsets)
        view = beliefs.get('public_path', {}).get(key, beliefs['p_public'])
        public = (1 - anchor) * view + anchor * beliefs['p_ml']
        public = public ** C['longshot_bias']; public /= public.sum()
        floor = min(C.get('floor_share', 0.0), 0.5 / n)
        public = (1 - n * floor) * public + floor
        omega = informed_weight(C, key)
        q = (1 - omega) * public + omega * beliefs['p_informed']
        concentration = float(np.clip(amount * C['dirichlet_concentration_per_dollar'], C.get('dirichlet_min_concentration', 30), 50000))
        if key == 'final':   # late money lands in concentrated bursts on a few horses
            q = q * np.exp(rng.normal(0, C.get('late_shock_sd', 0.0), n)); q /= q.sum()
        cumulative = cumulative + amount * rng.dirichlet(q * concentration + 1e-6)
        win_money[k] = cumulative
    # Step 2: Place and show money follow the win money with their own slopes and horse-level preferences
    place_pref = rng.normal(0, C['place_noise_sd'], n); show_pref = rng.normal(0, C['show_noise_sd'], n)
    place_money = np.zeros_like(win_money); show_money = np.zeros_like(win_money)
    for k in range(len(keys)):
        share = win_money[k] / win_money[k].sum()
        pl = np.exp(C['place_slope'] * np.log(share) + place_pref + rng.normal(0, 0.05, n)); pl /= pl.sum()
        sh = np.exp(C['show_slope'] * np.log(share) + show_pref + rng.normal(0, 0.05, n)); sh /= sh.sum()
        place_money[k] = pl * pools['place'] * arrival[k]; show_money[k] = sh * pools['show'] * arrival[k]
    return dict(keys=keys, arrival=arrival, win=win_money, place=place_money, show=show_money)


def wps_payoffs(money_final, finish, takeout, pool):
    """
    Official win, place and show payoffs per $2 from the final pools (dime breakage, $2.10 minimum, minus pools paid
    at the minimum).

    Parameters:
    - money_final: Dict with final win / place / show money arrays.
    - finish: Finish positions (1 = winner).
    - takeout: Dict of takeout rates by pool.
    """
    n = len(finish)
    out = dict(win=np.zeros(n), place=np.zeros(n), show=np.zeros(n))
    w = int(np.flatnonzero(finish == 1)[0])
    m = money_final['win']
    out['win'][w] = _breakage(2 * (1 - takeout['win']) * m.sum() / m[w], minimum=2.10)
    for pool, k in [('place', 2), ('show', 3)]:
        mm = money_final[pool]
        placed = np.flatnonzero(finish <= k)
        profit = (1 - takeout[pool]) * mm.sum() - mm[placed].sum()
        for i in placed:
            out[pool][i] = _breakage(2 * (1 + max(profit, 0) / len(placed) / mm[i]), minimum=2.10)
    return out


def exacta_matrix(world, win_share, rng):
    """
    Crowd shares of every exacta combination: the winner leg follows the win shares, the second leg uses a stronger
    longshot bias, and every combination carries its own lognormal noise.

    Returns:
    - (n, n) array of shares summing to one (zero diagonal).
    """
    E = world.params['exotics']
    lower = win_share ** E['second_leg_longshot_bias']
    lower = lower / lower.sum()
    cond = lower[None, :] / np.clip(1 - lower[:, None], 1e-9, None)
    matrix = win_share[:, None] * cond
    np.fill_diagonal(matrix, 0.0)
    matrix = matrix * np.exp(rng.normal(0, E['combo_noise_sd'], matrix.shape))
    np.fill_diagonal(matrix, 0.0)
    return matrix / matrix.sum()


def combination_share(world, win_share, combo, rng):
    """
    Crowd share of one ordered combination (trifecta, superfecta, ...), computed in closed form: the noise-free
    shares of all combinations sum to one, so only the winning combination needs to be evaluated.
    """
    E = world.params['exotics']
    lower = win_share ** E['second_leg_longshot_bias']
    lower = lower / lower.sum()
    share, used = win_share[combo[0]], [combo[0]]
    for leg in combo[1:]:
        share *= lower[leg] / max(1 - lower[used].sum(), 1e-9)
        used.append(leg)
    sd = E['combo_noise_sd']
    return float(share * np.exp(rng.normal(0, sd) - sd ** 2 / 2))


def exotic_payoffs(world, money, finish, pools, rng):
    """
    Exacta, trifecta and superfecta: exacta probables at three offsets, and official payoffs per $2.

    Returns:
    - (probables, payoffs): probables is a list of (offset_key, first, second, payout_per_2); payoffs is a dict
      pool -> (winning combination, payout_per_2).
    """
    takeout = world.params['tote']['takeout']
    order = np.argsort(finish)
    probables, payoffs = [], {}
    n = len(finish)
    final_share = money['win'][-1] / money['win'][-1].sum()
    for pool, depth in [('exacta', 2), ('trifecta', 3), ('superfecta', 4)]:
        if pool not in pools or n < depth:
            continue
        winning = tuple(int(i) for i in order[:depth])
        if pool == 'exacta':
            share = exacta_matrix(world, final_share, rng)[winning]
            for key in ['300', '60', '0']:
                row = money['keys'].index(key)
                seen = money['win'][row] / money['win'][row].sum()
                matrix = exacta_matrix(world, seen, rng)
                for a in range(n):
                    for b in range(n):
                        if a != b:
                            probables.append((key, a, b, float(_breakage(2 * (1 - takeout['exacta']) / max(matrix[a, b], 1e-9)))))
        else:
            share = combination_share(world, final_share, winning, rng)
        payoffs[pool] = (winning, float(_breakage(2 * (1 - takeout[pool]) / max(share, 1e-9), minimum=2.10)))
    return probables, payoffs
