"""
Race cards, entries and the running of a race.

The true strength of each starter is a sum of interpretable parts (ability, age, form, surface and distance fit,
trainer and jockey, post position, pace scenario, layoff), and the race is decided by adding normal running-time
noise. Normal noise (Thurstone / Henery) rather than Gumbel noise is what makes second and third place behave like
real races, where plain Harville overrates the favorite for the minor placings.
"""

import numpy as np

STYLE_NOISE = 0.6


def _pmf_draw(rng, pmf):
    keys = list(pmf); probs = np.array([pmf[k] for k in keys], dtype=float)
    return keys[rng.choice(len(keys), p=probs / probs.sum())]


def build_card(world, track, day, rng):
    """
    Writes the race conditions for one card: race type, surface, distance, purse and target field size.

    Returns:
    - List of dicts, one per race.
    """
    R = world.params['races']
    tier = world.tracks.tier.values[track]
    n_races = int(_pmf_draw(rng, R['races_per_card_pmf']))
    card = []
    for number in range(1, n_races + 1):
        rtype = _pmf_draw(rng, R['race_type_share'])
        turf = rng.random() < R['turf_share'][rtype]
        distance = float(_pmf_draw(rng, R['turf_distances'] if turf else R['dirt_distances']))
        log_purse = rng.normal(R['log_purse_mean'][rtype] + R['purse_tier_slope'] * tier, R['log_purse_sd'][rtype])
        card.append(dict(track=track, day=day, number=number, race_type=rtype, surface='T' if turf else 'D',
                         distance=distance, purse=float(np.round(np.exp(log_purse), -2)),
                         field_target=int(_pmf_draw(rng, R['field_size_pmf']))))
    return card


def class_rating(world, race):
    """Maps race type and purse to the rating level the race is written for."""
    R = world.params['races']
    tier = world.tracks.tier.values[race['track']]
    offset = {'maiden': -0.25, 'claiming': 0.0, 'allowance': 0.35, 'stakes': 0.75}[race['race_type']]
    z = (np.log(race['purse']) - R['log_purse_mean'][race['race_type']] - R['purse_tier_slope'] * tier) / R['log_purse_sd'][race['race_type']]
    return world.params['running']['tier_ability_slope'] * tier + offset + 0.25 * z


def select_field(world, race, rng):
    """
    Chooses the starters: horses at the track that are ready to run, eligible for the conditions (maidens for maiden
    races, winners otherwise), weighted by how close their rating is to the race's class and by surface / distance fit.

    Returns:
    - Array of horse indices (possibly smaller than the target when few horses fit).
    """
    H = world.horses
    day, track = race['day'], race['track']
    pool = np.flatnonzero(H['active'] & (H['track'] == track) & (H['ready_day'] <= day))
    if race['race_type'] == 'maiden':
        pool = pool[H['wins'][pool] == 0]
    else:
        pool = pool[H['wins'][pool] > 0]
    if len(pool) < 4:
        return np.array([], dtype=int)
    rating = world.rating(pool, day)
    target = class_rating(world, race)
    sd = world.params['races']['class_match_sd']
    turf = race['surface'] == 'T'
    fit = -np.abs(H['best_distance'][pool] - race['distance']) * 0.35 + (H['turf_pref'][pool] * (1.5 if turf else -0.5))
    log_w = -0.5 * ((rating - target) / sd) ** 2 + fit
    w = np.exp(log_w - log_w.max())
    n = min(race['field_target'], len(pool))
    if w.sum() <= 0:
        return np.array([], dtype=int)
    return rng.choice(pool, n, replace=False, p=w / w.sum())


def assign_jockeys(world, starters, race, rng):
    """Better jockeys gravitate to better horses; 55% of horses keep their last rider when that rider is available."""
    J = world.jockeys[world.jockeys.track == race['track']]
    ids, skill = J.jockey.values, J.skill.values
    rating = world.rating(starters, race['day'])
    z = (rating - rating.mean()) / (rating.std() + 1e-9)
    chosen, used = np.empty(len(starters), dtype=int), set()
    for k in np.argsort(-z):
        last = world.horses['last_jockey'][starters[k]]
        if last in ids and last not in used and rng.random() < 0.55:
            chosen[k] = last
        else:
            score = 4.0 * skill * (1 + 0.5 * z[k]) + rng.gumbel(0, 1, len(ids))
            score[[i for i, j in enumerate(ids) if j in used]] = -np.inf
            chosen[k] = ids[int(np.argmax(score))]
        used.add(chosen[k])
    return chosen


def strength_components(world, starters, jockeys, race, rng):
    """
    The true strength of each starter, broken into parts.

    Returns:
    - Dict of arrays: ability, fit, trainer, trainer_hidden, jockey, jockey_hidden, post, pace, layoff, total.
    """
    H, Rn, T, J = world.horses, world.params['running'], world.trainers, world.jockeys
    day, n = race['day'], len(starters)
    turf = race['surface'] == 'T'
    # Step 1: Ability today (base + age curve + form) with extra uncertainty for first-time starters
    ability = world.rating(starters, day)
    debut = H['starts'][starters] == 0
    ability = ability + debut * rng.normal(0, Rn['debut_sd'], n)
    # Step 2: Fit to today's surface and distance
    fit = (turf * H['turf_pref'][starters] - 0.06 * (H['best_distance'][starters] - race['distance']) ** 2)
    # Step 3: Connections, including the parts of their skill the crowd will not see
    tr = H['trainer'][starters]
    gap = day - H['last_day'][starters]
    layoff = (gap >= 60) & ~debut
    trainer = T.skill.values[tr] + turf * T.turf_skill.values[tr] + layoff * T.layoff_skill.values[tr]
    trainer_hidden = T.hidden.values[tr]
    jockey = J.skill.values[jockeys]; jockey_hidden = J.hidden.values[jockeys]
    # Step 4: Post position bias and the pace scenario (lone speed helps, a speed duel hurts both)
    rel_post = (np.arange(n) - (n - 1) / 2) / max(n - 1, 1)
    bias = world.tracks['post_bias_turf' if turf else 'post_bias_dirt'].values[race['track']]
    post = -bias * rel_post * 2
    speed = H['early_speed'][starters]
    order = np.argsort(-speed)
    pace = np.zeros(n)
    if n >= 3:
        lead_gap = speed[order[0]] - speed[order[1]]
        if lead_gap > 0.8:
            pace[order[0]] += Rn['lone_speed_bonus']
        elif lead_gap < 0.3:
            pace[order[:2]] -= Rn['speed_duel_penalty']
    layoff_effect = layoff * Rn['layoff_effect']
    pace = pace + Rn.get('early_speed_value', 0.0) * (speed - speed.mean())
    total = ability + fit + trainer + trainer_hidden + jockey + jockey_hidden + post + pace + layoff_effect
    return dict(ability=ability, fit=fit, trainer=trainer, trainer_hidden=trainer_hidden, jockey=jockey,
                jockey_hidden=jockey_hidden, post=post, pace=pace, layoff=layoff_effect, total=total, debut=debut)


def performance_noise(rng, shape, noise_sd, gumbel_mix=0.0):
    """Running-time noise: a blend of normal and (standardized) Gumbel draws scaled to ``noise_sd``."""
    z = rng.standard_normal(shape)
    if gumbel_mix > 0:
        g = (rng.gumbel(0, 1, shape) - 0.5772) / 1.2825
        z = np.sqrt(1 - gumbel_mix ** 2) * z + gumbel_mix * g
    return noise_sd * z


def finish_probabilities(strength, noise_sd, rng, draws=4000, gumbel_mix=0.0):
    """
    Monte Carlo finish-order probabilities under the running-time noise model.

    Returns:
    - Dict with win (n,), top2 (n,), top3 (n,), exacta (n, n) probability arrays.
    """
    n = len(strength)
    perf = strength[None, :] + performance_noise(rng, (draws, n), noise_sd, gumbel_mix)
    order = np.argsort(-perf, axis=1)
    first, second, third = order[:, 0], order[:, 1], order[:, 2] if n >= 3 else order[:, 1]
    win = np.bincount(first, minlength=n) / draws
    top2 = win + np.bincount(second, minlength=n) / draws
    top3 = top2 + (np.bincount(third, minlength=n) / draws if n >= 3 else 0)
    exacta = np.zeros((n, n))
    np.add.at(exacta, (first, second), 1.0 / draws)
    return dict(win=win, top2=top2, top3=top3, exacta=exacta)


def run_race(world, race, starters, jockeys, comp, rng):
    """
    Runs the race: draws performances, then derives finish order, running positions, beaten lengths and figures.

    Returns:
    - Dict of per-starter arrays: performance, finish, pos_start, pos_1, pos_2, pos_stretch, beaten_lengths, speed_figure.
    """
    Rn = world.params['running']
    n = len(starters)
    perf = comp['total'] + performance_noise(rng, n, Rn['performance_noise_sd'], Rn.get('gumbel_mix', 0.0))
    finish = np.empty(n, dtype=int); finish[np.argsort(-perf)] = np.arange(1, n + 1)
    # Running positions: early speed decides the first call, the finish decides the last, calls blend between them
    early_score = world.horses['early_speed'][starters] + STYLE_NOISE * rng.standard_normal(n)
    rank = lambda v: np.argsort(np.argsort(-v)) + 1
    calls = {}
    for name, w in [('pos_start', 0.0), ('pos_1', 0.15), ('pos_2', 0.4), ('pos_stretch', 0.75)]:
        z_early = (early_score - early_score.mean()) / (early_score.std() + 1e-9)
        z_fin = (perf - perf.mean()) / (perf.std() + 1e-9)
        calls[name] = rank((1 - w) * z_early + w * z_fin + 0.2 * rng.standard_normal(n))
    lengths = np.clip((perf.max() - perf) * Rn['lengths_per_unit'] + rng.normal(0, 0.3, n), 0, None)
    lengths[finish == 1] = 0.0
    figure = Rn['figure_mean'] + Rn['figure_scale'] * perf + rng.normal(0, Rn['figure_noise_sd'], n)
    return dict(performance=perf, finish=finish, beaten_lengths=np.round(lengths * 4) / 4,
                speed_figure=np.round(figure), **calls)


def update_careers(world, race, starters, jockeys, result, rng):
    """After the race: form moves, the next start is scheduled, some horses are claimed, some retire."""
    H, C, Rn = world.horses, world.params['careers'], world.params['running']
    day, n = race['day'], len(starters)
    won = result['finish'] == 1
    H['starts'][starters] += 1
    H['wins'][starters] += won
    H['last_day'][starters] = day
    H['last_jockey'][starters] = jockeys
    H['form'][starters] = Rn['form_ar'] * H['form'][starters] + rng.normal(0, Rn['form_sd'], n)
    gap = np.maximum(C['min_gap_days'], np.exp(rng.normal(C['gap_log_mean'], C['gap_log_sd'], n)))
    lay = rng.random(n) < C['layoff_probability']
    gap = np.where(lay, rng.exponential(C['layoff_mean_days'], n) + 60, gap)
    H['ready_day'][starters] = day + gap.astype(int)
    # Claims move a horse to another barn at the same track (better claiming barns claim more)
    if race['race_type'] == 'claiming':
        claimed = rng.random(n) < C['claim_probability']
        if claimed.any():
            T = world.trainers[world.trainers.track == race['track']]
            w = np.exp(3 * T.claim_skill.values) * T.barn_weight.values
            H['trainer'][starters[claimed]] = rng.choice(T.trainer.values, claimed.sum(), p=w / w.sum())
        result['claimed'] = claimed
    else:
        result['claimed'] = np.zeros(n, dtype=bool)
    # Retirement grows with age
    age = world.age(starters, day)
    retire = rng.random(n) < C['retire_per_start'] + C['retire_age_slope'] * np.clip(age - 4, 0, None)
    H['active'][starters[retire]] = False
