"""
A synthetic racing world for examples, tests and teaching.

Real racing data (charts, tote feeds) is licensed and cannot ship with the repo, so this module simulates a racing
circuit with the same structure and the same traps:

- Horses have a latent ability that drifts over time; trainers and jockeys add their own effects.
- Results are drawn from a Plackett-Luce (Gumbel) model, so finish orders behave like Harville races.
- The betting crowd sees ability well but systematically underrates some trainers, which is exactly the kind of
  mispricing the Lynk entity ratings are built to find.
- Late money arrives after the decision snapshot and is smarter than the early money, so the final odds are sharper
  than the odds you bet into.

Every number is generated from a seed, so examples are reproducible.
"""

import numpy as np
import pandas as pd

from lynk.core.parimutuel import apply_breakage


def make_races(n_days=240, races_per_day=8, field_range=(6, 11), n_horses=1500, n_trainers=60, n_jockeys=40,
               tracks=('AAA', 'BBB', 'CCC'), takeout=0.17, crowd_skill=0.85, trainer_mispricing=0.15, late_money_share=0.4,
               start_date='2023-01-01', seed=7):
    """
    Simulates a racing circuit.

    Parameters:
    - n_days: Racing days.
    - races_per_day: Races per day.
    - field_range: Min and max field size.
    - n_horses, n_trainers, n_jockeys: Population sizes.
    - tracks: Track codes (each race is run at one of them).
    - takeout: Win, place and show takeout.
    - crowd_skill: How much of the true strength the early crowd prices (1.0 = perfectly informed).
    - trainer_mispricing: Standard deviation of the trainer effect the crowd ignores (the planted edge).
    - late_money_share: Share of the final pool bet after the decision snapshot.
    - start_date: First racing day.
    - seed: Random seed.

    Returns:
    - Starter-level DataFrame following ``lynk.horse_racing.schema`` (pre-race, market snapshot and post-race
      columns). Market columns describe the pools at the decision time; payoffs come from the final pools.
    """
    rng = np.random.default_rng(seed)
    # Step 1: The population and its hidden qualities
    ability = rng.normal(0, 0.55, n_horses)
    horse_trainer = rng.integers(0, n_trainers, n_horses)
    trainer_skill = rng.normal(0, 0.18, n_trainers)
    trainer_mispriced = rng.normal(0, trainer_mispricing, n_trainers)          # the part of trainer skill the crowd ignores
    jockey_skill = rng.normal(0, 0.12, n_jockeys)
    surfaces = np.array(['D', 'T'])
    turf_pref = rng.normal(0, 0.25, n_horses)
    rows = []
    dates = pd.date_range(start_date, periods=n_days, freq='D')
    race_no = 0
    for day in dates:
        ability += rng.normal(0, 0.02, n_horses)                  # form drifts slowly
        for r in range(races_per_day):
            race_no += 1
            track = tracks[r % len(tracks)]
            surface = surfaces[rng.integers(0, 2)]
            distance = float(rng.choice([6.0, 6.5, 7.0, 8.0, 8.5, 9.0]))
            field = int(rng.integers(field_range[0], field_range[1] + 1))
            horses = rng.choice(n_horses, field, replace=False)
            trainers = horse_trainer[horses]
            jockeys = rng.integers(0, n_jockeys, field)
            # Step 2: True strength and the result (Plackett-Luce via Gumbel noise)
            true_strength = (ability[horses] + (surface == 'T') * turf_pref[horses] + trainer_skill[trainers]
                             + trainer_mispriced[trainers] + jockey_skill[jockeys] + rng.normal(0, 0.15, field))
            performance = true_strength + rng.gumbel(0, 1, field)
            finish = np.empty(field, int)
            finish[np.argsort(-performance)] = np.arange(1, field + 1)
            p_true = np.exp(true_strength - true_strength.max()); p_true /= p_true.sum()
            # Step 3: The crowd at the decision snapshot, then smarter late money
            seen = crowd_skill * (true_strength - trainer_mispriced[trainers]) + rng.normal(0, 0.25, field)
            p_crowd = np.exp(seen - seen.max()); p_crowd /= p_crowd.sum()
            p_late = 0.5 * p_crowd + 0.5 * p_true
            pool_final = float(np.exp(rng.normal(np.log(60000), 0.6)))
            pool_seen = pool_final * (1 - late_money_share)
            money_seen = p_crowd * pool_seen
            money_final = money_seen + p_late * (pool_final - pool_seen)
            share_final = money_final / money_final.sum()
            win_pay = apply_breakage((1 - takeout) / share_final)
            # Place and show pools use the same shares (simplification), sized like a typical US card
            place_final, show_final = 0.35 * pool_final * share_final, 0.25 * pool_final * share_final
            top2 = np.flatnonzero(finish <= 2); top3 = np.flatnonzero(finish <= 3)
            place_pay = np.zeros(field); show_pay = np.zeros(field)
            for i in top2:
                other = place_final[top2].sum() - place_final[i]
                place_pay[i] = apply_breakage(1 + max((1 - takeout) * place_final.sum() - place_final[i] - other, 0) / 2 / place_final[i], minimum=1.05)
            for i in top3:
                other = show_final[top3].sum() - show_final[i]
                show_pay[i] = apply_breakage(1 + max((1 - takeout) * show_final.sum() - show_final[i] - other, 0) / 3 / show_final[i], minimum=1.05)
            figures = 80 + 8 * performance + rng.normal(0, 3, field)
            for k in range(field):
                rows.append(dict(
                    race_id=f'{track}_{day:%Y%m%d}_{r + 1:02d}', date=day, track=track, surface=surface, distance=distance,
                    race_class=int(rng.integers(1, 6)), horse=f'H{horses[k]:05d}', trainer=f'T{trainers[k]:03d}',
                    jockey=f'J{jockeys[k]:03d}', post=k + 1, field_size=field,
                    morning_line=float(np.round(np.clip(1 / p_crowd[k] - 1, 1, 50) * 2) / 2),
                    win_money=money_seen[k], win_pool=pool_seen, place_money=0.35 * money_seen[k], place_pool=0.35 * pool_seen,
                    show_money=0.25 * money_seen[k], show_pool=0.25 * pool_seen, takeout=takeout,
                    finish=int(finish[k]), won=int(finish[k] == 1), speed_figure=float(figures[k]),
                    final_odds=float(1 / share_final[k] * (1 - takeout) - 1),
                    win_payoff=float(win_pay[k]) if finish[k] == 1 else 0.0,
                    place_payoff=float(place_pay[k]), show_payoff=float(show_pay[k]),
                    win_money_final=money_final[k], win_pool_final=pool_final, true_probability=p_true[k]))
    return pd.DataFrame(rows)
