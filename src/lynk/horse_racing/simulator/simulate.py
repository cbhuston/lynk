"""
Runs the simulated circuit day by day and returns tidy tables.

    from lynk.horse_racing.simulator import simulate
    result = simulate(n_days=120, seed=1)
    result.races, result.starters, result.wps_snapshots, result.payoffs, ...

Tables (all keyed by race_id; post positions are 1-based):
- races            conditions, purse, field size, scheduled post and the actual off delay
- starters         a result chart: connections, morning line, final odds, finish, calls, beaten lengths, figure, payoffs
- truth            what the model is trying to learn: true win / top-2 / top-3 probabilities and strength components
- wps_snapshots    win / place / show money per runner at every offset before the close and in the final cycle
- pool_totals      every pool's total at every offset (exotic pools scaled by the same arrival curve)
- exacta_probables exacta payouts per $2 at 5 minutes, 1 minute and the post-time cycle
- payoffs          official payoffs per $2 for win, place, show, exacta, trifecta, superfecta
- horizontals      daily double and Pick-N results (pool, winning sequence, payout, carryover)
- dd_willpays      daily double will-pays for every second-leg runner
- horses, trainers, jockeys, owners, sires   the population, including hidden qualities (for research only)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lynk.horse_racing.simulator.params import load_params
from lynk.horse_racing.simulator.population import World
from lynk.horse_racing.simulator import racing, tote, horizontals

_WEEKDAYS = [[3, 4, 5, 6, 2], [3, 4, 5, 6, 0], [4, 5, 6, 3], [4, 5, 6, 0], [5, 6, 4], [5, 6, 3]]


@dataclass
class SimulationResult:
    races: pd.DataFrame
    starters: pd.DataFrame
    truth: pd.DataFrame
    wps_snapshots: pd.DataFrame
    pool_totals: pd.DataFrame
    exacta_probables: pd.DataFrame
    payoffs: pd.DataFrame
    horizontals: pd.DataFrame
    dd_willpays: pd.DataFrame
    horses: pd.DataFrame
    trainers: pd.DataFrame
    jockeys: pd.DataFrame
    owners: pd.DataFrame
    sires: pd.DataFrame
    params: dict


def simulate(n_days=120, params=None, seed=0, params_path=None, progress=False):
    """
    Simulates ``n_days`` of racing on the circuit.

    Parameters:
    - n_days: Calendar days to simulate (tracks race on their own weekly schedules).
    - params: Parameter dict (defaults to ``load_params(params_path)``).
    - seed: Random seed.
    - params_path: Optional JSON with calibrated parameters.
    - progress: Print a line every 30 days.

    Returns:
    - ``SimulationResult`` with the tables described in the module docstring.
    """
    params = params or load_params(params_path)
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(params['world']['start_date'])
    start_day = int((start - pd.Timestamp('1970-01-01')).days)
    world = World.build(params, seed=seed, start_day=start_day)
    T = params['tote']
    takeout = T['takeout']
    rows = {k: [] for k in ['races', 'starters', 'truth', 'snaps', 'totals', 'probables', 'payoffs', 'horiz', 'willpays']}
    carry = {t: {} for t in range(len(world.tracks))}

    for d in range(n_days):
        day = start_day + d
        date = start + pd.Timedelta(days=d)
        if d % 7 == 0 and d > 0:   # Step 1: new horses arrive every week
            for t in range(len(world.tracks)):
                world.add_horses(t, params['careers']['new_horses_per_track_per_week'], day)
        for t in range(len(world.tracks)):
            schedule = _WEEKDAYS[t % len(_WEEKDAYS)][:world.tracks.days_per_week.values[t]]
            if date.weekday() not in schedule:
                continue
            card_states = []
            post_time = date + pd.Timedelta(hours=13)
            for race in racing.build_card(world, t, day, rng):
                # Step 2: Entries, riders and the hidden strength of every starter
                starters = racing.select_field(world, race, rng)
                if len(starters) < 4:
                    continue
                world.horses['ready_day'][starters] = day + 1
                jockeys = racing.assign_jockeys(world, starters, race, rng)
                comp = racing.strength_components(world, starters, jockeys, race, rng)
                probs = racing.finish_probabilities(comp['total'], params['running']['performance_noise_sd'], rng,
                                                    gumbel_mix=params['running'].get('gumbel_mix', 0.0))
                result = racing.run_race(world, race, starters, jockeys, comp, rng)
                # Step 3: The betting: crowd opinions, pool sizes, money cycle by cycle, payoffs
                beliefs = tote.crowd_beliefs(world, comp, rng)
                pools = tote.pool_sizes(world, race, len(starters), rng)
                money = tote.simulate_win_place_show(world, race, beliefs, pools, rng)
                final = dict(win=money['win'][-1], place=money['place'][-1], show=money['show'][-1])
                pays = tote.wps_payoffs(final, result['finish'], takeout, pools)
                probables, exotic_pays = tote.exotic_payoffs(world, money, result['finish'], pools, rng)
                trainers_of_record = world.horses['trainer'][starters].copy()   # before any claim in this race
                racing.update_careers(world, race, starters, jockeys, result, rng)
                race_id = f"{world.tracks.track.values[t]}_{date:%Y%m%d}_{race['number']:02d}"
                delay = float(rng.exponential(T['post_delay_mean_s']))
                post_time = post_time + pd.Timedelta(minutes=int(rng.integers(24, 33)))
                rows['races'].append(dict(race_id=race_id, track=world.tracks.track.values[t], date=date, race_number=race['number'],
                                          race_type=race['race_type'], surface=race['surface'], distance=race['distance'],
                                          purse=race['purse'], field_size=len(starters), post_time=post_time, off_delay_s=round(delay, 1),
                                          **{f'pool_{k}': round(v, 0) for k, v in pools.items()}))
                final_odds = np.round(((1 - takeout['win']) * final['win'].sum() / final['win'] - 1) * 20) / 20
                H = world.horses
                for k, h in enumerate(starters):
                    rows['starters'].append(dict(
                        race_id=race_id, date=date, track=world.tracks.track.values[t], post=k + 1, horse=int(h), horse_name=H['name'][h],
                        trainer=int(trainers_of_record[k]), jockey=int(jockeys[k]),
                        owner=int(H['owner'][h]), sire=int(H['sire'][h]), age=int(world.age(np.array([h]), day)[0]),
                        sex='F' if H['female'][h] else 'M', morning_line=float(beliefs['ml_odds'][k]), final_odds=float(final_odds[k]),
                        finish=int(result['finish'][k]), pos_start=int(result['pos_start'][k]), pos_call1=int(result['pos_1'][k]),
                        pos_call2=int(result['pos_2'][k]), pos_stretch=int(result['pos_stretch'][k]),
                        beaten_lengths=float(result['beaten_lengths'][k]), speed_figure=float(result['speed_figure'][k]),
                        win_payoff=float(pays['win'][k]), place_payoff=float(pays['place'][k]), show_payoff=float(pays['show'][k]),
                        claimed=bool(result['claimed'][k])))
                    rows['truth'].append(dict(race_id=race_id, post=k + 1, p_win=probs['win'][k], p_top2=probs['top2'][k],
                                              p_top3=probs['top3'][k], p_public=beliefs['p_public'][k], p_informed=beliefs['p_informed'][k],
                                              **{f's_{c}': float(comp[c][k]) for c in ['ability', 'fit', 'trainer', 'trainer_hidden', 'jockey',
                                                                                   'jockey_hidden', 'post', 'pace', 'layoff', 'total']}))
                for i, key in enumerate(money['keys']):
                    for k in range(len(starters)):
                        rows['snaps'].append((race_id, key, k + 1, money['win'][i, k], money['place'][i, k], money['show'][i, k]))
                    for pool, total in pools.items():
                        rows['totals'].append((race_id, key, pool, total * money['arrival'][i]))
                for key, a, b, pay in probables:
                    rows['probables'].append((race_id, key, a + 1, b + 1, pay))
                for pool in ['win', 'place', 'show']:
                    for k in np.flatnonzero(pays[pool] > 0):
                        rows['payoffs'].append((race_id, pool, str(k + 1), float(pays[pool][k])))
                for pool, (combo, pay) in exotic_pays.items():
                    rows['payoffs'].append((race_id, pool, '-'.join(str(c + 1) for c in combo), pay))
                card_states.append(dict(race_id=race_id, money=money, finish=result['finish'], pools=pools))
            # Step 4: Horizontal pools across the card
            dd, wp = horizontals.daily_doubles(world, card_states, rng)
            rows['horiz'] += dd + horizontals.pick_n(world, card_states, rng, carry[t])
            rows['willpays'] += wp
        if progress and d % 30 == 0:
            print(f'day {d}: {len(rows["races"])} races', flush=True)

    return SimulationResult(
        races=pd.DataFrame(rows['races']), starters=pd.DataFrame(rows['starters']),
        truth=pd.DataFrame(rows['truth']),
        wps_snapshots=pd.DataFrame(rows['snaps'], columns=['race_id', 'offset', 'post', 'win_money', 'place_money', 'show_money']),
        pool_totals=pd.DataFrame(rows['totals'], columns=['race_id', 'offset', 'pool', 'total']),
        exacta_probables=pd.DataFrame(rows['probables'], columns=['race_id', 'offset', 'first', 'second', 'payout']),
        payoffs=pd.DataFrame(rows['payoffs'], columns=['race_id', 'pool', 'combination', 'payout']),
        horizontals=pd.DataFrame(rows['horiz']), dd_willpays=pd.DataFrame(rows['willpays']),
        horses=pd.DataFrame({k: v for k, v in world.horses.items()}), trainers=world.trainers, jockeys=world.jockeys,
        owners=world.owners, sires=world.sires, params=params)


def to_schema_frame(result, decision_offset='0'):
    """
    Converts a simulation into the starter-level frame used by ``lynk.horse_racing`` features, models and bets, with
    the market snapshot taken at ``decision_offset`` (seconds before the close, as a string, e.g. '0' or '60').
    """
    snap = result.wps_snapshots[result.wps_snapshots.offset == decision_offset]
    totals = snap.groupby('race_id')[['win_money', 'place_money', 'show_money']].sum().rename(
        columns={'win_money': 'win_pool', 'place_money': 'place_pool', 'show_money': 'show_pool'})
    final = result.wps_snapshots[result.wps_snapshots.offset == 'final'][['race_id', 'post', 'win_money']].rename(columns={'win_money': 'win_money_final'})
    out = (result.starters.merge(result.races[['race_id', 'surface', 'distance', 'race_type', 'field_size', 'purse']], on='race_id')
           .merge(snap[['race_id', 'post', 'win_money', 'place_money', 'show_money']], on=['race_id', 'post'])
           .merge(totals, on='race_id').merge(final, on=['race_id', 'post'])
           .merge(result.truth[['race_id', 'post', 'p_win']].rename(columns={'p_win': 'true_probability'}), on=['race_id', 'post']))
    out['won'] = (out.finish == 1).astype(int)
    out['takeout'] = result.params['tote']['takeout']['win']
    out['win_payoff'] = out.win_payoff / 2.0           # schema uses payout per $1
    out['place_payoff'] = out.place_payoff / 2.0
    out['show_payoff'] = out.show_payoff / 2.0
    out['race_class'] = out.race_type
    out['field_size'] = out.field_size.astype(int)
    out['horse'] = 'H' + out.horse.astype(str); out['trainer'] = 'T' + out.trainer.astype(str); out['jockey'] = 'J' + out.jockey.astype(str)
    return out.sort_values(['date', 'race_id', 'post']).reset_index(drop=True)
