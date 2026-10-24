"""
Building, sizing and settling horse racing bets.

Two strategies are provided, matching the two levels of the Lynk research:

- ``win_strategy``: single-pool win betting. For every race it prices each horse at the FORECAST final odds,
  keeps horses whose expected value clears an edge cutoff, sizes the bets jointly with the pari-mutuel portfolio
  solver (Kelly with own-money impact), applies fractional Kelly and the 2/3 max-EV cap, and settles at the official
  payoff adjusted for dilution.
- ``multi_pool_race_bets``: the same solver across win, place, show and exacta for one race, using discounted
  Harville scenarios. Use it only for pools whose expected values are calibrated out of sample (see the bet sizing
  report: an uncalibrated pool attracts most of the money in a joint solve).
"""

import numpy as np
import pandas as pd

from lynk.core.parimutuel import max_ev_bet, diluted_actual_payout
from lynk.core.portfolio import Pool, Scenario, optimize_portfolio
from lynk.horse_racing.finish_order import finish_order_scenarios


def forecast_final_money(win_money, win_pool, late_money_share=0.4, smart_share=None, p_model=None):
    """
    Forecasts each horse's FINAL money from the snapshot at decision time.

    The simplest forecast scales today's money up by the expected late-money share. If a model probability is given,
    the late money is split between the crowd's current view and the model's view (``smart_share`` of it follows the
    model). In the research this step is a learned late-money model; this function is the transparent default.

    Returns:
    - Tuple (final_money_per_horse, final_pool).
    """
    win_money = np.asarray(win_money, dtype=float)
    pool_final = float(win_pool) / (1.0 - late_money_share)
    late = pool_final - float(win_pool)
    crowd = win_money / win_money.sum()
    if p_model is None or smart_share is None:
        return win_money + late * crowd, pool_final
    p_model = np.asarray(p_model, dtype=float)
    return win_money + late * ((1 - smart_share) * crowd + smart_share * p_model), pool_final


def win_strategy(frame, prob_col='p_model', bankroll_per_race=10000.0, edge_cutoff=0.05, kelly_multiplier=0.5,
                 late_money_share=0.4, smart_share=0.5, rebate=0.0, min_stake=2.0, max_ev_share=2.0 / 3.0):
    """
    Win-pool strategy with pricing at forecast final odds and impact-aware Kelly sizing. Returns a settled ledger.

    Parameters:
    - frame: Starter-level DataFrame with race_id, date, win_money, win_pool, takeout, won, win_payoff and ``prob_col``.
    - prob_col: Model win probability column (sums to one per race).
    - bankroll_per_race: Dollars the solver may use per race.
    - edge_cutoff: Minimum expected value per $1 at a $0 stake to consider a horse (0.05 = 5% edge).
    - kelly_multiplier: Fractional Kelly.
    - late_money_share, smart_share: Final-money forecast settings (see ``forecast_final_money``).
    - rebate: Rebate rate paid on every dollar wagered.
    - min_stake: Minimum ticket in dollars.
    - max_ev_share: Cap each stake at this share of its max-EV stake.

    Returns:
    - Ledger DataFrame with the ``lynk.core.backtest`` columns (event_id, date, pool, outcome, stake,
      payout_per_dollar, rebate_rate) ready for ``settle_ledger``.
    """
    rows = []
    for race_id, race in frame.groupby('race_id', sort=False):
        p = race[prob_col].values
        money, pool = forecast_final_money(race.win_money.values, race.win_pool.iloc[0], late_money_share, smart_share, p)
        takeout = float(race.takeout.iloc[0])
        # Step 1: Price at forecast final odds and keep horses that clear the edge cutoff
        ev = p * (1 - takeout) * pool / money + rebate - 1
        keep = ev > edge_cutoff
        if not keep.any():
            continue
        # Step 2: Size jointly with the pari-mutuel solver (only the kept horses may be bet)
        labels = list(race.post.values)
        win_pool = Pool('win', takeout, dict(zip(labels, money)), rebate=rebate, total=pool)
        scenarios = [Scenario(float(pi), {'win': (lab,)}) for lab, pi in zip(labels, p)]
        allowed = {('win', lab) for lab, k in zip(labels, keep) if k}
        bets = optimize_portfolio([win_pool], scenarios, bankroll_per_race, kelly_multiplier, allowed=allowed,
                                  min_stake=min_stake)
        if bets.empty:
            continue
        # Step 3: Cap each stake at 2/3 of its max-EV stake, then settle at the diluted official payoff
        by_post = race.set_index('post')
        for bet in bets.itertuples():
            i = labels.index(bet.outcome)
            cap = max_ev_share * max_ev_bet(p[i], pool, money[i], takeout, rebate)
            stake = float(np.floor(min(bet.stake, cap)))
            if stake < min_stake:
                continue
            horse = by_post.loc[bet.outcome]
            payout = diluted_actual_payout(horse.win_payoff, money[i], stake) if horse.won == 1 else 0.0
            rows.append(dict(event_id=race_id, date=horse.date, pool='win', outcome=bet.outcome, stake=stake,
                             payout_per_dollar=float(payout), rebate_rate=rebate, ev_at_zero=float(ev[i])))
    return pd.DataFrame(rows, columns=['event_id', 'date', 'pool', 'outcome', 'stake', 'payout_per_dollar',
                                       'rebate_rate', 'ev_at_zero'])


def multi_pool_race_bets(race, prob_col='p_model', bankroll=10000.0, pools=('win', 'place', 'show', 'exacta'),
                         kelly_multiplier=0.5, gamma_second=0.80, gamma_third=0.60, exacta_pool_share=0.5,
                         min_stake=2.0):
    """
    Solves one race across several pools at once with discounted-Harville finish-order scenarios.

    Parameters:
    - race: Starter rows of ONE race with post, win_money, win_pool, place_money, place_pool, show_money, show_pool,
            takeout and ``prob_col``.
    - pools: Which pools to include.
    - exacta_pool_share: Exacta pool size relative to the win pool, used to build exacta money from crowd
                         Harville probabilities when no exacta probables are available.

    Returns:
    - DataFrame of bets (pool, outcome, stake).
    """
    race = race.sort_values('post')
    labels = list(race.post.values)
    p = race[prob_col].values
    takeout = float(race.takeout.iloc[0])
    crowd = race.win_money.values / race.win_money.sum()
    pool_objects = []
    if 'win' in pools:
        pool_objects.append(Pool('win', takeout, dict(zip(labels, race.win_money.values))))
    if 'place' in pools:
        pool_objects.append(Pool('place', takeout, dict(zip(labels, race.place_money.values))))
    if 'show' in pools:
        pool_objects.append(Pool('show', takeout, dict(zip(labels, race.show_money.values))))
    if 'exacta' in pools:
        crowd_pairs = finish_order_scenarios(crowd, depth=2, gamma_second=1.0)
        total = exacta_pool_share * float(race.win_pool.iloc[0])
        pool_objects.append(Pool('exacta', takeout + 0.04,
                                 {(labels[i], labels[j]): max(total * q, 1.0) for (i, j), q in crowd_pairs}))
    # Scenarios: every top-3 order, carrying the winners of every pool
    scenarios = []
    for (i, j, k), prob in finish_order_scenarios(p, depth=3, gamma_second=gamma_second, gamma_third=gamma_third):
        scenarios.append(Scenario(prob, {'win': (labels[i],), 'place': (labels[i], labels[j]),
                                         'show': (labels[i], labels[j], labels[k]), 'exacta': ((labels[i], labels[j]),)}))
    return optimize_portfolio(pool_objects, scenarios, bankroll, kelly_multiplier, min_stake=min_stake)
