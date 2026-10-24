"""
Backtesting: ledgers, settlement summaries and bankroll simulation.

A backtest in Lynk is three steps:
1. A strategy turns predictions into a bet ledger (one row per bet with the stake).
2. A market module settles the ledger against what actually happened (payout per $1, dilution, breakage, rebate).
3. ``summarize_ledger`` reports turnover, profit, ROI with a day-block confidence interval, ROI before rebates,
   drawdown and Sharpe, and ``bankroll_paths`` answers "what could happen to a real bankroll over N days".
"""

import numpy as np
import pandas as pd

from lynk.core.metrics import ratio_bootstrap, max_drawdown, annualized_sharpe


LEDGER_COLUMNS = ['event_id', 'date', 'pool', 'outcome', 'stake', 'payout_per_dollar', 'rebate_rate']


def settle_ledger(ledger):
    """
    Adds profit columns to a settled ledger.

    Parameters:
    - ledger: DataFrame with at least stake, payout_per_dollar (0 for losing bets) and rebate_rate.

    Returns:
    - Copy of the ledger with returned, rebate, profit and profit_pre_rebate columns.
    """
    out = ledger.copy()
    out['returned'] = out.stake * out.payout_per_dollar
    out['rebate'] = out.stake * out.rebate_rate
    out['profit'] = out.returned + out.rebate - out.stake
    out['profit_pre_rebate'] = out.returned - out.stake
    return out


def summarize_ledger(ledger, date_col='date', n_boot=2000, seed=0):
    """
    One-line summary of a settled ledger.

    Parameters:
    - ledger: Output of ``settle_ledger``.
    - date_col: Column used as the bootstrap block (bets on the same day are resampled together).

    Returns:
    - Dict with bets, events, turnover, profit, roi, roi_lo, roi_hi, roi_pre_rebate, max_drawdown and sharpe.
    """
    if ledger.empty:
        return dict(bets=0, events=0, turnover=0.0, profit=0.0, roi=np.nan, roi_lo=np.nan, roi_hi=np.nan,
                    roi_pre_rebate=np.nan, max_drawdown=0.0, sharpe=0.0)
    roi, lo, hi = ratio_bootstrap(ledger.profit.values, ledger.stake.values, ledger[date_col].values, n_boot, seed)
    daily = ledger.groupby(date_col).profit.sum().sort_index()
    return dict(bets=int(len(ledger)), events=int(ledger.event_id.nunique()), turnover=float(ledger.stake.sum()),
                profit=float(ledger.profit.sum()), roi=roi, roi_lo=lo, roi_hi=hi,
                roi_pre_rebate=float(ledger.profit_pre_rebate.sum() / ledger.stake.sum()),
                max_drawdown=max_drawdown(daily.cumsum().values), sharpe=annualized_sharpe(daily.values))


def grid_search(make_ledger, settings, bankroll, date_col='date'):
    """
    Runs a strategy over a grid of settings and summarizes each one.

    Parameters:
    - make_ledger: Function(**setting) returning a SETTLED ledger.
    - settings: List of dicts, e.g. [{'edge_cutoff': 0.05, 'kelly_multiplier': 0.25}, ...].
    - bankroll: Bankroll the strategy was sized against, used for log growth.

    Returns:
    - DataFrame with one row per setting: the setting, the ``summarize_ledger`` columns and log_growth, the sum over
      days of log(1 + daily profit / bankroll). Log growth is what Kelly maximizes, so it is the natural metric for a
      pre-registered selection rule.
    """
    rows = []
    for setting in settings:
        ledger = make_ledger(**setting)
        summary = summarize_ledger(ledger, date_col)
        if len(ledger):
            daily_profit = ledger.groupby(date_col).profit.sum()
            summary['log_growth'] = float(np.log1p((daily_profit / bankroll).clip(lower=-0.999999)).sum())
        else:
            summary['log_growth'] = 0.0
        rows.append({**setting, **summary})
    return pd.DataFrame(rows)


def bankroll_paths(daily_stake, daily_profit, horizon_days, n_paths=4000, start_bankroll=1000.0,
                   target_turnover=None, seed=0):
    """
    Simulates bankroll outcomes by resampling whole days of a backtest.

    Each path draws ``horizon_days`` days at random (with replacement) from the backtest, so streaks and bad days
    keep their real shape. Optionally stops betting once ``target_turnover`` has been wagered.

    Parameters:
    - daily_stake, daily_profit: Arrays, one value per calendar day of the backtest (zeros on days with no bets).
    - horizon_days: Length of each simulated path.
    - n_paths: Number of paths.
    - start_bankroll: Starting bankroll in dollars.
    - target_turnover: Optional turnover goal in dollars.

    Returns:
    - DataFrame, one row per path: turnover, profit, min_bankroll, reached_target.
    """
    stake = np.asarray(daily_stake, dtype=float)
    profit = np.asarray(daily_profit, dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_paths):
        days = rng.integers(0, len(stake), horizon_days)
        cum_stake, cum_profit = np.cumsum(stake[days]), np.cumsum(profit[days])
        last = horizon_days - 1
        if target_turnover is not None:
            last = min(int(np.searchsorted(cum_stake, target_turnover)), horizon_days - 1)
        rows.append(dict(turnover=cum_stake[last], profit=cum_profit[last],
                         min_bankroll=start_bankroll + min(0.0, cum_profit[:last + 1].min()),
                         reached_target=bool(target_turnover is None or cum_stake[last] >= target_turnover)))
    return pd.DataFrame(rows)
