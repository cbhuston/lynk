"""
Handicapping features that are leakage-safe by construction.

Three families carry most of the signal in a fundamental (no odds) horse racing model:
1. Horse history: the horse's own recent figures and finishes, always lagged so today's race is never included.
2. Connections: trainer / jockey / owner / sire records over a trailing window that ends the day BEFORE the race.
3. Field context: every number compared with the rest of today's field (rank, gap to the best, z-score). A 90 speed
   figure means little until you know whether the next best horse ran 70 or 95.
"""

import numpy as np
import pandas as pd


def add_history_features(frame, value_cols, horse_col='horse', date_col='date', lags=(1, 2, 3), windows=(3, 5)):
    """
    Adds lagged and rolling summaries of each horse's previous starts.

    For each column in ``value_cols``: last 1..k values, the mean and best of the last n starts, plus days since the
    last start and the number of previous starts. Only strictly earlier races are used.

    Parameters:
    - frame: Starter-level DataFrame sorted by date.
    - value_cols: Post-race columns to summarize, e.g. ['speed_figure', 'finish_pct'].
    - horse_col, date_col: Identifier and date columns.
    - lags: Which previous starts to copy as features.
    - windows: Rolling window sizes (in starts).

    Returns:
    - Copy of the frame with new columns named like 'speed_figure_lag1', 'speed_figure_mean3', 'speed_figure_best5',
      'days_since_last', 'prior_starts'.
    """
    out = frame.sort_values([date_col, 'race_id']).copy()
    by_horse = out.groupby(horse_col, sort=False)
    # Step 1: Lags of each value (shift(1) is the previous start, never today's)
    for col in value_cols:
        for k in lags:
            out[f'{col}_lag{k}'] = by_horse[col].shift(k)
        previous = by_horse[col].shift(1)
        for n in windows:
            rolled = previous.groupby(out[horse_col]).rolling(n, min_periods=1)
            out[f'{col}_mean{n}'] = rolled.mean().reset_index(level=0, drop=True)
            out[f'{col}_best{n}'] = rolled.max().reset_index(level=0, drop=True)
    # Step 2: Freshness and experience
    out['days_since_last'] = (out[date_col] - by_horse[date_col].shift(1)).dt.days
    out['prior_starts'] = by_horse.cumcount()
    return out.sort_index()


def add_relative_features(frame, cols, event_col='race_id', higher_is_better=True):
    """
    Compares each value with the rest of the field in the same race.

    Adds '<col>_rank' (1 = best in the race), '<col>_gap_best' (difference to the best value) and '<col>_z'
    (standardized within the race). Missing values stay missing and do not count toward the field.
    """
    out = frame.copy()
    groups = out.groupby(event_col)
    for col in cols:
        out[f'{col}_rank'] = groups[col].rank(ascending=not higher_is_better, method='min')
        best = groups[col].transform('max' if higher_is_better else 'min')
        out[f'{col}_gap_best'] = (out[col] - best) * (1 if higher_is_better else -1)
        mean, sd = groups[col].transform('mean'), groups[col].transform('std')
        out[f'{col}_z'] = (out[col] - mean) / sd.replace(0, np.nan)
    return out


def add_connection_form(frame, entity_cols, outcome_col='won', expected_col=None, date_col='date',
                        window_days=365, name=None):
    """
    Trailing-window record of a connection (trainer, jockey, trainer at a track, ...) ending the day before the race.

    Adds '<name>_starts', '<name>_wins' and '<name>_win_rate'. When ``expected_col`` is given (for example a
    probability from a baseline model), also adds '<name>_ae', actual over expected wins with +1 smoothing: a value of
    1.2 means the connection won 20% more often than the baseline expected.

    Parameters:
    - frame: Starter-level DataFrame.
    - entity_cols: Column or list of columns identifying the connection, e.g. 'trainer' or ['trainer', 'track'].
    - outcome_col: 1 for a win.
    - expected_col: Optional baseline probability column.
    - window_days: Length of the trailing window in days.
    - name: Prefix for the new columns (defaults to the joined entity column names).
    """
    entity_cols = [entity_cols] if isinstance(entity_cols, str) else list(entity_cols)
    name = name or '_'.join(entity_cols)
    out = frame.copy()
    key = out[entity_cols].astype(str).agg('|'.join, axis=1)
    # Step 1: Collapse to one row per entity per day (so same-day results never leak into each other)
    daily = pd.DataFrame({'key': key, 'date': pd.to_datetime(out[date_col]), 'n': 1.0,
                          'w': out[outcome_col].astype(float),
                          'e': out[expected_col].astype(float) if expected_col else 0.0})
    daily = daily.groupby(['key', 'date'], sort=True)[['n', 'w', 'e']].sum().reset_index()
    # Step 2: Rolling sums over the window, closed on the left so the race day itself is excluded
    rolled = (daily.set_index('date').groupby('key')[['n', 'w', 'e']]
              .rolling(f'{window_days}D', closed='left').sum().reset_index())
    rolled = rolled.fillna(0.0)
    # Step 3: Attach to every starter of that entity on that day
    merged = pd.DataFrame({'key': key, 'date': pd.to_datetime(out[date_col])}).merge(rolled, on=['key', 'date'], how='left')
    out[f'{name}_starts'] = merged.n.values
    out[f'{name}_wins'] = merged.w.values
    out[f'{name}_win_rate'] = (merged.w / merged.n.replace(0, np.nan)).values
    if expected_col:
        out[f'{name}_ae'] = ((merged.w + 1.0) / (merged.e + 1.0)).values
    return out
