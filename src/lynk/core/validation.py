"""
Leakage-safe validation.

The rules every Lynk result follows:
1. Split by time, never by shuffled rows. Train on the past, choose settings on a validation period, and score the
   test period once.
2. Features may only use information that existed before the decision time.
3. Strategy settings are chosen by a rule written down before looking at the test period.
"""

import numpy as np
import pandas as pd


def date_split(dates, valid_start, test_start, test_end=None):
    """
    Labels every row 'train', 'valid' or 'test' by date.

    Parameters:
    - dates: Array-like of dates (anything pandas can parse).
    - valid_start: First date of the validation period.
    - test_start: First date of the test period.
    - test_end: Optional first date AFTER the test period (rows on or after it are labeled 'holdout').

    Returns:
    - numpy array of labels.
    """
    d = pd.to_datetime(pd.Series(dates)).values
    labels = np.where(d < np.datetime64(pd.Timestamp(valid_start)), 'train',
                      np.where(d < np.datetime64(pd.Timestamp(test_start)), 'valid', 'test'))
    if test_end is not None:
        labels = np.where(d >= np.datetime64(pd.Timestamp(test_end)), 'holdout', labels)
    return labels


def walk_forward_years(dates, first_year, last_year, train_start=None, early_stop_months=6):
    """
    Yields walk-forward masks: for each year Y, fit on [train_start, Y-1 minus early-stop window), early-stop on the
    last ``early_stop_months`` before Y, and predict year Y. Every prediction comes from a model that never saw it.

    Parameters:
    - dates: Array-like of row dates.
    - first_year, last_year: Years to predict.
    - train_start: Optional first training date.
    - early_stop_months: Length of the early-stopping window right before each predicted year.

    Yields:
    - Tuples (year, fit_mask, early_stop_mask, predict_mask).
    """
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    start = pd.Timestamp(train_start) if train_start else d.min()
    for year in range(first_year, last_year + 1):
        year_start = pd.Timestamp(year=year, month=1, day=1)
        es_start = year_start - pd.DateOffset(months=early_stop_months)
        fit = ((d >= start) & (d < es_start)).values
        early_stop = ((d >= es_start) & (d < year_start)).values
        predict = (d.dt.year == year).values
        yield year, fit, early_stop, predict


def assert_features_precede(frame, decision_time_col, feature_time_cols):
    """
    Raises an error if any feature was observed at or after the decision time of its row.

    Use it as a unit test on feature pipelines: store the timestamp each feature was last updated next to the
    feature and assert they all come strictly before the moment the bet is placed.

    Parameters:
    - frame: DataFrame with the decision time and one timestamp column per feature (or feature family).
    - decision_time_col: Column with the time the decision is made.
    - feature_time_cols: List of columns with the time each feature's information became available.
    """
    decision = pd.to_datetime(frame[decision_time_col])
    for col in feature_time_cols:
        seen = pd.to_datetime(frame[col])
        late = (seen >= decision) & seen.notna()
        if late.any():
            raise ValueError(f"{int(late.sum())} rows of '{col}' were observed at or after the decision time")


def select_preregistered(grid, metric='log_growth', min_bets=300, bets_col='bets'):
    """
    Applies a pre-registered selection rule to a grid of validation results.

    Rule: among settings with at least ``min_bets`` bets, pick the one with the highest ``metric``. The strategy is
    only active if that metric is positive. Writing the rule down before scoring the test period is what keeps the
    test result honest.

    Parameters:
    - grid: DataFrame, one row per strategy setting, scored on the VALIDATION period.
    - metric: Column to maximize.
    - min_bets: Minimum number of bets for a setting to be eligible.

    Returns:
    - Tuple (best_row as a Series, is_active as bool). best_row is None when nothing is eligible.
    """
    eligible = grid[grid[bets_col] >= min_bets]
    if eligible.empty:
        return None, False
    best = eligible.sort_values(metric, ascending=False).iloc[0]
    return best, bool(best[metric] > 0)
