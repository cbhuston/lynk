"""
The column contract for horse racing frames.

Lynk works on one row per starter (horse in a race). Map your data source (DRF charts, Equibase, a tote feed) onto
these names once, and every feature, model and bet function in ``lynk.horse_racing`` works on it.
"""

import pandas as pd

# Required to identify a starter and order events in time
KEYS = ['race_id', 'date', 'horse']

# Pre-race facts (known before betting opens)
PRE_RACE = ['track', 'surface', 'distance', 'race_class', 'trainer', 'jockey', 'owner', 'sire', 'post', 'field_size',
            'morning_line']

# Market snapshot at the decision time (known when the bet is placed)
MARKET = ['win_money', 'win_pool', 'place_money', 'place_pool', 'show_money', 'show_pool', 'takeout']

# Results and post-race facts (only usable as labels, for settlement, or LAGGED as history for later races)
POST_RACE = ['finish', 'won', 'speed_figure', 'final_odds', 'win_payoff', 'place_payoff', 'show_payoff']


def validate(frame, required=KEYS):
    """
    Checks that required columns exist and that each race's rows are contiguous.

    Parameters:
    - frame: Starter-level DataFrame.
    - required: Columns that must be present.

    Returns:
    - The frame sorted by date, race_id (and post when available), ready for the group functions.
    """
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f'missing columns: {missing}')
    sort_cols = ['date', 'race_id'] + (['post'] if 'post' in frame.columns else [])
    out = frame.sort_values(sort_cols).reset_index(drop=True)
    out['date'] = pd.to_datetime(out['date'])
    return out
