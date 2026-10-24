"""
Lynk quickstart: from raw races to a backtested win strategy, on the synthetic circuit.

Run from the repo root:
    python examples/quickstart.py

Steps:
1. Simulate a year of racing (no proprietary data needed).
2. Build leakage-safe features: horse history, connection form, field-relative ranks.
3. Rate trainers and jockeys against the closing odds of PAST races (online, day by day).
4. Split by date: train / valid / test.
5. Fit a fundamental model (no odds) and a market model that learns corrections to the odds at decision time.
6. Bet the win pool on VALID over a small grid, pick the setting with a pre-registered rule, score TEST once.
"""

import sys
from pathlib import Path

# On some Windows builds LightGBM and pandas/pyarrow ship different OpenMP runtimes; importing LightGBM first avoids
# an access violation when the Dataset is constructed.
try:
    import lightgbm  # noqa: F401
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import pandas as pd

from lynk.core import metrics, validation, backtest
from lynk.core.models import fit_group_softmax, predict_group_softmax
from lynk.core.probability import normalize_within_groups
from lynk.core.ratings import OnlineRatings, EntityType
from lynk.horse_racing import synthetic, features, bets, schema


def build_frame(seed=11):
    # Step 1: Simulate the circuit and sort it into the schema order
    races = schema.validate(synthetic.make_races(n_days=360, races_per_day=10, trainer_mispricing=0.35, seed=seed))
    races['finish_pct'] = (races.finish - 1) / (races.field_size - 1)

    # Step 2: Horse history, connection form and field-relative features (all strictly pre-race)
    races = features.add_history_features(races, ['speed_figure', 'finish_pct'])
    races = features.add_connection_form(races, 'trainer', window_days=180)
    races = features.add_connection_form(races, 'jockey', window_days=180)
    races = features.add_relative_features(races, ['speed_figure_mean3', 'speed_figure_best5'])
    races = features.add_relative_features(races, ['finish_pct_mean3'], higher_is_better=False)

    # Step 3: Online ratings vs the closing odds of past races (closing odds used as history only)
    closing = 1.0 / (races.final_odds + 1.0)
    races['log_closing_p'] = np.log(normalize_within_groups(closing.values, races.race_id.values))
    ratings = OnlineRatings([EntityType('trainer', ['trainer'], prior_variance=0.01),
                             EntityType('jockey', ['jockey'], prior_variance=0.005),
                             EntityType('trainer_track', ['trainer', 'track'], prior_variance=0.003)])
    rated = ratings.run(races, 'race_id', 'date', 'won', offset_col='log_closing_p')
    races = pd.concat([races, rated[['trainer_mean', 'jockey_mean', 'trainer_track_mean', 'rating_total']]], axis=1)
    print('ratings beat the closing odds by', round(ratings.diagnostics['gain'], 4), 'log-loss per race')

    # Step 4: Market at decision time (never the final pool)
    races['log_crowd_p'] = np.log(normalize_within_groups(races.win_money.values, races.race_id.values))
    races['split'] = validation.date_split(races.date, '2023-09-01', '2023-11-01')
    return races


FUNDAMENTAL = ['speed_figure_lag1', 'speed_figure_mean3', 'speed_figure_best5', 'finish_pct_mean3', 'days_since_last',
               'prior_starts', 'trainer_win_rate', 'jockey_win_rate', 'speed_figure_mean3_rank',
               'speed_figure_mean3_gap_best', 'speed_figure_best5_z', 'finish_pct_mean3_rank', 'distance', 'field_size']
MARKET_EXTRA = ['fund_logit', 'trainer_mean', 'jockey_mean', 'trainer_track_mean', 'rating_total']


def main():
    races = build_frame()
    fit_mask = (races.split == 'train') & (races.date < '2023-07-01')
    es_mask = (races.split == 'train') & (races.date >= '2023-07-01')

    # Step 5a: Fundamental model (no odds at all)
    tr, es = races[fit_mask], races[es_mask]
    fund = fit_group_softmax(tr[FUNDAMENTAL], tr.won, tr.race_id, es[FUNDAMENTAL], es.won, es.race_id,
                             params=dict(num_leaves=15, min_data_in_leaf=100))
    races['p_fund'] = predict_group_softmax(fund, races[FUNDAMENTAL], races.race_id)
    races['fund_logit'] = np.log(races.p_fund) - races.log_crowd_p

    # Step 5b: Market model: start from the crowd's odds and learn corrections
    cols = MARKET_EXTRA
    tr, es = races[fit_mask], races[es_mask]
    market = fit_group_softmax(tr[cols], tr.won, tr.race_id, es[cols], es.won, es.race_id,
                               init_score=tr.log_crowd_p.values, init_score_valid=es.log_crowd_p.values,
                               params=dict(num_leaves=7, min_data_in_leaf=200, learning_rate=0.02))
    races['p_model'] = predict_group_softmax(market, races[cols], races.race_id, init_score=races.log_crowd_p.values)

    for split in ['valid', 'test']:
        part = races[races.split == split]
        print(f'{split}: log-loss  fundamental {metrics.group_log_loss(part.p_fund, part.won, part.race_id):.4f}'
              f' | crowd at decision {metrics.group_log_loss(np.exp(part.log_crowd_p), part.won, part.race_id):.4f}'
              f' | model {metrics.group_log_loss(part.p_model, part.won, part.race_id):.4f}')

    # Step 6: Strategy grid on VALID, pre-registered pick, TEST once
    settings = [dict(edge_cutoff=c, kelly_multiplier=k) for c in (0.0, 0.05, 0.10, 0.20) for k in (0.25, 0.5)]
    run = lambda frame: (lambda **s: backtest.settle_ledger(bets.win_strategy(frame, bankroll_per_race=10000.0, **s)))
    grid = backtest.grid_search(run(races[races.split == 'valid']), settings, bankroll=10000.0)
    best, active = validation.select_preregistered(grid, metric='log_growth', min_bets=100)
    print(grid[['edge_cutoff', 'kelly_multiplier', 'bets', 'turnover', 'roi', 'log_growth']].round(3).to_string(index=False))
    print('pre-registered pick:', dict(edge_cutoff=best.edge_cutoff, kelly_multiplier=best.kelly_multiplier), 'active' if active else 'inactive')
    if not active:
        print('The rule says do not bet: no setting had positive validation log growth. Stopping before TEST.')
        return races, grid, None
    test_ledger = run(races[races.split == 'test'])(edge_cutoff=best.edge_cutoff, kelly_multiplier=best.kelly_multiplier)
    summary = backtest.summarize_ledger(test_ledger)
    print('TEST:', {k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()})
    return races, grid, summary


if __name__ == '__main__':
    main()
