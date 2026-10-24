"""Tests for the horse racing module, run on the synthetic circuit."""

import numpy as np
import pandas as pd
import pytest

from lynk.horse_racing import synthetic, features, finish_order, bets, schema


@pytest.fixture(scope='module')
def races():
    return schema.validate(synthetic.make_races(n_days=40, races_per_day=6, seed=3))


def test_one_winner_per_race(races):
    assert (races.groupby('race_id').won.sum() == 1).all()


def test_history_features_never_use_today(races):
    out = features.add_history_features(races, ['speed_figure'])
    first_starts = out.groupby('horse').head(1)
    assert first_starts.speed_figure_lag1.isna().all()
    # the lag of a start equals the previous start's figure
    horse = out[out.horse == out.horse.value_counts().index[0]].sort_values('date')
    assert np.allclose(horse.speed_figure_lag1.values[1:], horse.speed_figure.values[:-1])


def test_connection_form_excludes_race_day(races):
    out = features.add_connection_form(races, 'trainer', window_days=30)
    day_one = out[out.date == out.date.min()]
    assert (day_one.trainer_starts == 0).all()


def test_relative_features_rank_one_is_best(races):
    out = features.add_relative_features(races, ['speed_figure'])
    best = out.loc[out.groupby('race_id').speed_figure.idxmax()]
    assert (best.speed_figure_rank == 1).all()
    assert np.allclose(best.speed_figure_gap_best, 0)


def test_harville_pairs_sum_to_win_probability():
    p = np.array([0.4, 0.3, 0.2, 0.1])
    pairs = finish_order.pair_probabilities(p, gamma_second=1.0)
    assert np.allclose(pairs.sum(axis=1), p)
    top2 = finish_order.top_k_probabilities(p, 2, gamma_second=1.0)
    assert top2.sum() == pytest.approx(2.0)


def test_win_strategy_runs_and_settles(races):
    frame = races.copy()
    frame['p_model'] = frame.true_probability
    ledger = bets.win_strategy(frame, edge_cutoff=0.05, bankroll_per_race=5000.0)
    assert len(ledger) > 0
    assert (ledger.stake >= 2).all()
    assert set(ledger.columns) >= {'event_id', 'stake', 'payout_per_dollar'}


def test_multi_pool_solver_returns_bets(races):
    race = races[races.race_id == races.race_id.iloc[0]].copy()
    race['p_model'] = race.true_probability
    out = bets.multi_pool_race_bets(race, pools=('win', 'place'), bankroll=5000.0)
    assert set(out.pool) <= {'win', 'place'}
