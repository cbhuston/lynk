"""Tests for the pool-by-pool racing simulator."""

import json

import numpy as np
import pandas as pd
import pytest

from lynk.horse_racing.simulator import simulate, to_schema_frame, load_params
from lynk.horse_racing.simulator.fidelity import stylized_facts, compare


@pytest.fixture(scope='module')
def result():
    return simulate(n_days=28, seed=5)


def test_every_race_has_one_winner_and_a_full_chart(result):
    st = result.starters
    assert (st.groupby('race_id').finish.apply(lambda f: sorted(f) == list(range(1, len(f) + 1)))).all()
    assert (st[st.finish == 1].beaten_lengths == 0).all()
    assert st.speed_figure.notna().all()


def test_win_payoff_matches_the_final_pool(result):
    snaps = result.wps_snapshots[result.wps_snapshots.offset == 'final']
    take = result.params['tote']['takeout']['win']
    st = result.starters[result.starters.finish == 1].merge(snaps, on=['race_id', 'post'])
    totals = snaps.groupby('race_id').win_money.sum()
    raw = 2 * (1 - take) * st.race_id.map(totals) / st.win_money
    expected = np.maximum(np.floor(raw / 0.10 + 1e-9) * 0.10, 2.10)
    assert np.allclose(st.win_payoff.values, expected.values, atol=1e-6)


def test_pools_only_grow_toward_the_close(result):
    order = {k: i for i, k in enumerate(result.params['tote']['offsets'])}
    order = {str(k): i for k, i in order.items()}; order['final'] = len(order)
    t = result.pool_totals[result.pool_totals.pool == 'win'].copy()
    t['i'] = t.offset.map(order)
    diffs = t.sort_values(['race_id', 'i']).groupby('race_id').total.diff().dropna()
    assert (diffs >= -1e-6).all()


def test_place_and_show_payoffs_respect_the_minimum(result):
    for col in ['place_payoff', 'show_payoff']:
        paid = result.starters[result.starters[col] > 0][col]
        assert (paid >= 2.10 - 1e-9).all()


def test_pick_n_carryovers_have_no_payout(result):
    h = result.horizontals
    if len(h):
        assert h[h.carryover].payout.isna().all()
        assert h[~h.carryover].payout.notna().all()


def test_schema_frame_feeds_the_rest_of_the_package(result):
    frame = to_schema_frame(result, decision_offset='60')
    assert {'race_id', 'date', 'horse', 'won', 'win_money', 'win_pool', 'takeout', 'win_payoff'} <= set(frame.columns)
    assert (frame.groupby('race_id').won.sum() == 1).all()


def test_fidelity_scorecard_runs(result):
    facts = stylized_facts(result, gamma_races=200)
    table = compare(facts, facts)
    assert np.allclose(table.abs_error.values, 0)


def test_calibrated_file_overrides_defaults(tmp_path):
    path = tmp_path / 'calibrated.json'
    path.write_text(json.dumps({'running': {'performance_noise_sd': 0.9}}))
    params = load_params(str(path))
    assert params['running']['performance_noise_sd'] == 0.9
    assert params['tote']['takeout']['win'] == load_params().get('tote')['takeout']['win']
