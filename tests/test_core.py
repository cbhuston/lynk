"""Unit tests for the market-agnostic core."""

import numpy as np
import pandas as pd
import pytest

from lynk.core import kelly, parimutuel, portfolio, metrics, probability, validation, backtest
from lynk.core.ratings import OnlineRatings, EntityType


def test_group_softmax_sums_to_one_per_event():
    p = probability.group_softmax([1.0, 2.0, 3.0, 0.0, 0.0], ['a', 'a', 'a', 'b', 'b'])
    assert np.allclose([p[:3].sum(), p[3:].sum()], 1.0)
    assert p[2] > p[1] > p[0]


def test_group_log_loss_matches_hand_calculation():
    p = np.array([0.5, 0.25, 0.25, 0.8, 0.2])
    y = np.array([0, 1, 0, 1, 0])
    expected = (-np.log(0.25) - np.log(0.8)) / 2
    assert metrics.group_log_loss(p, y, ['a', 'a', 'a', 'b', 'b']) == pytest.approx(expected)


def test_kelly_fraction_textbook_example():
    # 60% coin flip at even money: f* = 2p - 1 = 0.2
    assert kelly.kelly_fraction(0.6, 2.0) == pytest.approx(0.2)
    assert kelly.kelly_fraction(0.4, 2.0) == 0.0


def test_mutually_exclusive_kelly_matches_numeric_solver():
    p = np.array([0.5, 0.3, 0.2])
    odds = np.array([2.4, 3.2, 3.0])          # 1/odds sums to 1.06: a book with a 6% margin
    exact = kelly.kelly_mutually_exclusive(p, odds)
    gross = np.diag(odds)                    # scenario s = outcome s wins
    numeric = kelly.kelly_scenarios(gross, p)
    assert np.allclose(exact, numeric, atol=2e-3)


def test_risk_constrained_kelly_bets_less_than_kelly():
    gross = np.array([[3.0], [0.0]])
    pi = np.array([0.45, 0.55])
    full = kelly.kelly_scenarios(gross, pi)
    safe = kelly.risk_constrained_kelly(gross, pi, alpha=0.8, beta=0.1)
    assert 0 < safe[0] < full[0]


def test_max_ev_bet_is_the_maximum_of_expected_profit():
    p, T, m, t = 0.25, 100000.0, 15000.0, 0.17
    best = parimutuel.max_ev_bet(p, T, m, t)
    grid = np.linspace(1, 20000, 20000)
    profits = parimutuel.expected_profit(p, T, m, t, grid)
    assert best == pytest.approx(grid[np.argmax(profits)], rel=0.01)


def test_max_ev_bet_zero_without_edge():
    assert parimutuel.max_ev_bet(0.10, 100000.0, 15000.0, 0.17) == 0.0


def test_breakage_rounds_down_to_the_dime():
    # $2 ticket paying $7.37 rounds down to $7.30 -> 3.65 per $1
    assert parimutuel.apply_breakage(3.685) == pytest.approx(3.65)
    assert parimutuel.apply_breakage(1.01, minimum=1.05) == pytest.approx(1.05)


def test_portfolio_gradient_matches_finite_differences():
    pools = [portfolio.Pool('win', 0.17, {1: 4000.0, 2: 3000.0, 3: 2000.0, 4: 1000.0}),
             portfolio.Pool('place', 0.17, {1: 1500.0, 2: 1200.0, 3: 900.0, 4: 400.0})]
    p = np.array([0.35, 0.3, 0.2, 0.15])
    scen = [portfolio.Scenario(p[i] * p[j] / (1 - p[i]), {'win': (i + 1,), 'place': (i + 1, j + 1)})
            for i in range(4) for j in range(4) if i != j]
    scaled = [portfolio.Pool(x.name, x.takeout, {k: v / 10000 for k, v in x.money.items()}) for x in pools]
    cols, money, pool_of, _, winners = portfolio._index_problem(scaled, scen, None)
    probs = np.array([s.probability for s in scen])
    f = np.full(len(cols), 0.01)
    w, g = portfolio._wealth_and_gradient(f, scaled, money, pool_of, winners, probs)
    eps = 1e-7
    for j in range(len(f)):
        f2 = f.copy(); f2[j] += eps
        w2, _ = portfolio._wealth_and_gradient(f2, scaled, money, pool_of, winners, probs)
        assert np.allclose((w2 - w) / eps, g[:, j], atol=1e-5)


def test_portfolio_single_pool_bets_the_value_horse():
    # Horse 4 is 20% to win but only 10% of the money: it should get the biggest stake
    pool = portfolio.Pool('win', 0.17, {1: 40000.0, 2: 30000.0, 3: 20000.0, 4: 10000.0})
    p = {1: 0.35, 2: 0.27, 3: 0.18, 4: 0.20}
    scen = [portfolio.Scenario(p[k], {'win': (k,)}) for k in p]
    bets = portfolio.optimize_portfolio([pool], scen, bankroll=10000.0)
    assert bets.sort_values('stake').outcome.iloc[-1] == 4


def test_date_split_and_preregistered_selection():
    labels = validation.date_split(['2023-05-01', '2024-02-01', '2024-08-01'], '2024-01-01', '2024-07-01')
    assert list(labels) == ['train', 'valid', 'test']
    grid = pd.DataFrame({'cut': [0.0, 0.1, 0.2], 'bets': [5000, 800, 120], 'log_growth': [-1.0, 2.0, 5.0]})
    best, active = validation.select_preregistered(grid, min_bets=300)
    assert best.cut == 0.1 and active


def test_leakage_assertion_catches_late_features():
    frame = pd.DataFrame({'decision': pd.to_datetime(['2024-01-02', '2024-01-02']),
                          'feat_time': pd.to_datetime(['2024-01-01', '2024-01-02'])})
    with pytest.raises(ValueError):
        validation.assert_features_precede(frame, 'decision', ['feat_time'])


def test_ratings_learn_a_planted_effect():
    rng = np.random.default_rng(0)
    rows = []
    for d in range(300):
        for r in range(4):
            trainers = rng.integers(0, 5, 6)
            strength = np.where(trainers == 0, 0.6, 0.0)            # trainer 0 is underrated by the offset
            p = np.exp(strength); p /= p.sum()
            winner = rng.choice(6, p=p)
            for k in range(6):
                rows.append(dict(event=f'{d}_{r}', date=pd.Timestamp('2020-01-01') + pd.Timedelta(days=d),
                                 trainer=trainers[k], won=int(k == winner), offset=np.log(1 / 6)))
    frame = pd.DataFrame(rows)
    ratings = OnlineRatings([EntityType('trainer', ['trainer'], prior_variance=0.05)])
    out = ratings.run(frame, 'event', 'date', 'won', 'offset')
    on_last_day = (frame.date == frame.date.max()).values
    trainer_zero = (frame.trainer == 0).values
    means = out.trainer_mean.values
    assert means[on_last_day & trainer_zero].mean() > means[on_last_day & ~trainer_zero].mean() + 0.3
    assert ratings.diagnostics['gain'] > 0


def test_summarize_ledger_roi():
    ledger = pd.DataFrame({'event_id': ['a', 'b'], 'date': ['2024-01-01', '2024-01-02'], 'pool': 'win',
                           'outcome': [1, 2], 'stake': [10.0, 10.0], 'payout_per_dollar': [3.0, 0.0],
                           'rebate_rate': [0.1, 0.1]})
    summary = backtest.summarize_ledger(backtest.settle_ledger(ledger), n_boot=200)
    assert summary['profit'] == pytest.approx(30 + 2 - 20)
    assert summary['roi'] == pytest.approx(12 / 20)
