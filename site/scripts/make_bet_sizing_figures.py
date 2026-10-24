"""
Reproduces every figure and number in the bet sizing report using only the lynk package.

Run from the repo root:  python site/scripts/make_bet_sizing_figures.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lynk.core import kelly, parimutuel
from lynk.core.portfolio import Pool, Scenario, optimize_portfolio, expected_log_growth
from lynk.horse_racing.finish_order import finish_order_scenarios
from lynk.horse_racing import synthetic

OUT = ROOT / 'site' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
BLUE, GRAY, LIGHT, ORANGE, INK2, GRID = '#2a78d6', '#8a8983', '#d9d8d2', '#eb6834', '#52514e', '#e7e6e1'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': LIGHT, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 200})
numbers = {}


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches='tight')
    plt.close(fig)


# Step 1: Growth rate versus bet fraction for one fixed-odds bet (25% to win at 4-1, i.e. decimal 5.0)
p, odds = 0.25, 5.0
f_star = float(kelly.kelly_fraction(p, odds))
f = np.linspace(0, 0.30, 400)
g = kelly.expected_log_growth(f, p, odds)
fig, ax = plt.subplots(figsize=(7.2, 3.0))
ax.plot(f * 100, g * 100, color=BLUE, lw=2)
for frac, lab, c in [(1.0, 'full Kelly', BLUE), (0.5, 'half Kelly', GRAY), (2.0, '2x Kelly', ORANGE)]:
    x = frac * f_star
    y = float(kelly.expected_log_growth(x, p, odds))
    ax.plot(x * 100, y * 100, 'o', color=c, ms=6)
    offset = {1.0: (8, 6), 0.5: (-10, -42), 2.0: (6, -30)}[frac]
    ax.annotate(f'{lab}\n{x*100:.1f}% of bankroll\ngrowth {y*100:.2f}%/bet', (x * 100, y * 100), xytext=offset,
                textcoords='offset points', fontsize=7.5, color=c)
ax.axhline(0, color=INK2, lw=0.7)
ax.set_xlabel('Share of bankroll bet (%)'); ax.set_ylabel('Expected log growth per bet (%)')
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
save(fig, 'kelly_growth_curve.png')
numbers['kelly'] = dict(p=p, odds=odds, f_star=round(f_star, 4), g_full=round(float(kelly.expected_log_growth(f_star, p, odds)), 5),
                        g_half=round(float(kelly.expected_log_growth(f_star / 2, p, odds)), 5),
                        g_double=round(float(kelly.expected_log_growth(2 * f_star, p, odds)), 5))

# Step 2: Bankroll paths when the probability estimate is a little optimistic (true p = 0.23, model says 0.25)
rng = np.random.default_rng(1)
n_bets, n_paths, true_p = 1000, 2000, 0.23
wins = rng.random((n_paths, n_bets)) < true_p
fig, ax = plt.subplots(figsize=(7.2, 3.0))
finals = {}
for frac, c, lab in [(1.0, BLUE, 'full Kelly'), (0.5, GRAY, 'half Kelly'), (0.25, LIGHT, 'quarter Kelly')]:
    stake = frac * f_star
    log_w = np.cumsum(np.where(wins, np.log1p(stake * (odds - 1)), np.log1p(-stake)), axis=1)
    med = np.median(log_w, axis=0); lo = np.percentile(log_w, 10, axis=0)
    ax.plot(np.exp(med), color=c, lw=2, label=f'{lab}: median')
    ax.plot(np.exp(lo), color=c, lw=1, ls='--', label=f'{lab}: 10th percentile')
    finals[lab] = dict(median=round(float(np.exp(med[-1])), 3), p10=round(float(np.exp(lo[-1])), 3),
                       p_below_half=round(float((np.exp(log_w).min(axis=1) < 0.5).mean()), 3))
ax.set_yscale('log'); ax.set_xlabel('Number of bets'); ax.set_ylabel('Bankroll multiple (log scale)')
ax.legend(frameon=False, fontsize=7.5); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
save(fig, 'kelly_paths.png')
numbers['paths'] = dict(true_p=true_p, model_p=p, **finals)

# Step 3: Mutually exclusive outcomes: one race, fixed odds
p_race = np.array([0.32, 0.24, 0.16, 0.12, 0.09, 0.07])
odds_race = np.array([2.6, 3.6, 7.0, 7.5, 12.0, 14.0])
single = kelly.kelly_fraction(p_race, odds_race)
joint = kelly.kelly_mutually_exclusive(p_race, odds_race)
numbers['mutually_exclusive'] = dict(p=p_race.tolist(), odds=odds_race.tolist(), ev=(p_race * odds_race - 1).round(3).tolist(),
                                     separate=single.round(4).tolist(), joint=joint.round(4).tolist())

# Step 4: Pari-mutuel impact: expected dollars versus stake
T, m, t = 100000.0, 15000.0, 0.17
stakes = np.linspace(0, 12000, 600)
profit = parimutuel.expected_profit(p, T, m, t, stakes)
s_max = float(parimutuel.max_ev_bet(p, T, m, t))
fig, ax = plt.subplots(figsize=(7.2, 3.0))
ax.plot(stakes / 1000, profit, color=BLUE, lw=2)
ax.axvline(s_max / 1000, color=INK2, lw=0.8, ls='--'); ax.axvline(2 / 3 * s_max / 1000, color=ORANGE, lw=0.8, ls='--')
ax.text(s_max / 1000 + 0.1, profit.max() * 0.15, f'max-EV bet\n${s_max:,.0f}', fontsize=7.5, color=INK2)
ax.text(2 / 3 * s_max / 1000 - 0.1, -700, f'2/3 cap\n${2/3*s_max:,.0f}', fontsize=7.5, color=ORANGE, ha='right')
ax.axhline(0, color=INK2, lw=0.7)
ax.set_xlabel('Stake ($ thousands)'); ax.set_ylabel('Expected profit ($)')
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
save(fig, 'parimutuel_impact.png')
ev0 = p * (1 - t) * T / m - 1
numbers['impact'] = dict(T=T, m=m, takeout=t, ev_at_zero=round(ev0, 4), max_ev_bet=round(s_max, 0),
                         max_ev_profit=round(float(parimutuel.expected_profit(p, T, m, t, s_max)), 0),
                         two_thirds_profit=round(float(parimutuel.expected_profit(p, T, m, t, 2 / 3 * s_max)), 0),
                         odds_at_max=round(float(parimutuel.payout_per_dollar(T, m, t, s_max)), 3))

# Step 5: Late money and the conditional payout (synthetic circuit)
races = synthetic.make_races(n_days=120, races_per_day=8, trainer_mispricing=0.35, crowd_skill=0.7, late_money_share=0.5, seed=5)
seen = races.win_money / races.groupby('race_id').win_money.transform('sum')
final = races.win_money_final / races.groupby('race_id').win_money_final.transform('sum')
races['drift'] = np.log(final / seen)
fig, ax = plt.subplots(figsize=(7.2, 2.8))
bins = np.linspace(-0.6, 0.6, 50)
ax.hist(races.drift[races.won == 0], bins=bins, density=True, color=LIGHT, label='horses that lost')
ax.hist(races.drift[races.won == 1], bins=bins, density=True, histtype='step', lw=2, color=BLUE, label='horses that won')
ax.set_xlabel('log(final money share / share at decision time)'); ax.set_ylabel('Density')
ax.legend(frameon=False, fontsize=7.5); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
save(fig, 'conditional_payout.png')
numbers['late_money'] = dict(mean_drift_winners=round(float(races.drift[races.won == 1].mean()), 4),
                             mean_drift_all=round(float(races.drift.mean()), 4),
                             payout_haircut_winners=round(float(1 - np.exp(-races.drift[races.won == 1].mean())), 4))

# Step 6: The portfolio solver on one race: win only vs win + place vs win + place + exacta
model_p = np.array([0.30, 0.22, 0.18, 0.12, 0.10, 0.08])
crowd_p = np.array([0.36, 0.25, 0.12, 0.12, 0.09, 0.06])
labels = [1, 2, 3, 4, 5, 6]
win_total, place_total = 80000.0, 30000.0
pools = {'win': Pool('win', 0.17, dict(zip(labels, crowd_p * win_total))),
         'place': Pool('place', 0.17, dict(zip(labels, np.array([0.36, 0.26, 0.09, 0.13, 0.09, 0.07]) * place_total))),
         'exacta': Pool('exacta', 0.21, {(labels[i], labels[j]): 60000.0 * q for (i, j), q in finish_order_scenarios(crowd_p, 2, 1.0)})}
scen = [Scenario(prob, {'win': (labels[i],), 'place': (labels[i], labels[j]), 'exacta': ((labels[i], labels[j]),)})
        for (i, j, k), prob in finish_order_scenarios(model_p, 3)]
bankroll = 20000.0
results = {}
for name, use in [('win only', ['win']), ('win + place', ['win', 'place']), ('win + place + exacta', ['win', 'place', 'exacta'])]:
    chosen = [pools[u] for u in use]
    b = optimize_portfolio(chosen, scen, bankroll, min_stake=2.0)
    results[name] = dict(bets=int(len(b)), staked=round(float(b.stake.sum()), 0),
                         growth=round(expected_log_growth(chosen, scen, bankroll, b) * 1e4, 2))
numbers['portfolio'] = results
fig, ax = plt.subplots(figsize=(7.2, 2.4))
names = list(results)
ax.barh(names, [results[n]['growth'] for n in names], color=[GRAY, '#7aaae6', BLUE], height=0.55)
for i, n in enumerate(names):
    ax.text(results[n]['growth'] + 0.2, i, f"{results[n]['growth']:.1f} bp, {results[n]['bets']} bets, ${results[n]['staked']:,.0f}",
            va='center', fontsize=7.5)
ax.set_xlabel('Expected log growth of the bankroll for this race (basis points)')
ax.grid(axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True)
save(fig, 'portfolio_example.png')

# Step 7: Risk-constrained Kelly versus fractional Kelly on the single bet
gross = np.array([[odds], [0.0]]); pi = np.array([p, 1 - p])
rck = {f'alpha={a}, beta={b}': round(float(kelly.risk_constrained_kelly(gross, pi, alpha=a, beta=b)[0]), 4)
       for a, b in [(0.7, 0.1), (0.5, 0.1), (0.8, 0.05)]}
numbers['risk_constrained'] = rck

(OUT / 'bet_sizing_numbers.json').write_text(json.dumps(numbers, indent=1))
print(json.dumps(numbers, indent=1))
