"""
Stylized facts: the statistics a realistic racing simulator has to reproduce.

``stylized_facts(result)`` computes them from a simulation. The same statistics, measured on real data in the same
structure, make a scorecard: ``compare(sim_facts, real_facts)`` returns one row per statistic with both values and the
relative error. Nothing here needs real data to run.
"""

import numpy as np
import pandas as pd

OFFSETS = ['1800', '1200', '900', '600', '300', '180', '120', '90', '60', '30', '15', '0']
qs = lambda x, ps=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95): {str(p): float(np.nanquantile(x, p)) for p in ps}
FLB_BINS = [0, .02, .04, .07, .10, .15, .20, .30, .40, .55, 1]


def _flb(p, won):
    b = pd.cut(p, FLB_BINS)
    t = pd.DataFrame({'p': p, 'w': won}).groupby(b, observed=True).agg(pred=('p', 'mean'), actual=('w', 'mean'), n=('w', 'size'))
    return t.reset_index(drop=True).round(5).to_dict('list')


def _gamma2(frame):
    best = None
    for g in np.arange(0.5, 1.21, 0.05):
        ll, n = 0.0, 0
        for _, r in frame.groupby('race_id'):
            if (r.finish == 1).sum() != 1 or (r.finish == 2).sum() != 1:
                continue
            q = r.p.values ** g; q /= q.sum()
            i = np.flatnonzero(r.finish.values == 1)[0]; j = np.flatnonzero(r.finish.values == 2)[0]
            ll -= np.log(q[j] / (1 - q[i])); n += 1
        if n and (best is None or ll / n < best[1]):
            best = (round(float(g), 2), ll / n)
    return best[0] if best else np.nan


def stylized_facts(result, gamma_races=3000):
    """
    Computes the scorecard statistics from a ``SimulationResult``.

    Returns:
    - Nested dict with sections racing, market_close, tote and exotics (same keys as the real-data targets).
    """
    races, st = result.races, result.starters.copy()
    out = {'racing': {}, 'market_close': {}, 'tote': {}, 'exotics': {}}
    R, M, T, E = out['racing'], out['market_close'], out['tote'], out['exotics']
    # ---- racing structure
    cards = races.groupby(['track', 'date']).size()
    R['races_per_card'] = (cards.value_counts(normalize=True).sort_index().round(5)).rename(index=str).to_dict()
    R['race_type_share'] = races.race_type.value_counts(normalize=True).round(4).to_dict()
    R['field_size_mean'] = float(races.field_size.mean())
    R['surface_share'] = {t: g.surface.value_counts(normalize=True).round(4).to_dict() for t, g in races.groupby('race_type')}
    st = st.sort_values(['horse', 'date', 'race_id'])
    gaps = st.groupby('horse').date.diff().dt.days.dropna()
    R['days_between_starts'] = qs(gaps, (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)); R['layoff_share_90d'] = float((gaps >= 90).mean())
    R['age_pmf'] = st.age.clip(upper=10).value_counts(normalize=True).sort_index().round(4).rename(index=str).to_dict()
    starts_by_trainer = st.groupby('trainer').size().sort_values(ascending=False)
    R['trainer_starts_share_top10pct'] = float(starts_by_trainer.iloc[:max(1, len(starts_by_trainer) // 10)].sum() / starts_by_trainer.sum())
    st['won'] = (st.finish == 1).astype(int)
    st['p_final'] = 1 / (st.final_odds + 1)
    st['p_final'] = st.p_final / st.groupby('race_id').p_final.transform('sum')
    for who in ['trainer', 'jockey']:
        g = st.groupby(who).agg(n=('won', 'size'), w=('won', 'sum'), e=('p_final', 'sum'))
        big = g[g.n >= 100]
        wr = big.w / big.n
        R[f'{who}_winrate_excess_sd'] = float(np.sqrt(max(wr.var() - (wr * (1 - wr) / big.n).mean(), 0)))
        R[f'{who}_ae_sd_200plus'] = float(((big.w / big.n) / (big.e / big.n)).std())
    R['speed_figure'] = dict(mean=float(st.speed_figure.mean()), sd=float(st.speed_figure.std()))
    R['speed_figure_within_horse_sd'] = float(st.groupby('horse').speed_figure.std().median())
    prev = st.groupby('horse').speed_figure.shift(1)
    R['speed_figure_lag1_corr'] = float(pd.concat([st.speed_figure, prev], axis=1).dropna().corr().iloc[0, 1])
    st = st.merge(races[['race_id', 'field_size']], on='race_id')
    e = (st.pos_call1 - 1) / (st.field_size - 1); fpct = (st.finish - 1) / (st.field_size - 1)
    R['early_vs_finish_corr'] = float(np.corrcoef(e[st.field_size >= 5], fpct[st.field_size >= 5])[0, 1])
    R['winner_early_pct'] = qs(e[st.won == 1])
    R['beaten_lengths_2nd'] = qs(st.beaten_lengths[st.finish == 2])
    # ---- market at the close
    M['flb_final'] = _flb(st.p_final.values, st.won.values)
    fav = st.loc[st.groupby('race_id').p_final.idxmax()]
    M['favorite_win_rate'] = float(fav.won.mean()); M['favorite_implied'] = qs(fav.p_final)
    M['final_odds_logloss'] = float(-np.log(st.p_final[st.won == 1]).mean())
    M['final_odds_quantiles'] = qs(st.final_odds, (0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99))
    ids = st.race_id.drop_duplicates().sample(min(gamma_races, st.race_id.nunique()), random_state=0)
    M['harville_gamma2_mle'] = _gamma2(st[st.race_id.isin(set(ids))].rename(columns={'p_final': 'p'})[['race_id', 'p', 'finish']])
    M['favorite_show_minimum_rate'] = float(((fav.show_payoff > 0) & (fav.show_payoff <= 2.10)).sum() / max((fav.show_payoff > 0).sum(), 1))
    # ---- tote dynamics
    snaps = result.wps_snapshots
    wide = snaps.pivot_table(index=['race_id', 'post'], columns='offset', values='win_money').reset_index()
    wide = wide.merge(st[['race_id', 'post', 'won', 'p_final']], on=['race_id', 'post'])
    tot = wide.groupby('race_id')[OFFSETS + ['final']].transform('sum')
    p0 = wide['0'] / tot['0']; pf = wide['final'] / tot['final']
    late = np.log(pf / p0)
    bucket = pd.cut(p0, [0, .03, .07, .15, .3, 1])
    T['late_drift_sd_by_odds'] = {str(k): float(v) for k, v in late.groupby(bucket, observed=True).std().items()}
    T['late_drift_mean_winners'] = float(late[wide.won == 1].mean()); T['late_drift_mean_losers'] = float(late[wide.won == 0].mean())
    for o in ['1800', '600', '300', '60', '15']:
        po = wide[o] / tot[o]
        d = np.log(p0 / po.clip(1e-5))
        T[f'drift_to_posttime_from_{o}'] = dict(sd=float(d.std()), mean_winners=float(d[wide.won == 1].mean()), mean_losers=float(d[wide.won == 0].mean()))
    T['flb_posttime'] = _flb(p0.values, wide.won.values)
    T['logloss_posttime'] = float(-np.log(p0[wide.won == 1]).mean()); T['logloss_final'] = float(-np.log(pf[wide.won == 1]).mean())
    pools = result.pool_totals.pivot_table(index=['race_id', 'offset'], columns='pool', values='total').reset_index()
    post = pools[pools.offset == '0'].set_index('race_id'); final = pools[pools.offset == 'final'].set_index('race_id')
    wps0 = post.win + post.place + post.show
    T['win_share_of_wps'] = qs(post.win / wps0); T['place_share_of_wps'] = qs(post.place / wps0)
    T['final_wps_over_posttime'] = qs((final.win + final.place + final.show) / wps0)
    T['arrival_share_of_posttime_win_pool'] = {o: qs(pools[pools.offset == o].set_index('race_id').win / post.win, (0.1, 0.25, 0.5, 0.75, 0.9)) for o in OFFSETS}
    for pool, col in [('pl', 'place_money'), ('sh', 'show_money')]:
        s0 = snaps[snaps.offset == '0'].merge(wide[['race_id', 'post']], on=['race_id', 'post'])
        share = s0[col] / s0.groupby('race_id')[col].transform('sum')
        win_share = s0.win_money / s0.groupby('race_id').win_money.transform('sum')
        X = np.c_[np.ones(len(s0)), np.log(win_share)]
        b, *_ = np.linalg.lstsq(X, np.log(share), rcond=None)
        T[f'{pool}_share_vs_win'] = dict(intercept=float(b[0]), slope=float(b[1]), resid_sd=float(np.std(np.log(share) - X @ b)))
    # ---- exotics
    for pool in ['exacta', 'trifecta', 'superfecta']:
        col = f'pool_{pool}'
        if col in races:
            E[f'{pool}_offered'] = float(races[col].notna().mean())
            E[f'{pool}_over_win'] = qs((races[col] / races.pool_win).dropna(), (0.1, 0.25, 0.5, 0.75, 0.9))
    hz = result.horizontals
    if len(hz):
        for bet, g in hz.groupby('bet'):
            E[f'{bet}_carryover_rate'] = float(g.carryover.mean())
    pr = result.exacta_probables[result.exacta_probables.offset == '0']
    w0 = wide[['race_id', 'post']].assign(p=p0.values)
    pr = pr.merge(w0.rename(columns={'post': 'first', 'p': 'pa'}), on=['race_id', 'first']).merge(w0.rename(columns={'post': 'second', 'p': 'pb'}), on=['race_id', 'second'])
    pr['implied'] = 1 / pr.payout
    pr['implied'] = pr.implied / pr.groupby('race_id').implied.transform('sum')
    pr['r'] = np.log(pr.implied / (pr.pa * pr.pb / (1 - pr.pa)))
    E['exacta_vs_harville_by_second_odds'] = {str(k): float(v) for k, v in pr.groupby(pd.cut(pr.pb, [0, .03, .07, .15, .3, 1]), observed=True).r.mean().items()}
    return out


def _flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = f'{prefix}.{k}' if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, list):
            for i, x in enumerate(v):
                out[f'{key}[{i}]'] = x
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def compare(sim_facts, real_facts):
    """
    One row per statistic present in both dicts: real value, simulated value and relative error.

    Returns:
    - DataFrame sorted by absolute relative error (worst first).
    """
    s, r = _flatten(sim_facts), _flatten(real_facts)
    rows = [dict(stat=k, real=r[k], sim=s[k], abs_error=s[k] - r[k], rel_error=(s[k] - r[k]) / abs(r[k]) if r[k] else np.nan)
            for k in s if k in r and not k.split('.')[-1].startswith('n[') and np.isfinite(s[k]) and np.isfinite(r[k])]
    out = pd.DataFrame(rows)
    return out.reindex(out.rel_error.abs().sort_values(ascending=False).index).reset_index(drop=True)
