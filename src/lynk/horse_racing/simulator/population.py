"""
The population of a simulated racing circuit: tracks, horses, trainers, jockeys, owners and sires.

Every entity is invented. Names are built from neutral word lists so the output reads like a racing program without
copying anyone real. Hidden qualities (true ability, a trainer's skill with layoff horses, the part of a jockey's skill
the crowd does not price) live here and drive everything downstream.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_WORDS_A = ['Midnight', 'Silver', 'Copper', 'Northern', 'Quiet', 'Golden', 'Lucky', 'River', 'Iron', 'Summer', 'Desert',
            'Velvet', 'Rapid', 'Hidden', 'Royal', 'Bold', 'Crimson', 'Prairie', 'Harbor', 'Winter', 'Clever', 'Distant',
            'Rolling', 'Electric', 'Polar', 'Canyon', 'Granite', 'Sterling', 'Wild', 'Brave']
_WORDS_B = ['Ledger', 'Comet', 'Arrow', 'Echo', 'Harvest', 'Signal', 'Breeze', 'Anthem', 'Compass', 'Ember', 'Gambit',
            'Horizon', 'Lantern', 'Meadow', 'Nomad', 'Orbit', 'Parade', 'Quest', 'Rally', 'Sonnet', 'Tempo', 'Voyage',
            'Whistle', 'Cadence', 'Summit', 'Ripple', 'Beacon', 'Charter', 'Dividend', 'Frontier']
_SURNAMES = ['Alder', 'Birch', 'Cobalt', 'Dune', 'Elm', 'Flint', 'Gale', 'Hollow', 'Ivory', 'Juniper', 'Kestrel', 'Larch',
             'Moss', 'North', 'Oakes', 'Pike', 'Quill', 'Rowan', 'Sage', 'Thorne', 'Umber', 'Vale', 'Wren', 'Yarrow']


def _names(rng, n, words=(_WORDS_A, _WORDS_B)):
    a = rng.integers(0, len(words[0]), n); b = rng.integers(0, len(words[1]), n)
    return [f'{words[0][i]} {words[1][j]}' for i, j in zip(a, b)]


def _people(rng, n, prefix):
    initials = rng.integers(0, 26, n); last = rng.integers(0, len(_SURNAMES), n)
    return [f'{chr(65 + i)}. {_SURNAMES[j]} ({prefix}{k:04d})' for k, (i, j) in enumerate(zip(initials, last))]


@dataclass
class World:
    """All entities of the circuit plus the mutable state of every horse."""
    params: dict
    rng: np.random.Generator
    tracks: pd.DataFrame = None
    trainers: pd.DataFrame = None
    jockeys: pd.DataFrame = None
    owners: pd.DataFrame = None
    sires: pd.DataFrame = None
    horses: dict = field(default_factory=dict)   # column name -> numpy array (grows as horses are foaled)

    # ---------------------------------------------------------------- construction
    @classmethod
    def build(cls, params, seed=0, start_day=0):
        """
        Creates tracks, connections and an initial horse population of mixed ages.

        Parameters:
        - params: Parameter dict from ``load_params``.
        - seed: Random seed.
        - start_day: Day index of the first racing day (horses are created ready to run around it).
        """
        rng = np.random.default_rng(seed)
        world = cls(params=params, rng=rng)
        P = params['population']; W = params['world']
        n_tracks = W['n_tracks']
        tiers = np.array(W['track_tiers'][:n_tracks], dtype=float)
        # Step 1: Tracks with a quality tier, a racing schedule and small post-position biases by surface
        world.tracks = pd.DataFrame({
            'track': [f'T{chr(65 + i)}{chr(65 + (3 * i + 7) % 26)}' for i in range(n_tracks)], 'tier': tiers,
            'days_per_week': W['racing_days_per_week'][:n_tracks],
            'post_bias_dirt': rng.normal(0, params['running']['post_bias_sd'], n_tracks),
            'post_bias_turf': rng.normal(0, params['running']['post_bias_sd'], n_tracks),
            'pool_effect': rng.normal(0, 0.25, n_tracks) + 0.6 * tiers})
        # Step 2: Trainers (heavy-tailed barn sizes, skill, hidden skill and specialties), jockeys, owners, sires
        nt = P['trainers_per_track'] * n_tracks
        barn = rng.pareto(P['trainer_barn_pareto_alpha'], nt) + 1.0
        world.trainers = pd.DataFrame({
            'trainer': np.arange(nt), 'name': _people(rng, nt, 'TR'), 'track': np.repeat(np.arange(n_tracks), P['trainers_per_track']),
            'barn_weight': barn / barn.mean(),
            'skill': rng.normal(0, P['trainer_skill_sd'], nt) + 0.04 * np.log(barn),
            'hidden': rng.normal(0, P['trainer_hidden_sd'], nt),
            'layoff_skill': rng.normal(0, P['trainer_specialty_sd'], nt),
            'turf_skill': rng.normal(0, P['trainer_specialty_sd'], nt),
            'claim_skill': rng.normal(0, P['trainer_specialty_sd'], nt)})
        nj = P['jockeys_per_track'] * n_tracks
        world.jockeys = pd.DataFrame({
            'jockey': np.arange(nj), 'name': _people(rng, nj, 'JK'), 'track': np.repeat(np.arange(n_tracks), P['jockeys_per_track']),
            'skill': rng.normal(0, P['jockey_skill_sd'], nj), 'hidden': rng.normal(0, P['jockey_hidden_sd'], nj)})
        no = int(nt * P['owners_per_trainer'])
        world.owners = pd.DataFrame({'owner': np.arange(no), 'name': _people(rng, no, 'OW'),
                                     'trainer': rng.choice(nt, no, p=barn / barn.sum())})
        ns = P['n_sires']
        world.sires = pd.DataFrame({'sire': np.arange(ns), 'name': _names(rng, ns), 'turf': rng.normal(0, P['sire_surface_sd'], ns),
                                    'route': rng.normal(0, P['sire_distance_sd'], ns),
                                    'popularity': rng.pareto(1.5, ns) + 1})
        # Step 3: Initial horses of mixed ages per track
        for t in range(n_tracks):
            n = P['horses_per_track']
            ages = rng.choice([2, 3, 4, 5, 6, 7, 8], n, p=[0.10, 0.28, 0.26, 0.17, 0.10, 0.06, 0.03])
            world.add_horses(t, n, start_day, ages=ages, warm_start=True)
        return world

    def add_horses(self, track, n, day, ages=None, warm_start=False):
        """
        Foals new horses into the circuit (or creates the initial population when ``warm_start``).

        New horses are maidens with a hidden base ability, preferences inherited partly from their sire, a running style
        and a trainer chosen in proportion to barn size at their home track.
        """
        rng, P, Rn = self.rng, self.params['population'], self.params['running']
        tier = self.tracks.tier.values[track]
        ages = np.full(n, 2) if ages is None else np.asarray(ages)
        tr_local = self.trainers[self.trainers.track == track]
        trainers = rng.choice(tr_local.trainer.values, n, p=(tr_local.barn_weight / tr_local.barn_weight.sum()).values)
        own_pool = self.owners.groupby('trainer').owner.apply(np.array).to_dict()
        owners = np.array([rng.choice(own_pool[tid]) if tid in own_pool else rng.integers(0, len(self.owners)) for tid in trainers])
        sires = rng.choice(len(self.sires), n, p=(self.sires.popularity / self.sires.popularity.sum()).values)
        year_day = day / 365.25
        new = {
            'horse': np.arange(n) + (len(self.horses.get('horse', [])) if self.horses else 0),
            'track': np.full(n, track), 'trainer': trainers, 'owner': owners, 'sire': sires,
            'female': rng.random(n) < P['female_share'],
            'birth': year_day - ages - rng.random(n) * 0.5,
            'ability': rng.normal(Rn['tier_ability_slope'] * tier, Rn['ability_sd'], n)
                       + P.get('trainer_stock_corr', 0.0) * self.trainers.skill.values[trainers],
            'turf_pref': rng.normal(0, Rn['surface_pref_sd'], n) + self.sires.turf.values[sires],
            'best_distance': np.clip(rng.normal(7.0, 1.2, n) + 4 * self.sires.route.values[sires], 5.0, 11.0),
            'early_speed': rng.normal(0, Rn['early_speed_sd'], n),
            'form': np.zeros(n),
            'starts': np.zeros(n, dtype=int) if not warm_start else np.maximum(0, ((ages - 2) * rng.poisson(5, n))).astype(int),
            'wins': np.zeros(n, dtype=int),
            'last_day': np.full(n, -10_000) if not warm_start else day - rng.integers(10, 60, n),
            'ready_day': day + (rng.integers(0, 45, n) if warm_start else rng.integers(20, 90, n)),
            'last_jockey': np.full(n, -1),
            'active': np.ones(n, dtype=bool),
            'name': np.array(_names(rng, n), dtype=object),
        }
        if warm_start:   # older horses have mostly broken their maiden
            new['wins'] = (rng.random(n) < np.clip(0.25 * (ages - 2), 0, 0.9)).astype(int)
        if not self.horses:
            self.horses = new
        else:
            for k in new:
                self.horses[k] = np.concatenate([self.horses[k], new[k]])

    # ---------------------------------------------------------------- state helpers
    def age(self, idx, day):
        return np.floor(day / 365.25 - self.horses['birth'][idx]).astype(int)

    def rating(self, idx, day):
        """Current class rating: base ability + age curve + form (what a trainer uses to place the horse)."""
        Rn = self.params['running']
        age = day / 365.25 - self.horses['birth'][idx]
        return self.horses['ability'][idx] - Rn['age_curvature'] * (age - Rn['age_peak']) ** 2 + self.horses['form'][idx]
