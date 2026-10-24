"""
Simulator parameters.

Every behavior of the simulated racing world is controlled by one nested dictionary. The values below are public,
hand-set defaults chosen to look like North American thoroughbred racing in broad strokes. A calibrated parameter
file (JSON with the same structure, any subset of keys) can override them:

    params = load_params('my_calibrated_params.json')
    params = load_params()            # uses the LYNK_SIM_PARAMS environment variable when it is set

Sections:
- world        circuit size and the racing calendar
- population   horses, trainers, jockeys, owners and sires
- races        card structure, conditions, purses and field sizes
- running      how a race is run: ability, fit, connections, pace, noise, figures and beaten lengths
- careers      time between starts, layoffs, claims and retirement
- crowd        what the betting public believes and its biases
- tote         pool sizes, money arrival by time to post, late money, cycles and takeout
- exotics      exacta, trifecta, superfecta, daily double and Pick-N behavior
"""

import copy
import json
import os

DEFAULTS = {
    'world': {
        'n_tracks': 6,
        'track_tiers': [1.0, 0.6, 0.3, 0.0, -0.3, -0.6],      # quality index per track (higher = bigger purses, better horses)
        'racing_days_per_week': [5, 5, 4, 4, 3, 3],
        'start_date': '2024-01-01',
    },
    'population': {
        'horses_per_track': 2200,
        'trainers_per_track': 140,
        'jockeys_per_track': 35,
        'owners_per_trainer': 6.0,
        'n_sires': 400,
        'trainer_barn_pareto_alpha': 1.1,            # heavy tail: a few barns run a large share of starters
        'trainer_skill_sd': 0.12,                    # trainer effect on the logit-like strength scale
        'trainer_hidden_sd': 0.05,                   # part of trainer skill the crowd does not price
        'trainer_specialty_sd': 0.08,                # layoff / turf / claim specialties
        'jockey_skill_sd': 0.10,
        'jockey_hidden_sd': 0.03,
        'sire_surface_sd': 0.10,
        'sire_distance_sd': 0.08,
        'female_share': 0.42,
        'trainer_stock_corr': 1.2,                   # better barns get better horses (ability += corr * trainer skill)
    },
    'races': {
        'races_per_card_pmf': {'6': 0.05, '7': 0.12, '8': 0.34, '9': 0.26, '10': 0.15, '11': 0.06, '12': 0.02},
        'race_type_share': {'maiden': 0.28, 'claiming': 0.47, 'allowance': 0.19, 'stakes': 0.06},
        'turf_share': {'maiden': 0.18, 'claiming': 0.12, 'allowance': 0.21, 'stakes': 0.30},
        'field_size_pmf': {'5': 0.08, '6': 0.18, '7': 0.21, '8': 0.19, '9': 0.13, '10': 0.10, '11': 0.06, '12': 0.05},
        'dirt_distances': {'5.0': 0.07, '5.5': 0.15, '6.0': 0.30, '6.5': 0.09, '7.0': 0.07, '8.0': 0.16, '8.5': 0.11, '9.0': 0.05},
        'turf_distances': {'5.0': 0.15, '5.5': 0.10, '7.5': 0.05, '8.0': 0.25, '8.5': 0.25, '9.0': 0.15, '11.0': 0.05},
        'log_purse_mean': {'maiden': 10.1, 'claiming': 9.7, 'allowance': 10.5, 'stakes': 11.5},
        'log_purse_sd': {'maiden': 0.6, 'claiming': 0.5, 'allowance': 0.6, 'stakes': 0.8},
        'purse_tier_slope': 0.45,                    # log purse added per unit of track tier
        'class_match_sd': 0.45,                      # how tightly trainers place horses at their level
    },
    'running': {
        'ability_sd': 0.55,
        'tier_ability_slope': 0.35,
        'age_peak': 4.5,
        'age_curvature': 0.035,
        'form_ar': 0.85,                             # form persistence between starts
        'form_sd': 0.18,
        'surface_pref_sd': 0.20,
        'distance_pref_sd': 0.15,
        'early_speed_sd': 1.0,
        'early_speed_value': 0.18,                   # front runners win more often than closers
        'lone_speed_bonus': 0.10,
        'speed_duel_penalty': 0.08,
        'post_bias_sd': 0.04,
        'layoff_effect': -0.08,
        'debut_sd': 0.35,
        'performance_noise_sd': 0.62,                # running-time noise (sets how predictable races are)
        'gumbel_mix': 0.3,                           # 0 = normal noise (strong Harville discount), 1 = Gumbel (plain Harville)
        'figure_mean': 76.0,
        'figure_scale': 11.0,
        'figure_noise_sd': 6.0,
        'lengths_per_unit': 4.5,
    },
    'careers': {
        'gap_log_mean': 3.1,                         # days between starts ~ lognormal
        'gap_log_sd': 0.45,
        'min_gap_days': 7,
        'layoff_probability': 0.11,
        'layoff_mean_days': 150,
        'claim_probability': 0.08,
        'retire_per_start': 0.07,
        'retire_age_slope': 0.02,
        'new_horses_per_track_per_week': 18,
    },
    'crowd': {
        'visible_share': 0.92,                       # share of the true strength signal the public can see
        'public_noise_sd': 0.35,                     # race-specific noise in the public's opinion
        'morning_line_noise_sd': 0.45,
        'ml_anchor': 0.25,                           # weight of the morning line in early money
        'longshot_bias': 0.88,                       # public allocates in proportion to p ** longshot_bias
        'floor_share': 0.004,                        # every horse attracts at least this share of public money
        'informed_noise_sd': 0.15,                   # how close informed late money is to the truth
        'early_noise_sd': 0.6,                       # extra, fading noise in the public's opinion far from post
        'info_learned': {'1800': 0.0, '600': 0.15, '300': 0.25, '60': 0.4, '0': 0.5, 'final': 0.55},  # share of its blind spot the public learns by each offset
        'informed_weight': {'1800': 0.05, '900': 0.08, '600': 0.10, '300': 0.15, '120': 0.22, '60': 0.30,
                            '30': 0.40, '15': 0.45, '0': 0.50, 'final': 0.60},
        'dirichlet_concentration_per_dollar': 0.05,  # smaller pools are noisier
        'dirichlet_min_concentration': 200,
        'late_shock_sd': 0.5,                        # money after the post-time cycle is concentrated on a few horses
        'place_slope': 0.84,                         # log place share ~ intercept + slope * log win share
        'show_slope': 0.71,
        'place_noise_sd': 0.20,
        'show_noise_sd': 0.30,
    },
    'tote': {
        'offsets': [1800, 1200, 900, 600, 300, 180, 120, 90, 60, 30, 15, 0],
        'arrival_at_offset': {'1800': 0.05, '1200': 0.07, '900': 0.09, '600': 0.14, '300': 0.24, '180': 0.33,
                              '120': 0.40, '90': 0.46, '60': 0.55, '30': 0.66, '15': 0.68, '0': 0.72},
        'arrival_log_sd': 0.35,
        'wps_pool_intercept': 2.3,
        'wps_pool_log_purse': 0.77,
        'wps_pool_field': 0.14,
        'wps_pool_sd': 0.7,
        'win_share_of_wps': 0.68,
        'place_share_of_wps': 0.20,
        'wps_share_sd': 0.045,
        'takeout': {'win': 0.17, 'place': 0.17, 'show': 0.17, 'exacta': 0.22, 'trifecta': 0.24, 'superfecta': 0.24,
                    'daily_double': 0.20, 'pick3': 0.22, 'pick4': 0.20, 'pick5': 0.15},
        'breakage': 0.10,
        'minimum_payout': 2.10,
        'post_delay_mean_s': 90,
        'cycle_seconds': 30,
    },
    'exotics': {
        'offered': {'exacta': 0.99, 'trifecta': 0.98, 'superfecta': 0.93, 'daily_double': 0.71, 'pick3': 0.67,
                    'pick4': 0.23, 'pick5': 0.13},
        'pool_over_win': {'exacta': 0.93, 'trifecta': 0.63, 'superfecta': 0.43, 'daily_double': 0.17, 'pick3': 0.23,
                          'pick4': 0.80, 'pick5': 1.10},
        'pool_over_win_log_sd': 0.4,
        'second_leg_longshot_bias': 0.75,            # exacta bettors overweight longshots underneath (exponent < 1)
        'combo_noise_sd': 0.35,
        'harville_discount_second': 0.85,
        'harville_discount_third': 0.70,
        'carryover_share': 0.0,
    },
}


def _merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_params(path=None):
    """
    Returns the full parameter dictionary: the public defaults, overridden by a JSON file when one is given.

    Parameters:
    - path: Optional JSON file. When omitted, the ``LYNK_SIM_PARAMS`` environment variable is used if set.

    Returns:
    - Nested dict with every section of ``DEFAULTS``.
    """
    params = copy.deepcopy(DEFAULTS)
    path = path or os.environ.get('LYNK_SIM_PARAMS')
    if path and os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            _merge(params, json.load(f))
    return params


def save_params(params, path):
    """Writes a parameter dictionary (or any subset of it) to JSON."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=1)
