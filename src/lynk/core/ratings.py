"""
Online entity ratings for multi-outcome events (Kalman / Glicko style, jointly estimated).

Use case: "does the market misprice this trainer, this jockey, this owner, this trainer at this track?". Every
event is a conditional logit:

    score_i = offset_scale * offset_i + sum over entity types k of theta_k[entity_k(i)]

where the offset is a strong baseline (for example the log of the closing-price probability) and each theta is a
Gaussian random walk with prior variance v0_k and daily drift q_k. Days are processed in order: every event on day D
is predicted with the ratings as of the end of day D-1 (those predictions are the leakage-safe FEATURES), then the
ratings are updated with day D's results by one diagonal Newton (Laplace) step per entity type. Types are updated
one after another so later types only explain what earlier ones left over, which keeps correlated entities (a
trainer and that trainer at one track) from double counting the same evidence.

References: Elo (1978); Kalman (1960); Glickman (1999).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lynk.core.probability import group_index


@dataclass
class EntityType:
    """
    One family of ratings.

    Attributes:
    - name: Short name used in output columns, e.g. 'trainer' or 'trainer_track'.
    - columns: Columns whose values together identify the entity, e.g. ['trainer'] or ['trainer', 'track'].
    - prior_variance: v0, how far a rating may move from zero (on the logit scale). Small = heavy shrinkage.
    - drift_per_day: q, variance added per day of inactivity (0 = durable tendencies, > 0 = ratings can trend).
    - condition: Optional column; the entity only exists on rows where this column is not null
                 (e.g. 'layoff_flag' for a trainer-off-a-layoff rating).
    """
    name: str
    columns: list
    prior_variance: float = 0.002
    drift_per_day: float = 0.0
    condition: str = None


class OnlineRatings:
    """
    Fits ratings day by day and returns leakage-safe features for every row.

    Example:
        ratings = OnlineRatings([EntityType('trainer', ['trainer']), EntityType('jockey', ['jockey'])])
        features = ratings.run(frame, event_col='race_id', date_col='date', outcome_col='won', offset_col='log_p')
    """

    def __init__(self, entity_types, offset_scale=1.0):
        self.entity_types = list(entity_types)
        self.offset_scale = offset_scale

    def _codes(self, frame, et):
        key = frame[et.columns[0]].astype(str)
        for col in et.columns[1:]:
            key = key + '|' + frame[col].astype(str)
        valid = frame[et.columns[0]].notna()
        if et.condition is not None:
            valid &= frame[et.condition].notna()
        codes, uniques = pd.factorize(key.where(valid), use_na_sentinel=True)
        return codes.astype(np.int64), len(uniques)

    def run(self, frame, event_col, date_col, outcome_col, offset_col=None, score_from=None):
        """
        Processes the frame chronologically.

        Parameters:
        - frame: DataFrame with one row per outcome (e.g. per horse). Sorted internally by date and event.
        - event_col: Event id column.
        - date_col: Date column (ratings update once per day).
        - outcome_col: 1 for the outcome that happened, else 0.
        - offset_col: Optional baseline log-probability column. Without it the ratings measure raw strength;
                      with the closing price as offset they measure mispricing.
        - score_from: Optional date; log-loss diagnostics are only accumulated from this date.

        Returns:
        - DataFrame aligned to the input index with, per entity type, '<name>_mean', '<name>_sd', '<name>_n',
          plus 'rating_total' (sum of all means). Attribute ``diagnostics`` holds the log-loss gain over the offset.
        """
        # Step 1: Sort by time and event, encode every entity type as integer codes
        order = np.lexsort((frame[event_col].values, pd.to_datetime(frame[date_col]).values))
        f = frame.iloc[order]
        day = pd.to_datetime(f[date_col]).values.astype('datetime64[D]').astype(np.int64)
        event = pd.factorize(f[event_col])[0]
        y = f[outcome_col].values.astype(float)
        offset = np.zeros(len(f)) if offset_col is None else self.offset_scale * f[offset_col].values.astype(float)
        codes = {et.name: self._codes(f, et) for et in self.entity_types}
        mean = {k: np.zeros(n) for k, (_, n) in codes.items()}
        var = {et.name: np.full(codes[et.name][1], et.prior_variance) for et in self.entity_types}
        last_seen = {k: np.full(n, -1, np.int64) for k, (_, n) in codes.items()}
        count = {k: np.zeros(n, np.int64) for k, (_, n) in codes.items()}
        out = {f'{et.name}_{s}': np.full(len(f), np.nan) for et in self.entity_types for s in ('mean', 'sd', 'n')}
        total = np.zeros(len(f))
        ll_base = ll_model = n_events = 0.0
        score_day = None if score_from is None else np.datetime64(pd.Timestamp(score_from), 'D').astype(np.int64)
        day_bounds = np.r_[np.flatnonzero(np.r_[True, day[1:] != day[:-1]]), len(f)]

        for a, b in zip(day_bounds[:-1], day_bounds[1:]):
            starts, gid = group_index(event[a:b])
            score = offset[a:b].copy()
            # Step 2: Predict today's events with yesterday's ratings (these are the features)
            for et in self.entity_types:
                c = codes[et.name][0][a:b]
                ok = c >= 0
                e = c[ok]
                gap = np.where(last_seen[et.name][e] >= 0, day[a] - last_seen[et.name][e], 0)
                v_now = np.minimum(var[et.name][e] + et.drift_per_day * gap, et.prior_variance)
                var[et.name][e] = v_now
                last_seen[et.name][e] = day[a]
                score[ok] += mean[et.name][e]
                out[f'{et.name}_mean'][a:b][ok] = mean[et.name][e]
                out[f'{et.name}_sd'][a:b][ok] = np.sqrt(v_now)
                out[f'{et.name}_n'][a:b][ok] = count[et.name][e]
            total[a:b] = score - offset[a:b]
            yy = y[a:b]
            totals = np.add.reduceat(yy, starts)
            target = yy / np.where(totals > 0, totals, 1.0)[gid]
            has_result = (totals > 0)[gid]
            if score_day is None or day[a] >= score_day:
                p_model = self._softmax(score, starts, gid)
                p_base = self._softmax(offset[a:b], starts, gid)
                ll_model -= np.sum(target * np.log(np.clip(p_model, 1e-12, 1)))
                ll_base -= np.sum(target * np.log(np.clip(p_base, 1e-12, 1)))
                n_events += float((totals > 0).sum())
            # Step 3: Update each entity type with today's results (one Newton step, sequential)
            for et in self.entity_types:
                c = codes[et.name][0][a:b]
                ok = (c >= 0) & has_result
                if not ok.any():
                    continue
                p = self._softmax(score, starts, gid)
                grad, hess = target - p, p * (1 - p)
                ue, inv = np.unique(c[ok], return_inverse=True)
                G, H = np.bincount(inv, grad[ok]), np.bincount(inv, hess[ok])
                new_var = 1.0 / (1.0 / var[et.name][ue] + H)
                step = new_var * G
                mean[et.name][ue] += step
                var[et.name][ue] = new_var
                count[et.name][ue] += np.bincount(inv)
                score[ok] += step[inv]

        # Step 4: Return features in the caller's original row order
        result = pd.DataFrame(out)
        result['rating_total'] = total
        result.index = f.index
        self.diagnostics = dict(events=n_events, logloss_offset=ll_base / max(n_events, 1),
                                logloss_with_ratings=ll_model / max(n_events, 1),
                                gain=(ll_base - ll_model) / max(n_events, 1))
        return result.loc[frame.index]

    @staticmethod
    def _softmax(score, starts, gid):
        shifted = score - np.maximum.reduceat(score, starts)[gid]
        w = np.exp(shifted)
        return w / np.add.reduceat(w, starts)[gid]
