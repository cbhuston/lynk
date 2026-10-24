"""
Event-level probability models.

``GroupSoftmaxObjective`` turns LightGBM into a conditional logit model: the trees learn a strength score per outcome
and the loss is the multinomial log-loss inside each event (Bolton and Chapman, 1986; Benter, 1994). Passing a
baseline as ``init_score`` (for example the log of market probabilities) makes the trees learn CORRECTIONS to that
baseline instead of rediscovering it, which is the core design idea of the Lynk models.

LightGBM is an optional dependency: ``pip install lynk[models]``.
"""

import numpy as np

from lynk.core.probability import group_index, group_softmax


class GroupSoftmaxObjective:
    """
    Custom LightGBM objective: softmax cross-entropy within each event.

    Gradient for outcome i in an event: p_i - y_i. Hessian: p_i * (1 - p_i) (the diagonal of the softmax Hessian).
    """

    def __init__(self, group_ids, outcomes):
        self.starts, self.gid = group_index(group_ids)
        y = np.asarray(outcomes, dtype=float)
        totals = np.add.reduceat(y, self.starts)
        self.target = y / np.where(totals > 0, totals, 1.0)[self.gid]

    def __call__(self, raw_scores, dataset):
        shifted = raw_scores - np.maximum.reduceat(raw_scores, self.starts)[self.gid]
        w = np.exp(shifted)
        p = w / np.add.reduceat(w, self.starts)[self.gid]
        return p - self.target, np.maximum(p * (1 - p), 1e-9)


def fit_group_softmax(X, outcomes, group_ids, X_valid=None, outcomes_valid=None, group_ids_valid=None,
                      init_score=None, init_score_valid=None, params=None, num_rounds=2000, early_stopping=100):
    """
    Trains a LightGBM conditional logit model.

    Parameters:
    - X, outcomes, group_ids: Training features, 0/1 outcomes and event ids (rows of an event contiguous).
    - X_valid, outcomes_valid, group_ids_valid: Optional early-stopping set (must be LATER in time than training).
    - init_score, init_score_valid: Optional baseline log-probabilities; the model learns corrections to them.
    - params: LightGBM parameters (learning_rate, num_leaves, min_data_in_leaf, ...).
    - num_rounds: Maximum boosting rounds.
    - early_stopping: Rounds without improvement on the early-stopping set before stopping.

    Returns:
    - Trained ``lightgbm.Booster``.
    """
    import lightgbm as lgb

    base = dict(learning_rate=0.03, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8,
                bagging_freq=1, lambda_l2=10.0, verbose=-1)
    base.update(params or {})
    base['objective'] = GroupSoftmaxObjective(group_ids, outcomes)
    train = lgb.Dataset(np.asarray(X, dtype=np.float32), label=np.asarray(outcomes, dtype=float), init_score=init_score,
                        free_raw_data=False)
    callbacks, valid_sets = [], []
    if X_valid is not None:
        valid = lgb.Dataset(np.asarray(X_valid, dtype=np.float32), label=np.asarray(outcomes_valid, dtype=float),
                            init_score=init_score_valid, reference=train)
        starts_v, gid_v = group_index(group_ids_valid)

        def group_logloss(raw, data):
            p = group_softmax(raw, group_ids_valid)
            y = data.get_label()
            totals = np.add.reduceat(y, starts_v)
            target = y / np.where(totals > 0, totals, 1.0)[gid_v]
            return 'group_logloss', float(-np.sum(target * np.log(np.clip(p, 1e-12, 1))) / max((totals > 0).sum(), 1)), False

        valid_sets = [valid]
        base['metric'] = 'None'
        callbacks = [lgb.early_stopping(early_stopping, verbose=False)]
        return lgb.train(base, train, num_rounds, valid_sets=valid_sets, feval=group_logloss, callbacks=callbacks)
    return lgb.train(base, train, num_rounds)


def predict_group_softmax(booster, X, group_ids, init_score=None):
    """
    Probabilities from a trained group-softmax booster (raw scores + optional baseline, softmaxed per event).
    """
    raw = booster.predict(np.asarray(X, dtype=np.float32), raw_score=True, num_iteration=booster.best_iteration or None)
    if init_score is not None:
        raw = raw + np.asarray(init_score, dtype=float)
    return group_softmax(raw, group_ids)
