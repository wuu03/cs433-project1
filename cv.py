"""5-fold stratified cross-validation of every ensemble member (out-of-fold log-odds).

Writes artifacts/cv_oof.npz, used by run.py --train to choose blend weights / threshold.
Runtime is long (about 1-2 h on a laptop) because all trees are grown in NumPy.
"""

import numpy as np
from ensemble import GB_MEMBERS, NN_PARAMS, NN_POS_WEIGHT, NN_SEEDS, SEED, deshift
from gbdt import GBDT
from nn import EmbedMLP


def stratified_folds(y, k=5, seed=SEED):
    rng = np.random.default_rng(seed)
    folds = np.empty(len(y), int)
    for c in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == c))
        folds[idx] = np.arange(len(idx)) % k
    return folds


def scores(y, s):
    """Best-threshold F1 and average precision (threshold-free ranking quality)."""
    o = np.argsort(-s)
    ys = y[o]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    return float((2 * tp / (k + ys.sum())).max()), float(np.mean((tp / k)[ys == 1]))


def run_cv(B_rows, y, out_path, log=print):
    folds = stratified_folds(y)
    Bt = np.ascontiguousarray(B_rows.T)
    oof = {}
    for name, p in GB_MEMBERS.items():
        oof[name] = np.zeros(len(y))
        for k in range(5):
            tr = folds != k
            m = GBDT(seed=1000 * k, **p).fit(Bt[:, tr], y[tr])
            oof[name][~tr] = deshift(
                m.decision_function(Bt[:, ~tr]), p.get("pos_weight", 1.0)
            )
        log(name, "F1 %.4f AP %.4f" % scores(y, oof[name]))
    for s in NN_SEEDS:
        name = f"nn{s}"
        oof[name] = np.zeros(len(y))
        for k in range(5):
            tr = folds != k
            m = EmbedMLP(seed=k + 100 * s, **NN_PARAMS).fit(
                B_rows[tr], y[tr], pos_weight=NN_POS_WEIGHT
            )
            oof[name][~tr] = deshift(m.decision_function(B_rows[~tr]), NN_POS_WEIGHT)
        log(name, "F1 %.4f AP %.4f" % scores(y, oof[name]))
    np.savez(out_path, y=y, folds=folds, **oof)
    return oof
