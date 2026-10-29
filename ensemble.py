"""Model configuration, full-data training, blending and decision threshold.

Members (all NumPy-only, see gbdt.py / nn.py):
  gbA  histogram GBDT, 31 leaves, row bagging            (CV: F1 0.4474, AP 0.4360)
  gbB  histogram GBDT with GOSS sampling, positive weight 2 (CV: F1 0.4481, AP 0.4352)
  nn*  embedding MLPs, positive weight 4, 5 random seeds   (CV bag of 3: F1 0.4467, AP 0.4337)
The blend averages member log-odds after removing the known shift log(pos_weight)
introduced by class weighting, then mixes trees and networks with weight W_TREES.
"""

import numpy as np
from gbdt import Binner, GBDT
from nn import EmbedMLP

SEED = 2026
GB_MEMBERS = {
    # name: (params, number of trees used in CV, full-data multiplier applied below)
    "gbA": dict(
        n_trees=600,
        lr=0.03,
        num_leaves=31,
        min_child=100,
        lam=5.0,
        subsample=0.8,
        colsample=0.5,
    ),
    "gbB": dict(
        n_trees=1000,
        lr=0.02,
        num_leaves=31,
        min_child=150,
        lam=5.0,
        goss=(0.2, 0.1),
        pos_weight=2.0,
        colsample=0.4,
    ),
}
NN_PARAMS = dict(
    epochs=4, dropout=0.25, l2_emb=1e-4, dim=6, hidden=(256, 64), lr=2e-3, batch=1024
)
NN_POS_WEIGHT = 4.0
NN_SEEDS = [0, 1, 2, 3, 4]
FULL_DATA_TREE_FACTOR = (
    1.15  # 25% more rows than a CV fold -> slightly more boosting rounds
)
W_TREES = 0.6  # selected on out-of-fold predictions (0.5-0.7 are equivalent)


def blend(gb_margins, nn_margins):
    """gb_margins: dict name->log-odds, nn_margins: list of log-odds (already de-shifted)."""
    trees = np.mean([gb_margins[k] for k in GB_MEMBERS], axis=0)
    nets = np.mean(nn_margins, axis=0)
    return W_TREES * trees + (1 - W_TREES) * nets


def deshift(margin, pos_weight):
    """Class weight w multiplies the odds by w; subtract log(w) to restore the population scale."""
    return margin - np.log(pos_weight)


def best_threshold(y, score, n_boot=50, seed=SEED):
    """F1-optimal cut-off, smoothed as the median of bootstrap optima (less noisy than one argmax)."""

    def opt(yy, ss):
        o = np.argsort(-ss)
        ys = yy[o]
        tp = np.cumsum(ys)
        f1 = 2 * tp / (np.arange(1, len(ys) + 1) + ys.sum())
        i = int(np.argmax(f1))
        return ss[o][i], f1[i]

    rng = np.random.default_rng(seed)
    cuts = [
        opt(y[i], score[i])[0]
        for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))
    ]
    return float(np.median(cuts)), opt(y, score)[1]


def train_full(B_rows, y, log=print):
    """Trains every member on all labelled rows. B_rows: rows x features uint8 bins."""
    Bt = np.ascontiguousarray(B_rows.T)
    models = {}
    for name, p in GB_MEMBERS.items():
        p = dict(p)
        p["n_trees"] = int(round(p["n_trees"] * FULL_DATA_TREE_FACTOR))
        log(f"training {name} ({p['n_trees']} trees)")
        models[name] = GBDT(seed=SEED, **p).fit(Bt, y)
    for s in NN_SEEDS:
        log(f"training nn seed {s}")
        models[f"nn{s}"] = EmbedMLP(seed=SEED + s, **NN_PARAMS).fit(
            B_rows, y, pos_weight=NN_POS_WEIGHT
        )
    return models


def predict(models, B_rows):
    Bt = np.ascontiguousarray(B_rows.T)
    gb = {
        k: deshift(
            models[k].decision_function(Bt), GB_MEMBERS[k].get("pos_weight", 1.0)
        )
        for k in GB_MEMBERS
    }
    nn = [
        deshift(models[f"nn{s}"].decision_function(B_rows), NN_POS_WEIGHT)
        for s in NN_SEEDS
    ]
    return blend(gb, nn), gb, nn
