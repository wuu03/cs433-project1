"""Linear baselines built from the six required methods (implementations.py), evaluated
with the same 5-fold stratified CV as the ensemble (python baselines.py)."""
import sys
from pathlib import Path
import numpy as np
from data import load
from features import engineer
from cv import stratified_folds, scores
from implementations import ridge_regression, reg_logistic_regression, logistic_regression

ROOT = Path(__file__).resolve().parent


def design(f_train, f_eval):
    """Training-fold statistics only: clip to 1st/99th percentile, median-impute, standardise,
    plus a missing-value indicator per column that has missing values."""
    lo, hi = np.nanpercentile(f_train, [1, 99], axis=0)
    med = np.nanmedian(f_train, axis=0)
    miss_cols = np.isnan(f_train).any(0)
    def make(f):
        m = np.isnan(f)
        z = np.clip(np.where(m, med, f), lo, hi)
        return z, m[:, miss_cols].astype(np.float64)
    z, m = make(f_train)
    mu, sd = z.mean(0), z.std(0) + 1e-9
    keep = sd > 1e-6
    build = lambda f: (lambda zm: np.column_stack([np.ones(len(f)), ((zm[0] - mu) / sd)[:, keep], zm[1]]))(make(f))
    return build(f_train), build(f_eval)


def main(data_dir=ROOT / "data" / "dataset", out=ROOT / "artifacts" / "baselines_oof.npz"):
    x, y, _, _, names = load(data_dir)
    f, _ = engineer(x, names); del x
    folds = stratified_folds(y)
    oof = {k: np.zeros(len(y)) for k in ("ridge", "logistic", "logistic_balanced")}
    for k in range(5):
        tr = folds != k
        a, b = design(f[tr], f[~tr])
        yt = y[tr].astype(np.float64)
        w, _ = ridge_regression(yt, a, 1e-4)
        oof["ridge"][~tr] = b @ w
        w, _ = reg_logistic_regression(yt, a, 1e-5, np.zeros(a.shape[1]), 300, 0.5)
        oof["logistic"][~tr] = b @ w
        # class balance by duplicating positives (weight 4) within the required GD routine
        pos = np.flatnonzero(yt == 1); idx = np.concatenate([np.arange(len(yt)), np.repeat(pos, 3)])
        w, _ = logistic_regression(yt[idx], a[idx], np.zeros(a.shape[1]), 300, 0.5)
        oof["logistic_balanced"][~tr] = b @ w
        print("fold", k, flush=True)
    for name, s in oof.items():
        print(name, "F1 %.4f AP %.4f" % scores(y, s))
    np.savez(out, **oof)


if __name__ == "__main__":
    main(*sys.argv[1:])
