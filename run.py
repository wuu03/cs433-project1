"""Reproduces artifacts/submission.csv.

  python run.py            inference with the saved models in artifacts/model.npz (fast)
  python run.py --train    retrain all ensemble members on the full training set, then predict
  python run.py --cv       (slow) recompute the 5-fold out-of-fold predictions artifacts/cv_oof.npz

Only NumPy and the Python standard library are used.
"""

import argparse
import time
from pathlib import Path
import numpy as np
from data import load
from features import engineer
from gbdt import (
    Binner,
    binner_to_arrays,
    binner_from_arrays,
    gbdt_to_arrays,
    gbdt_from_arrays,
)
from nn import nn_to_arrays, nn_from_arrays
from ensemble import (
    GB_MEMBERS,
    NN_SEEDS,
    NN_POS_WEIGHT,
    train_full,
    predict,
    blend,
    best_threshold,
    deshift,
)
from helpers import create_csv_submission

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def oof_threshold():
    d = np.load(ART / "cv_oof.npz")
    gb = {k: d[k] for k in GB_MEMBERS}
    nn = [d[f"nn{s}"] for s in NN_SEEDS if f"nn{s}" in d.files]
    score = blend(gb, nn)
    t, f1 = best_threshold(d["y"], score)
    return t, f1, float(np.mean(score >= t))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "dataset"))
    ap.add_argument("--out", default=str(ART / "submission.csv"))
    args = ap.parse_args()
    ART.mkdir(exist_ok=True)
    train = args.train or args.cv or not (ART / "model.npz").exists()
    log("loading data")
    x, y, xt, test_ids, names = load(args.data_dir, train=train)
    ft, _ = engineer(xt, names)
    if train:
        f, _ = engineer(x, names)
        del x
        binner = Binner().fit(f)
        B = np.ascontiguousarray(binner.transform(f).T)
        del f
        if args.cv:
            from cv import run_cv

            run_cv(B, y, ART / "cv_oof.npz", log=log)
        threshold, f1, rate = oof_threshold()
        log(
            f"OOF blend: F1 {f1:.4f}, threshold {threshold:.4f}, positive rate {rate:.4f}"
        )
        models = train_full(B, y, log=log)
        arrays = binner_to_arrays(binner)
        for k in GB_MEMBERS:
            arrays.update(gbdt_to_arrays(models[k], k + "_"))
        for s in NN_SEEDS:
            arrays.update(nn_to_arrays(models[f"nn{s}"], f"nn{s}_"))
        arrays["threshold"] = np.array(threshold)
        np.savez_compressed(ART / "model.npz", **arrays)
    d = np.load(ART / "model.npz")
    binner = binner_from_arrays(d)
    models = {k: gbdt_from_arrays(d, k + "_") for k in GB_MEMBERS}
    models.update({f"nn{s}": nn_from_arrays(d, f"nn{s}_") for s in NN_SEEDS})
    Bt = np.ascontiguousarray(binner.transform(ft).T)
    score, _, _ = predict(models, Bt)
    threshold = float(d["threshold"])
    pred = np.where(score >= threshold, 1, -1)
    create_csv_submission(test_ids, pred, args.out)
    log(
        f"wrote {args.out}: {len(pred)} rows, {int((pred == 1).sum())} positives ({np.mean(pred == 1):.4f})"
    )


if __name__ == "__main__":
    main()
