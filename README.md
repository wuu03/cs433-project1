# EPFL CS-433 Project 1 — MICHD prediction (v7)

NumPy-only pipeline predicting coronary heart disease / myocardial infarction (MICHD)
from the BRFSS 2015 survey. Only NumPy and the Python standard library are used
(no pandas / scikit-learn / external data).

## Result

| | 5-fold OOF F1 | OOF average precision |
|---|---:|---:|
| v5/v6 best single backbone (previous) | 0.4446 | 0.4332 |
| **v7 ensemble (this version)** | **0.4487** | **0.4380** |

Authors: Ruhan Liu, Yun-Chia Shih, Jingru Wang.

`artifacts/submission.csv`: 109,379 rows, 14,580 predicted positives (13.3%),
labels in {-1, +1}, test-ID order preserved. **AIcrowd: F1 0.460, accuracy 0.881** (best so far).
Previous AIcrowd F1: v1 0.432, v2 0.436, v3 0.454, v4 0.444, v5 0.452, v6 0.452.

## Model

1. `features.py` — codebook cleaning (`survey_codes.json`: "don't know/refused" → -1,
   "none" (88) → 0), removal of 14 sampling/administration fields, and derived
   features: exact age (recovered from `MAXVO2_`, which is computed from un-capped
   age), alcohol days/month, number of chronic conditions, activity limitations,
   unknown answers and missing answers.
2. `gbdt.py` — histogram gradient-boosted trees written from scratch: 63 quantile
   bins + a dedicated missing-value bin, leaf-wise growth (31 leaves), Newton leaf
   values with L2, histogram subtraction, row bagging or GOSS sampling, feature
   subsampling.
3. `nn.py` — embedding MLP: every (feature, bin) pair has a learned 6-d embedding,
   followed by a 256-64 ReLU MLP with dropout, Adam and cosine learning-rate decay,
   positive-class weight 4.
4. `ensemble.py` — two GBDTs (`gbA` bagging, `gbB` GOSS + positive weight 2) and five
   NN seeds, trained on **all** labelled rows. Log-odds are de-shifted by
   log(class weight) and blended 0.6 trees / 0.4 networks. The decision threshold is
   the median of 50 bootstrap F1-optimal thresholds on the out-of-fold blend.
5. `cv.py` — 5-fold stratified CV of each member (out-of-fold scores in
   `artifacts/cv_oof.npz`).

## Reproduce

```sh
python run.py           # inference with artifacts/model.npz -> artifacts/submission.csv (~1 min)
python run.py --train   # retrain all members on the full training set (~30 min), then predict
python run.py --cv      # recompute out-of-fold predictions (slow, 1-2 h), then retrain
```

Data are expected in `data/dataset/` (`x_train.csv`, `y_train.csv`, `x_test.csv`);
use `--data-dir` otherwise. Set `OPENBLAS_NUM_THREADS=1` to avoid thread oversubscription.

## Files

- `implementations.py` — the six required methods (unchanged); `helpers.py` — course helpers.
- `run.py`, `data.py`, `features.py`, `gbdt.py`, `nn.py`, `ensemble.py`, `cv.py` — v7 pipeline.
- `artifacts/model.npz` (saved models + threshold), `artifacts/cv_oof.npz` (OOF scores).
- `docs/v7_analysis.md` — bottleneck analysis, all experiments (kept and rejected).
- `baselines.py` — linear baselines from the required methods under the same CV
  (ridge F1 0.4290, logistic GD 0.3114, logistic with 1/4 of negatives 0.4224).
- `report.tex` / `report.pdf` — 2-page report for v7; `make_figure.py` draws `report_fig.pdf`.
  Build: `TEXINPUTS=.:./latex-example-paper//: pdflatex report.tex` (twice).

Older versions (v1–v6 code and artifacts) are in the git history (`proj 1 v1` … `proj 1 v6`).
