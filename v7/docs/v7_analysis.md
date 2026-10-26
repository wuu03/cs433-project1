# v7: why F1 was stuck near 0.45, and what moved it

All numbers: 5-fold stratified CV (seed 2026) on the 328,135 labelled rows, natural
8.83% positive rate. "F1" = best-threshold out-of-fold F1, "AP" = average precision
(threshold-free; more stable than F1 for comparing models).

## Bottlenecks

1. **Information, not model capacity.** A strong, tuned off-the-shelf GBDT (LightGBM,
   2,500 trees, used *only* as an outside yardstick during this analysis, never in the
   project code) reaches F1 0.4477 / AP 0.4365 / AUC 0.870. v6's best backbone was at
   0.4446 / 0.4332, so v1–v6 were within ~0.003 F1 of what any single model extracts.
   Positives and negatives overlap heavily (self-reported diagnosis, 8.8% prevalence).
2. **Leaderboard noise is as large as the gains being chased.** Bootstrapping the OOF
   predictions over random 109,379-row test sets gives an F1 standard deviation of
   0.0034. v3 (0.454) vs v6 (0.452) is noise; the leaderboard has so far scored
   ~0.008 above local OOF F1.
3. **Data quantity still matters.** Learning curve (fold 0): 25% of a training fold
   AP 0.4245 → 50% 0.4306 → 100% 0.4362 (LightGBM), and with our NumPy gbB 0.3898 → 0.4263 →
   0.4353 (fixed hyper-parameters), i.e. +0.006 to +0.009 AP per doubling. Training
   final models on all rows (not 80% CV folds) is a cheap, real gain.
4. **Previous pipeline drops information**: columns with >80% missing were removed
   (routing/missingness is informative, e.g. all 773 respondents with a HAREHAB1
   answer are positive), and the representation was a hand-built linear/tree mix.
5. **Complexity.** ~90 scripts and 2.2 GB of artifacts made the system hard to reason
   about; v4–v6 added calibration layers that did not move the leaderboard.

## Plan that was executed

Replace the stack by a small, strong, diverse ensemble written in NumPy, keep every
column (missing = its own bin), train on all rows, and blend only a few global weights.

## Experiments (kept ✔ / rejected ✘)

| Model | F1 | AP | |
|---|---:|---:|---|
| v6 backbone (previous best single) | 0.4446 | 0.4332 | |
| NumPy GBDT gbA, 300 / 450 / 600 trees | 0.4464 / 0.4475 / 0.4474 | 0.4346 / 0.4356 / 0.4360 | ✔ 600 |
| gbA with positive weight 3 | 0.4471 | 0.4354 | ✘ no gain |
| gbB: GOSS, lr 0.02, pos. weight 2, 1000 trees | 0.4481 | 0.4352 | ✔ |
| GBDT 63 leaves, λ=20 | 0.4450 | 0.4313 | ✘ |
| GBDT 24 leaves, GOSS, 1100 trees | 0.4483 | 0.4351 | ✘ no blend gain |
| Embedding MLP, no class weight | 0.4435 | 0.4268 | |
| + dropout 0.25, embedding L2, dim 6 | 0.4438 | 0.4290 | |
| + positive weight 3 / 4 / 6 | 0.4462 / 0.4462 / 0.4464 | 0.4322 / 0.4324 / 0.4326 | ✔ 4 |
| MLP + ordinal bin input | 0.4446 | 0.4302 | ✘ |
| MLP + 10% input-feature dropout | 0.4455 | 0.4320 | ✘ |
| MLP bag of 3 / 5 seeds | 0.4467 / 0.4471 | 0.4337 / 0.4339 | ✔ 5 |
| **Blend 0.6·(gbA+gbB)/2 + 0.4·MLP bag** | **0.4487** | **0.4380** | ✔ final |
| + third GBDT | 0.4485 | 0.4381 | ✘ |
| Pseudo-labelled test rows (teacher = fold-k ensemble), folds 0–1 | +0.0000 / +0.0002 in blend | | ✘ only distils the ensemble |
| Linear baselines (required methods): ridge / logistic GD / logistic, 1/4 negatives | 0.4290 / 0.3114 / 0.4224 | 0.4119 / 0.2693 / 0.3907 | baselines |
| gbB on raw columns (no feature engineering) | 0.4477 | 0.4336 | feature eng. +0.0016 AP |
| Engineered features vs raw (LightGBM yardstick) | 0.4473 vs 0.4464 | AUC 0.8700 vs 0.8696 | ✔ small |

Blend weights 0.5–0.7 are equivalent (F1 0.4484–0.4489); 0.6 was used.
Calibration transfer check: the OOF-selected threshold yields 13.2% positives on OOF
and 13.3% on the test set (full-data models), so the threshold transfers.

## Leaderboard result

v7 scored **F1 0.460, accuracy 0.881** on AIcrowd (v6: 0.452 / 0.887).

## Expected leaderboard effect (written before submission)

v7 improves OOF AP by +0.0048 and OOF F1 by ~+0.004 over the v5/v6 backbone, plus the
full-data refit (≈ +0.002 AP from the learning curve). With the historical ~+0.008
leaderboard offset this points to roughly 0.456–0.460; the test-set noise (±0.003)
is of the same size, so the exact score can only be known after submission.
