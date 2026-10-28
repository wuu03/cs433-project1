# EPFL CS-433 Project 1 — MICHD Prediction

Prediction of self-reported coronary heart disease and myocardial infarction
(MICHD) from the BRFSS 2015 survey. The repository contains the final ensemble,
pretrained weights, training and evaluation scripts, and the six required course
method implementations. Model training and inference use NumPy and the Python
standard library.

**Authors:** Ruhan Liu, Yun-Chia Shih, Jingru Wang.

## Results

| Metric | Final ensemble |
|---|---:|
| 5-fold OOF best-threshold F1 | 0.4487 |
| OOF average precision | 0.4380 |
| AIcrowd F1 | 0.460 |
| AIcrowd accuracy | 0.881 |

The saved submission contains 109,379 predictions, including 14,580 positive
predictions (13.3%). Labels are `-1` and `+1`, with IDs in test-file order.

## Setup and data

From the repository root, install the runtime dependency:

```sh
python -m pip install -r requirements.txt
```

Download the competition data from
[AIcrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1)
and place the three files at these paths:

| File | Contents |
|---|---|
| `data/dataset/x_train.csv` | Training IDs and features |
| `data/dataset/y_train.csv` | Training IDs and labels |
| `data/dataset/x_test.csv` | Test IDs and features |

The `--data-dir` option supports another directory containing these filenames.

## Reproduce the submission

Run from the repository root:

```sh
python run.py
```

This loads `artifacts/model.npz`, including the saved binning parameters,
fitted model parameters and decision threshold, and writes predictions to
`artifacts/submission.csv`. Ensemble combination weights are configured in
`ensemble.py`. The default inference path reads `x_test.csv`.

To choose another data directory or output file:

```sh
python run.py --data-dir /path/to/dataset --out artifacts/reproduced_submission.csv
```

## Training and evaluation

```sh
python run.py --train
python run.py --cv
python baselines.py
```

- `--train` trains all ensemble members on the full labelled dataset and selects
  a threshold using the saved OOF scores in `artifacts/cv_oof.npz`, then generates
  predictions.
- `--cv` recomputes 5-fold stratified OOF predictions, selects a threshold, trains
  the full-data ensemble, and generates predictions.
- `baselines.py` evaluates ridge regression and logistic regression under the
  same fold assignment. Logistic runs use 300 gradient-descent steps.

The training commands update the model and submission artifacts. The `--cv`
command also updates the saved OOF scores. Training configuration is defined in
`ensemble.py`.

OOF scores support model selection and threshold search. For the tree and neural
models, feature-bin boundaries are fitted on all labelled feature rows before
cross-validation, so validation-fold feature distributions contribute to the
shared boundaries. Reported OOF F1 uses the F1-maximising cutoff; submission
predictions use the median of optimal cutoffs from 50 bootstrap OOF resamples.

## Final model

1. **Features:** survey-code cleaning, removal of 14 administrative fields, and
   derived age, alcohol-frequency, chronic-condition, activity-limitation and
   response-count features, giving 314 inputs.
2. **GBDT:** two histogram-based gradient-boosted tree configurations with
   leaf-wise growth, up to 31 leaves, missing-value handling, and row and feature
   sampling.
3. **Neural networks:** five embedding MLPs, each using 6-dimensional feature-bin
   embeddings, 256–64 hidden layers, dropout, Adam and positive-class weighting.
4. **Ensemble:** class-weight-adjusted log-odds are averaged within each model
   family and combined with weights of 0.6 for trees and 0.4 for networks.
   Submission members are fitted on all labelled rows, with 15% more GBDT trees
   than the CV configurations.

## Repository contents

| Path | Purpose |
|---|---|
| `run.py` | Inference, full-data training and CV entry point |
| `implementations.py` | Six required course methods |
| `helpers.py`, `data.py` | Submission CSV generation and data loading |
| `features.py`, `survey_codes.json` | Feature engineering and survey-code rules |
| `gbdt.py`, `nn.py` | NumPy model implementations |
| `ensemble.py`, `cv.py` | Model configuration, blending, thresholds and CV |
| `baselines.py` | Linear baseline evaluation |
| `artifacts/model.npz` | Saved binning parameters, fitted members and threshold |
| `artifacts/cv_oof.npz` | OOF labels, folds and member scores |
| `artifacts/submission.csv` | Final competition predictions |
| `docs/` | Feature-processing notes and model analysis |
| `report.pdf`, `report.tex`, `report_fig.pdf` | Report and supporting figure |
| `make_figure.py` | Report figure generation |
| `requirements.txt` | Runtime dependency |

## Course tests and report

Run the latest public tests using the environment and commands in the course
[grading instructions](https://github.com/epfml/ML_course/tree/main/projects/project1/grading_tests).

Compile the report with a LaTeX installation providing the `IEEEtran` class:

```sh
pdflatex report.tex
pdflatex report.tex
```
