# Survey-code handling in the follow-up experiment

Source: [CDC 2015 BRFSS codebook](https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf), linked by the course's dataset-background page. This is documentation for the supplied dataset, not an external dataset. No source frequencies, labels for additional people, external observations or external model code enter training.

`survey_codes.json` records 240 field-specific mappings, the source page numbers, the source PDF hash and the mapping version. The mapping is checked in; reproducing training or inference needs no download. Source labels explicitly identifying unknown/not-sure/refused/missing responses supply the nonresponse codes. A code is never replaced globally across all fields.

Examples:

| Field | Missing codes | Other interpretation |
|---|---|---|
| GENHLTH | 7, 9 | 1-5 remain ordered health responses |
| PHYSHLTH, MENTHLTH, POORHLTH | 77, 99 | 88 means no days and becomes 0 |
| CHILDREN | 99 | 88 means no children and becomes 0; 77 is a valid count |
| DRNK3GE5 | 77, 99 | 88 means no episodes and becomes 0 |
| INCOME2 | 77, 99 | Category 7 is valid and retained |
| BPHIGH4 | 7, 9 | Pregnancy-only and borderline responses remain valid |
| _AGEG5YR | 14 | Other age-group codes are retained |
| _AGE65YR | 3 | The two actual age groups are retained |

The five zero recodings apply only to PHYSHLTH, MENTHLTH, POORHLTH, CHILDREN and DRNK3GE5. Blank or nonresponse entries are not inferred to be zero from other answers or questionnaire routing.

## Order and reproducibility

1. Apply the explicit mapping to each original column.
2. On training rows only, apply the existing exclusion rules: administrative fields, >80% missing, and fewer than two observed distinct values.
3. Fit the existing numeric or expanded preprocessing on the cleaned training data.
4. Store the relevant cleaning rule in each retained feature's fitted specification.
5. At inference, apply those saved rules before imputation, scaling or encoding. The missingness indicator includes both original blanks and decoded nonresponses.

Both cleaning on/off runs use the same split, lambda=0.0001, gradient-descent algorithm and checkpoint budgets. Differences in retained columns or learned cutoffs caused by cleaning are part of the cleaning ablation.

## Deliberately unchanged

Existing missingness flags `_FRUITEX`, `_VEGETEX`, `PAMISS1_` remain observed indicators. Six calculated continuous fields (`WTKG3`, `MAXVO2_`, `FC60_`, `PAFREQ1_`, `PAFREQ2_`, `STRFREQ_`) are excluded from the mapping because the supplied CSV has rescaled calculated measurements and this study does not establish all sentinel scaling conventions. No blanket transformation based on a 7/9 suffix is used.

Mixed-unit raw weight, height, frequency and duration codes are not normalized by this experiment. Valid `never` responses outside the five named count fields remain separate observed codes, not missing. These are limitations and possible future work; the cleaning does not claim to be a complete reconstruction of the BRFSS processing pipeline.

`artifacts/improvements/cleaning_audit.json` reports counts affected in the training partition only. `test_improvements.py` checks field-specific behavior, preservation of valid categories, fitted-state serialization and correspondence of checkpoint training to the required gradient-descent implementation.
