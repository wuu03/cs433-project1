"""Feature engineering for the BRFSS 2015 MICHD task (NumPy only).

All transformations are deterministic, row-wise functions of the raw survey
answers (no statistics are fitted here), so they cannot leak labels and are
applied identically to training and test rows.
"""

import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
# Survey administration / sampling-design fields: not respondent characteristics.
ADMIN = set(
    "IDATE IMONTH IDAY IYEAR FMONTH SEQNO _PSU DISPCODE _STSTR _STRWT _RAWRAKE "
    "_WT2RAKE _CLLCPWT _LLCPWT".split()
)
# yes(1)/no(2) chronic-condition questions; diabetes coded 1 = yes
CHRONIC = [
    "CVDSTRK3",
    "ASTHMA3",
    "CHCSCNCR",
    "CHCOCNCR",
    "CHCCOPD1",
    "HAVARTH3",
    "ADDEPEV2",
    "CHCKIDNY",
    "DIABETE3",
    "BPHIGH4",
    "TOLDHI2",
]
LIMITS = ["QLACTLM2", "USEEQUIP", "BLIND", "DECIDE", "DIFFWALK", "DIFFDRES", "DIFFALON"]


def load_codebook():
    return json.loads((HERE / "survey_codes.json").read_text())["rules"]


def engineer(x, names, rules=None):
    """Returns (features, feature_names).

    * raw columns, minus administrative fields;
    * codebook cleaning: 'don't know / refused' codes -> -1 (kept distinct from
      'not asked' = NaN) and 'none' codes (88) -> 0, so ordinal splits are meaningful;
    * derived: exact age (from MAXVO2_ which is computed from un-capped age and sex),
      days of alcohol per month, counts of chronic conditions / activity
      limitations / non-responses.
    """
    rules = load_codebook() if rules is None else rules
    col = {n: j for j, n in enumerate(names)}
    out, out_names = [], []
    for n, j in col.items():
        if n in ADMIN:
            continue
        v = x[:, j].astype(np.float64).copy()
        r = rules.get(n)
        if r:
            if r["missing_codes"]:
                v[np.isin(v, r["missing_codes"])] = -1.0
            if r["zero_codes"]:
                v[np.isin(v, r["zero_codes"])] = 0.0
        out.append(v)
        out_names.append(n)

    def raw(n):
        return x[:, col[n]].astype(np.float64)

    sex = raw("SEX")
    vo2 = raw("MAXVO2_")
    age = np.where(sex == 1, (60.0 - vo2) / 0.55, (48.0 - vo2) / 0.37)
    age[(vo2 > 900) | ~np.isfinite(vo2)] = np.nan
    age = np.where(np.isfinite(age), np.round(age), raw("_AGE80"))
    out.append(age)
    out_names.append("AGE_EXACT")

    alc = raw("ALCDAY5")
    month = np.where(
        (alc >= 101) & (alc <= 199),
        (alc - 100) * 30 / 7.0,
        np.where(
            (alc >= 201) & (alc <= 299), alc - 200, np.where(alc == 888, 0, np.nan)
        ),
    )
    out.append(month)
    out_names.append("ALC_DAYS_MONTH")

    chronic = np.zeros(len(x))
    unknown = np.zeros(len(x))
    for n in CHRONIC:
        v = raw(n)
        chronic += v == 1
        unknown += np.isin(v, [7, 9])
    limits = sum((raw(n) == 1).astype(float) for n in LIMITS)
    out += [chronic, limits, unknown, np.isnan(x).sum(1).astype(float)]
    out_names += ["N_CHRONIC", "N_LIMITS", "N_UNKNOWN_CHRONIC", "N_MISSING"]
    # age x sex risk index (men develop CHD ~10 years earlier)
    out.append(age + 10 * (sex == 1))
    out_names.append("AGE_SEX_SHIFT")
    return np.column_stack(out), out_names
