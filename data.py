"""Memory-friendly CSV loading (NumPy + standard library only)."""

import csv
from pathlib import Path
import numpy as np


def read_csv(path):
    """Returns (header, float32 matrix); empty cells become NaN."""
    path = Path(path)
    with path.open() as f:
        n = sum(1 for _ in f) - 1
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        out = np.empty((n, len(header)), dtype=np.float32)
        for i, row in enumerate(reader):
            out[i] = [float(v) if v else np.nan for v in row]
    return header, out


def load(data_dir, train=True):
    """Returns x_train, y_train (0/1), x_test, test_ids, feature names."""
    data_dir = Path(data_dir)
    header, test = read_csv(data_dir / "x_test.csv")
    if not train:
        return None, None, test[:, 1:], test[:, 0].astype(np.int64), header[1:]
    h2, tr = read_csv(data_dir / "x_train.csv")
    _, lab = read_csv(data_dir / "y_train.csv")
    assert h2 == header and np.array_equal(tr[:, 0], lab[:, 0]), "schema / id mismatch"
    return (
        tr[:, 1:],
        (lab[:, 1] == 1).astype(np.int8),
        test[:, 1:],
        test[:, 0].astype(np.int64),
        header[1:],
    )
