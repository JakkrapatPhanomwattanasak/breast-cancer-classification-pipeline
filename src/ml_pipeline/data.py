import csv

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_data(csv_path=None, target="target"):
    """Use a bundled research benchmark offline, or a binary-target numeric CSV."""
    if csv_path is None:
        frame = load_breast_cancer(as_frame=True).frame
        target = "target"
    else:
        with open(csv_path, encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), [])
        if len(header) != len(set(header)):
            raise ValueError("CSV column names must be unique")
        frame = pd.read_csv(csv_path)
    if target not in frame or not frame.columns.is_unique:
        raise ValueError("CSV must have unique columns and the requested target")
    y = frame[target]
    x = frame.drop(columns=target)
    if len(frame) < 30 or x.empty or y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("Need at least 30 rows, features, and a complete binary 0/1 target")
    if y.value_counts().min() < 10:
        raise ValueError("Need at least 10 rows in each class for stratified evaluation")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in x.dtypes):
        raise ValueError("All features must be numeric; encode categoricals first")
    if np.isinf(x.to_numpy(dtype=float)).any() or x.isna().all().any():
        raise ValueError("Features cannot contain infinity or entirely missing columns")
    return x, y.astype(int)
