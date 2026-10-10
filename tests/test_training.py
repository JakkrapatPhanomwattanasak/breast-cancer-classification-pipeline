import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ml_pipeline.data import load_data
from ml_pipeline.train import build_pipeline, train


def test_training_artifacts_and_deterministic_metrics(tmp_path):
    first = train(output=tmp_path / "first")
    second = train(output=tmp_path / "second")
    assert first["holdout"] == second["holdout"]
    assert first["comparisons"] == second["comparisons"]
    assert first["dataset_sha256"] == second["dataset_sha256"]
    assert first["train_rows"] + first["test_rows"] == 569
    best = max(first["comparisons"], key=lambda item: item["cv_f1_mean"])
    assert first["selected_model"] == best["model"]
    x, _ = load_data()
    model = joblib.load(tmp_path / "first" / "model.joblib")
    assert len(model.predict(x.iloc[:3])) == 3
    assert json.loads((tmp_path / "first" / "metrics.json").read_text())["seed"] == 42
    assert len(pd.read_csv(tmp_path / "first" / "holdout_predictions.csv")) == first["test_rows"]


def test_preprocessing_fits_training_data_only():
    x = pd.DataFrame({"feature": [1.0, np.nan, 3.0, 4.0]})
    model = build_pipeline(LogisticRegression()).fit(x, [0, 0, 1, 1])
    assert model.named_steps["imputer"].statistics_[0] == 3
    model.predict(pd.DataFrame({"feature": [10000.0]}))
    assert model.named_steps["imputer"].statistics_[0] == 3


def test_duplicate_csv_headers_rejected(tmp_path):
    path = tmp_path / "duplicate.csv"
    path.write_text("feature,feature,target\n1,2,0\n")
    with pytest.raises(ValueError, match="unique"):
        load_data(path)


@pytest.mark.parametrize(
    "failure", ["missing_target", "text", "infinity", "single_class", "all_missing"]
)
def test_bad_csv(tmp_path, failure):
    frame = pd.DataFrame({"value": range(40), "target": [0, 1] * 20})
    if failure == "missing_target":
        frame = frame.drop(columns="target")
    elif failure == "text":
        frame["value"] = "bad"
    elif failure == "infinity":
        frame["value"] = frame["value"].astype(float)
        frame.loc[0, "value"] = np.inf
    elif failure == "single_class":
        frame["target"] = 0
    else:
        frame["value"] = np.nan
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_data(path)
