"""Select using training-only CV, then evaluate the winner once on held-out data."""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_data


def build_pipeline(estimator):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def train(csv_path=None, target="target", output="artifacts", seed=42):
    x, y = load_data(csv_path, target)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y
    )
    candidates = {
        "baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=100, min_samples_leaf=2, random_state=seed, n_jobs=1
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    comparisons = []
    for name, estimator in candidates.items():
        scores = cross_validate(
            build_pipeline(estimator),
            x_train,
            y_train,
            cv=cv,
            scoring={"f1": "f1", "roc_auc": "roc_auc"},
            error_score="raise",
        )
        comparisons.append(
            {
                "model": name,
                "cv_f1_mean": float(scores["test_f1"].mean()),
                "cv_f1_std": float(scores["test_f1"].std()),
                "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
            }
        )
    best = max(comparisons, key=lambda row: row["cv_f1_mean"])["model"]
    model = build_pipeline(candidates[best]).fit(x_train, y_train)
    prediction = model.predict(x_test)
    probability = model.predict_proba(x_test)[:, 1]
    fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(pd.concat([x, y], axis=1), index=True).values.tobytes()
    ).hexdigest()
    report = {
        "dataset": str(csv_path) if csv_path else "sklearn breast_cancer benchmark",
        "dataset_sha256": fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "seed": seed,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "features": list(x.columns),
        "selected_model": best,
        "selection_metric": "training-only stratified 5-fold CV F1",
        "comparisons": comparisons,
        "holdout": {
            "accuracy": float(accuracy_score(y_test, prediction)),
            "f1": float(f1_score(y_test, prediction)),
            "roc_auc": float(roc_auc_score(y_test, probability)),
            "confusion_matrix": confusion_matrix(y_test, prediction).tolist(),
        },
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(comparisons).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(
        {
            "row_index": x_test.index,
            "actual": y_test.to_numpy(),
            "prediction": prediction,
            "probability_1": probability,
        }
    ).to_csv(output / "holdout_predictions.csv", index=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--target", default="target")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        report = train(args.csv, args.target, args.output, args.seed)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Training failed: {error}\n")
    print(
        json.dumps(
            {"selected_model": report["selected_model"], "holdout": report["holdout"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
