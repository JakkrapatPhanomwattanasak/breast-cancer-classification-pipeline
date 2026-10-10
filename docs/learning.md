# Understanding Breast Cancer Classification Pipeline

- Entry point: `ml_pipeline.train`; validation lives in `data.load_data`.
- Explain the difference between training, cross-validation, and held-out testing.
- Why a Pipeline? Imputation and scaling are fitted separately inside each CV training fold, preventing leakage.
- Why compare a DummyClassifier? A model should earn its complexity against a simple reference.
- Why choose on CV F1? Choosing on held-out performance contaminates the final evaluation.
- Explain F1, ROC AUC, confusion matrices, and the positive class (benign = 1 for the bundled data).
- Explain the tradeoff between Logistic Regression and Random Forest and why scaling mainly matters for the former.
- Explain seeds, dataset fingerprints, version metadata, and trusted-only model loading.

Practice: rerun with seed 42, read comparison.csv, inspect the holdout confusion matrix, reload model.joblib, then try another seed without selecting based on holdout performance.

Resume wording after you can reproduce and explain the implementation:
"Built a reproducible classification benchmark comparing a baseline, Logistic Regression, and Random Forest using leakage-safe preprocessing and stratified cross-validation."

The reported benchmark scores are educational results, not evidence of clinical readiness.
