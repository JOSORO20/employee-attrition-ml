from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from .config import paths
from .data_prep import load_data, split_data, save_splits, TARGET_COL
from .features import build_model_pipeline, ALL_FEATURES

def train() -> dict:
    df = load_data()
    train_df, test_df = split_data(df)
    save_splits(train_df, test_df)

    X_train = train_df[ALL_FEATURES]
    y_train = train_df[TARGET_COL].astype(int)
    X_test = test_df[ALL_FEATURES]
    y_test = test_df[TARGET_COL].astype(int)

    pipe = build_model_pipeline()
    pipe.fit(X_train, y_train)

    # Evaluate
    proba = pipe.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
    }

    # Persist
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, paths.model_path)
    with open(paths.metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics

if __name__ == "__main__":
    print(train())
