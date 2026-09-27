from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TRANSACTION_FEATURE_SOURCES,
    TRANSACTION_SOURCE_COLUMNS,
    is_transaction_record_schema,
    normalize_transactions,
    validate_transactions,
)


def train_models(
    frame: pd.DataFrame,
    output_path: str | Path,
    seed: int = 42,
    data_note: str = "Synthetic demonstration data, not real transaction observations.",
) -> dict:
    input_schema = "transaction_records" if is_transaction_record_schema(frame) else frame.attrs.get("input_schema", "creditcard")
    frame = normalize_transactions(frame)
    validate_transactions(frame)
    if frame[TARGET_COLUMN].nunique() < 2 or frame[TARGET_COLUMN].value_counts().min() < 2:
        raise ValueError("Training requires at least two examples in each class")

    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURE_COLUMNS], frame[TARGET_COLUMN].astype(int),
        test_size=0.25, random_state=seed, stratify=frame[TARGET_COLUMN],
    )
    candidates = {
        "Logistic_Regression": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1200, random_state=seed)
        ),
        "Linear_SVM": CalibratedClassifierCV(
            estimator=make_pipeline(
                StandardScaler(),
                LinearSVC(class_weight="balanced", max_iter=5000, random_state=seed),
            ),
            method="sigmoid",
            cv=3,
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=160, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=160, max_depth=4, learning_rate=0.06, subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()),
            eval_metric="logloss", tree_method="hist", n_jobs=-1, random_state=seed,
        ),
    }
    fitted = {}
    metrics = {}
    matrices = {}
    for name, estimator in candidates.items():
        estimator.fit(x_train, y_train)
        probabilities = estimator.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        fitted[name] = estimator
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "pr_auc": float(average_precision_score(y_test, probabilities)),
        }
        matrices[name] = confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()

    best_name = max(metrics, key=lambda name: metrics[name]["pr_auc"])
    best_estimator = fitted[best_name]
    if best_name == "Logistic_Regression":
        classifier = best_estimator.named_steps["logisticregression"]
    elif best_name == "Linear_SVM":
        classifier = best_estimator.calibrated_classifiers_[0].estimator.named_steps["linearsvc"]
    else:
        classifier = best_estimator
    raw_importance = getattr(classifier, "feature_importances_", None)
    if raw_importance is None:
        raw_importance = np.abs(classifier.coef_[0])
    if input_schema == "transaction_records":
        source_importance = {column: 0.0 for column in TRANSACTION_SOURCE_COLUMNS}
        for feature, importance in zip(FEATURE_COLUMNS, raw_importance.tolist()):
            sources = TRANSACTION_FEATURE_SOURCES[feature]
            for source in sources:
                source_importance[source] += importance / len(sources)
        ranked = sorted(source_importance.items(), key=lambda item: item[1], reverse=True)
    else:
        ranked = sorted(zip(FEATURE_COLUMNS, raw_importance.tolist()), key=lambda item: item[1], reverse=True)

    bundle = {
        "models": fitted,
        "metrics": metrics,
        "confusion_matrices": matrices,
        "best_model": best_name,
        "feature_importance": ranked,
        "features": FEATURE_COLUMNS,
        "threshold": 0.5,
        "input_schema": input_schema,
        "data_note": data_note,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
    return bundle