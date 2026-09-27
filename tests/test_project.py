import pytest

import numpy as np
import pandas as pd

from src.data import FEATURE_COLUMNS, TRANSACTION_SOURCE_COLUMNS, generate_transactions, normalize_transactions, validate_transactions
from src.modeling import train_models
from app import sample_transactions_by_class


def test_synthetic_data_is_reproducible_and_has_expected_schema():
    first = generate_transactions(rows=2_000, seed=13)
    second = generate_transactions(rows=2_000, seed=13)
    assert first.equals(second)
    assert list(first.columns) == FEATURE_COLUMNS + ["Class"]
    assert first["Class"].sum() == 3


def test_data_validation_rejects_missing_features():
    frame = generate_transactions(rows=1_000)
    with pytest.raises(ValueError, match="Missing required"):
        validate_transactions(frame.drop(columns="V28"))


def test_normalize_transactions_converts_numeric_csv_values():
    frame = generate_transactions(rows=1_000)
    frame["Amount"] = frame["Amount"].astype(str)
    normalized = normalize_transactions(frame)
    assert np.issubdtype(normalized["Amount"].dtype, np.number)


def test_data_validation_rejects_non_finite_features():
    frame = generate_transactions(rows=1_000)
    frame.loc[0, "V1"] = np.inf
    with pytest.raises(ValueError, match="finite numeric"):
        validate_transactions(frame)


def test_transaction_record_dataset_maps_fraud_labels_and_sets_prediction_schema():
    records = pd.DataFrame({
        "TransactionDate": ["2024-01-01 10:30:00", "2024-02-02 21:15:00"],
        "Amount": [25.50, 900.0],
        "MerchantID": [10, 24],
        "TransactionType": ["purchase", "refund"],
        "Location": ["New York", "Dallas"],
        "IsFraud": [0, 1],
    })
    normalized = normalize_transactions(records)
    assert list(normalized.columns) == FEATURE_COLUMNS + ["Class"]
    assert normalized["Class"].tolist() == [0, 1]
    assert normalized.attrs["input_schema"] == "transaction_records"


def test_generator_rejects_invalid_row_count():
    with pytest.raises(ValueError, match="at least 100"):
        generate_transactions(rows=99)


def test_dashboard_sample_includes_both_classes():
    frame = generate_transactions(rows=10_000, seed=37)
    sample = sample_transactions_by_class(frame)
    assert set(sample["Class"].unique()) == {0, 1}


def test_training_saves_four_models_and_metrics(tmp_path):
    frame = generate_transactions(rows=6_000, seed=29)
    output = tmp_path / "model_bundle.joblib"
    bundle = train_models(frame, output, seed=29)
    assert output.exists()
    assert set(bundle["models"]) == {"Logistic_Regression", "Linear_SVM", "Random_Forest", "XGBoost"}
    probabilities = bundle["models"]["Linear_SVM"].predict_proba(frame[FEATURE_COLUMNS].iloc[:5])
    assert probabilities.shape == (5, 2)
    assert all(0 <= values["pr_auc"] <= 1 for values in bundle["metrics"].values())
    assert bundle["best_model"] in bundle["models"]


def test_training_transaction_records_sets_prediction_schema(tmp_path):
    canonical = generate_transactions(rows=6_000, seed=43, fraud_rate=0.02)
    records = pd.DataFrame({
        "TransactionDate": pd.date_range("2024-01-01", periods=len(canonical), freq="min"),
        "Amount": canonical["Amount"],
        "MerchantID": np.arange(len(canonical)) % 500,
        "TransactionType": np.where(canonical["V1"] > 0, "purchase", "refund"),
        "Location": np.where(canonical["V2"] > 0, "New York", "Dallas"),
        "IsFraud": canonical["Class"],
    })
    bundle = train_models(records, tmp_path / "records_bundle.joblib")
    assert bundle["input_schema"] == "transaction_records"
    assert {feature for feature, _ in bundle["feature_importance"]} == set(TRANSACTION_SOURCE_COLUMNS)
    assert not any(feature.startswith("V") for feature, _ in bundle["feature_importance"])