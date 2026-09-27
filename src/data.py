from __future__ import annotations

import numpy as np
import pandas as pd
import zlib


FEATURE_COLUMNS = ["Time", "Amount", *[f"V{index}" for index in range(1, 29)]]
TARGET_COLUMN = "Class"
TRANSACTION_RECORD_COLUMNS = [
    "TransactionDate", "Amount", "MerchantID", "TransactionType", "Location", "IsFraud"
]
TRANSACTION_SOURCE_COLUMNS = ["TransactionDate", "Amount", "MerchantID", "TransactionType", "Location"]
TRANSACTION_FEATURE_SOURCES = {
    "Time": ("TransactionDate",),
    "Amount": ("Amount",),
    "V1": ("MerchantID",),
    "V2": ("TransactionType",),
    "V3": ("Location",),
    "V4": ("TransactionDate",),
    "V5": ("TransactionDate",),
    "V6": ("TransactionDate",),
    "V7": ("TransactionDate",),
    "V8": ("TransactionDate",),
    "V9": ("TransactionDate",),
    "V10": ("Amount",),
    "V11": ("Amount",),
    "V12": ("MerchantID",),
    "V13": ("MerchantID",),
    "V14": ("MerchantID",),
    "V15": ("TransactionDate",),
    "V16": ("TransactionDate",),
    "V17": ("TransactionDate",),
    "V18": ("TransactionDate",),
    "V19": ("TransactionDate",),
    "V20": ("TransactionDate",),
    "V21": ("MerchantID",),
    "V22": ("MerchantID",),
    "V23": ("Amount", "TransactionType"),
    "V24": ("Amount", "Location"),
    "V25": ("MerchantID", "TransactionType"),
    "V26": ("MerchantID", "Location"),
    "V27": ("TransactionDate", "TransactionType"),
    "V28": ("TransactionDate", "Location"),
}


def is_transaction_record_schema(frame: pd.DataFrame) -> bool:
    return set(TRANSACTION_RECORD_COLUMNS).issubset(frame.columns)


def _stable_category_code(values: pd.Series) -> np.ndarray:
    return values.fillna("unknown").astype(str).str.strip().str.lower().map(
        lambda value: (zlib.crc32(value.encode("utf-8")) % 20001) / 10000 - 1
    ).to_numpy(dtype=float)


def _normalize_transaction_records(frame: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(frame["TransactionDate"], errors="coerce", utc=True)
    amounts = pd.to_numeric(frame["Amount"], errors="coerce")
    merchants = pd.to_numeric(frame["MerchantID"], errors="coerce")
    labels = pd.to_numeric(frame["IsFraud"], errors="coerce")
    if dates.isna().any():
        raise ValueError("TransactionDate contains invalid or missing timestamps")
    if amounts.isna().any() or merchants.isna().any():
        raise ValueError("Amount and MerchantID must contain numeric values")

    merchant_values = merchants.to_numpy(dtype=float)
    amount_values = amounts.to_numpy(dtype=float)
    transaction_type = _stable_category_code(frame["TransactionType"])
    location = _stable_category_code(frame["Location"])
    hour = dates.dt.hour.to_numpy(dtype=float)
    minute = dates.dt.minute.to_numpy(dtype=float)
    weekday = dates.dt.dayofweek.to_numpy(dtype=float)
    day = dates.dt.day.to_numpy(dtype=float)
    month = dates.dt.month.to_numpy(dtype=float)
    weekend = (weekday >= 5).astype(float)
    log_amount = np.log1p(np.maximum(amount_values, 0))
    hour_angle = 2 * np.pi * hour / 24
    weekday_angle = 2 * np.pi * weekday / 7
    month_angle = 2 * np.pi * (month - 1) / 12
    merchant_scaled = merchant_values / 1000

    features = np.column_stack([
        merchant_values,
        transaction_type,
        location,
        hour,
        minute,
        weekday,
        day,
        month,
        weekend,
        log_amount,
        np.sqrt(np.maximum(amount_values, 0)),
        np.mod(merchant_values, 10),
        np.mod(merchant_values, 50),
        np.mod(merchant_values, 100),
        np.sin(hour_angle),
        np.cos(hour_angle),
        np.sin(weekday_angle),
        np.cos(weekday_angle),
        np.sin(month_angle),
        np.cos(month_angle),
        np.sin(merchant_scaled),
        np.cos(merchant_scaled),
        log_amount * transaction_type,
        log_amount * location,
        merchant_scaled * transaction_type,
        merchant_scaled * location,
        hour * transaction_type,
        weekend * location,
    ])

    normalized = pd.DataFrame(features, columns=[f"V{index}" for index in range(1, 29)])
    normalized.insert(0, "Amount", amount_values)
    normalized.insert(0, "Time", dates.astype("int64").to_numpy(dtype=float) / 1e9)
    normalized[TARGET_COLUMN] = labels.to_numpy()
    normalized.attrs["input_schema"] = "transaction_records"
    return normalize_transactions(normalized)


def generate_transactions(rows: int = 20_000, seed: int = 42, fraud_rate: float = 0.0017) -> pd.DataFrame:
    """Build a deterministic, intentionally synthetic imbalanced transaction dataset."""
    if rows < 100:
        raise ValueError("rows must be at least 100")
    if not 0 < fraud_rate < 1:
        raise ValueError("fraud_rate must be between 0 and 1")

    rng = np.random.default_rng(seed)
    fraud_count = max(1, int(round(rows * fraud_rate)))
    labels = np.zeros(rows, dtype=np.int8)
    fraud_indices = rng.choice(rows, size=fraud_count, replace=False)
    labels[fraud_indices] = 1

    elapsed = rng.uniform(0, 172_800, rows)
    elapsed[fraud_indices] = rng.uniform(0, 172_800, fraud_count)
    amount = rng.lognormal(mean=3.3, sigma=1.15, size=rows)
    amount[fraud_indices] *= rng.uniform(1.2, 3.4, fraud_count)
    amount = np.clip(amount, 0.5, 25_000)

    features = rng.normal(0, 1, size=(rows, 28))
    features[fraud_indices, :5] += rng.normal(1.7, 0.55, size=(fraud_count, 5))
    features[fraud_indices, 10:14] -= rng.normal(1.3, 0.45, size=(fraud_count, 4))

    frame = pd.DataFrame(features, columns=[f"V{index}" for index in range(1, 29)])
    frame.insert(0, "Amount", amount)
    frame.insert(0, "Time", elapsed)
    frame[TARGET_COLUMN] = labels
    return frame


def validate_transactions(frame: pd.DataFrame) -> None:
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required transaction columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("Transaction dataset is empty")
    labels = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError("Class must contain only non-null binary labels 0 and 1")
    features = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(features).all():
        raise ValueError("Transaction features must contain only finite numeric values")


def normalize_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert the expected transaction columns to numeric values and validate the schema."""
    if is_transaction_record_schema(frame):
        return _normalize_transaction_records(frame)

    normalized = frame.copy()
    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        if column in normalized:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    validate_transactions(normalized)
    return normalized