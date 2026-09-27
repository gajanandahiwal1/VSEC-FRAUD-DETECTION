from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data import generate_transactions, normalize_transactions
from src.modeling import train_models


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic transactions and train demo models.")
    parser.add_argument("--rows", type=int, default=20_000, help="Number of synthetic transactions (default: 20000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--input", type=Path, help="Optional labeled transaction CSV to train instead of generating synthetic data")
    args = parser.parse_args()
    data_path = ROOT / "data" / "credit_card_fraud_dataset.csv"
    model_path = ROOT / "models" / "model_bundle.joblib"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if args.input:
        input_path = args.input if args.input.is_absolute() else ROOT / args.input
        raw_frame = pd.read_csv(input_path, low_memory=False)
        frame = normalize_transactions(raw_frame)
        data_note = f"Uploaded dataset: {input_path.name} ({len(frame):,} rows)."
    else:
        frame = generate_transactions(rows=args.rows, seed=args.seed)
        raw_frame = frame
        data_note = "Synthetic demonstration data, not real transaction observations."
        frame.to_csv(data_path, index=False)
    bundle = train_models(raw_frame, model_path, seed=args.seed, data_note=data_note)
    print(f"Trained on {len(frame):,} rows from {data_path.name if not args.input else input_path.name}")
    print(f"Selected model: {bundle['best_model']} (PR-AUC {bundle['metrics'][bundle['best_model']]['pr_auc']:.3f})")


if __name__ == "__main__":
    main()