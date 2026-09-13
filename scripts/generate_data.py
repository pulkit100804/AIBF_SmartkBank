#!/usr/bin/env python
"""
Generate synthetic datasets for SmartBank AI.

Usage:
    python scripts/generate_data.py

Outputs:
    data/credit_applications.csv
    data/transactions.csv
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.config.settings import CREDIT_DATA_PATH, TRANSACTION_DATA_PATH
from app.data.generator import generate_credit_data, generate_transaction_data


def main() -> None:
    """Generate all synthetic datasets."""
    print("=" * 60)
    print("SmartBank AI — Synthetic Data Generation")
    print("=" * 60)

    # Generate credit data
    print("\n[1/2] Generating credit application data...")
    credit_df = generate_credit_data()
    credit_df.to_csv(CREDIT_DATA_PATH, index=False)
    print(f"  [OK] Saved {len(credit_df)} records to {CREDIT_DATA_PATH.name}")
    print(f"  Risk distribution:")
    for label, count in credit_df["risk_label"].value_counts().items():
        pct = count / len(credit_df) * 100
        print(f"    {label:8s}: {count:5d} ({pct:.1f}%)")

    # Generate transaction data
    print("\n[2/2] Generating transaction data...")
    txn_df = generate_transaction_data()
    txn_df.to_csv(TRANSACTION_DATA_PATH, index=False)
    print(f"  [OK] Saved {len(txn_df)} records to {TRANSACTION_DATA_PATH.name}")
    fraud_count = txn_df["is_fraud"].sum()
    print(f"  Fraud distribution:")
    print(f"    Normal:  {len(txn_df) - fraud_count:5d} ({(1 - fraud_count/len(txn_df))*100:.1f}%)")
    print(f"    Fraud:   {fraud_count:5d} ({fraud_count/len(txn_df)*100:.1f}%)")

    print("\n" + "=" * 60)
    print("Data generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
