"""
Feature engineering and preprocessing for SmartBank AI.

Transforms raw input features into model-ready features with
derived financial indicators like DTI, LTI, and debt burden ratios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.config.settings import EMPLOYMENT_TYPES, TRANSACTION_TYPES


# ─── Credit Feature Engineering ──────────────────────────────────────────────

CREDIT_RAW_FEATURES = [
    "age", "annual_income", "employment_type", "employment_stability_years",
    "credit_score", "existing_loans_count", "existing_monthly_debt",
    "requested_loan_amount", "loan_tenure_months", "dependents",
    "savings_balance",
]

CREDIT_MODEL_FEATURES = [
    "age", "annual_income", "employment_type_encoded",
    "employment_stability_years", "credit_score", "existing_loans_count",
    "existing_monthly_debt", "requested_loan_amount", "loan_tenure_months",
    "dependents", "savings_balance", "dti_ratio", "lti_ratio",
    "monthly_installment", "debt_burden", "savings_ratio",
    "income_per_dependent",
]


def engineer_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived financial features to credit application data.

    Engineered features:
        - dti_ratio: Debt-to-Income ratio (annual)
        - lti_ratio: Loan-to-Income ratio
        - monthly_installment: Estimated EMI
        - debt_burden: Total debt burden including new loan
        - savings_ratio: Savings relative to income
        - income_per_dependent: Income per family member

    Args:
        df: DataFrame with raw credit features.

    Returns:
        DataFrame with additional engineered feature columns.
    """
    result = df.copy()

    monthly_income = result["annual_income"] / 12

    result["dti_ratio"] = np.where(
        monthly_income > 0,
        result["existing_monthly_debt"] / monthly_income,
        0,
    )

    result["lti_ratio"] = np.where(
        result["annual_income"] > 0,
        result["requested_loan_amount"] / result["annual_income"],
        0,
    )

    result["monthly_installment"] = np.where(
        result["loan_tenure_months"] > 0,
        result["requested_loan_amount"] / result["loan_tenure_months"],
        0,
    )

    result["debt_burden"] = np.where(
        monthly_income > 0,
        (result["existing_monthly_debt"] + result["monthly_installment"]) / monthly_income,
        0,
    )

    result["savings_ratio"] = np.where(
        result["annual_income"] > 0,
        result["savings_balance"] / result["annual_income"],
        0,
    )

    result["income_per_dependent"] = result["annual_income"] / (result["dependents"] + 1)

    # Encode employment type
    emp_map = {emp: i for i, emp in enumerate(EMPLOYMENT_TYPES)}
    result["employment_type_encoded"] = result["employment_type"].map(emp_map).fillna(0).astype(int)

    return result


def get_credit_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only the model input features from an engineered DataFrame.

    Args:
        df: DataFrame after engineer_credit_features.

    Returns:
        DataFrame with only model feature columns.
    """
    return df[CREDIT_MODEL_FEATURES].copy()


# ─── Fraud Feature Engineering ───────────────────────────────────────────────

FRAUD_RAW_FEATURES = [
    "amount", "transaction_type", "hour_of_day", "day_of_week",
    "location_change", "new_device", "transaction_frequency_24h",
    "customer_avg_amount", "distance_from_home_km", "account_age_days",
]

FRAUD_MODEL_FEATURES = [
    "amount", "transaction_type_encoded", "hour_of_day", "day_of_week",
    "location_change", "new_device", "transaction_frequency_24h",
    "customer_avg_amount", "distance_from_home_km", "account_age_days",
    "amount_ratio", "high_amount_flag", "is_night", "is_weekend",
    "risk_composite",
]


def engineer_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived behavioral features to transaction data.

    Engineered features:
        - amount_ratio: Transaction amount vs customer average
        - high_amount_flag: 1 if amount > 3x customer average
        - is_night: 1 if transaction during 0:00-5:59 or 23:00
        - is_weekend: 1 if Saturday (5) or Sunday (6)
        - risk_composite: Sum of location_change + new_device + high frequency

    Args:
        df: DataFrame with raw transaction features.

    Returns:
        DataFrame with additional engineered feature columns.
    """
    result = df.copy()

    result["amount_ratio"] = np.where(
        result["customer_avg_amount"] > 0,
        result["amount"] / result["customer_avg_amount"],
        result["amount"] / 1000,
    )

    result["high_amount_flag"] = (
        result["amount"] > result["customer_avg_amount"] * 3
    ).astype(int)

    result["is_night"] = (
        (result["hour_of_day"] < 6) | (result["hour_of_day"] >= 23)
    ).astype(int)

    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)

    result["risk_composite"] = (
        result["location_change"].astype(int)
        + result["new_device"].astype(int)
        + (result["transaction_frequency_24h"] > 10).astype(int)
    )

    # Encode transaction type
    txn_map = {t: i for i, t in enumerate(TRANSACTION_TYPES)}
    result["transaction_type_encoded"] = result["transaction_type"].map(txn_map).fillna(0).astype(int)

    return result


def get_fraud_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only the model input features from an engineered DataFrame.

    Args:
        df: DataFrame after engineer_fraud_features.

    Returns:
        DataFrame with only model feature columns.
    """
    return df[FRAUD_MODEL_FEATURES].copy()
