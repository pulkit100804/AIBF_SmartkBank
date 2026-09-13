"""
Synthetic data generator for SmartBank AI.

Generates realistic credit application and transaction datasets with
meaningful feature-target relationships and fixed random seeds for
reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.config.settings import (
    CREDIT_SAMPLES,
    EMPLOYMENT_TYPES,
    FRAUD_RATIO,
    RANDOM_SEED,
    TRANSACTION_SAMPLES,
    TRANSACTION_TYPES,
)


def generate_credit_data(
    n_samples: int = CREDIT_SAMPLES,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate synthetic credit application dataset.

    The target (risk_label) is derived from a weighted combination of
    features to ensure meaningful relationships without trivial
    predictability.

    Args:
        n_samples: Number of samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with credit application features and risk labels.
    """
    rng = np.random.RandomState(seed)

    # ── Generate raw features ───────────────────────────────────────
    age = rng.randint(21, 66, size=n_samples)

    # Income: log-normal, centered around 6-8 LPA
    annual_income = np.exp(
        rng.normal(loc=13.2, scale=0.6, size=n_samples)
    ).clip(150_000, 5_000_000).round(-3)

    employment_type = rng.choice(EMPLOYMENT_TYPES, size=n_samples, p=[0.45, 0.25, 0.20, 0.10])

    employment_stability_years = np.where(
        age < 25,
        rng.uniform(0, 4, size=n_samples),
        rng.uniform(0, min(30, 65) , size=n_samples),
    ).clip(0, age - 18).round(1)

    # Credit score: normal distribution centered at 680
    credit_score = rng.normal(680, 80, size=n_samples).clip(300, 900).astype(int)

    existing_loans_count = rng.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=[0.25, 0.30, 0.25, 0.12, 0.05, 0.03])

    # Monthly debt proportional to income and loan count
    base_debt_ratio = rng.uniform(0.0, 0.15, size=n_samples) * existing_loans_count
    existing_monthly_debt = (annual_income / 12 * base_debt_ratio).clip(0).round(0)

    # Requested loan: 0.5x to 5x annual income
    loan_multiplier = rng.uniform(0.3, 4.0, size=n_samples)
    requested_loan_amount = (annual_income * loan_multiplier).round(-3)

    loan_tenure_months = rng.choice([12, 24, 36, 48, 60, 84, 120], size=n_samples)

    dependents = rng.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=[0.20, 0.25, 0.30, 0.15, 0.07, 0.03])

    # Savings: correlated with income and credit score
    savings_multiplier = rng.uniform(0.05, 1.5, size=n_samples)
    credit_score_factor = (credit_score - 300) / 600  # 0 to 1
    savings_balance = (
        annual_income * savings_multiplier * (0.3 + 0.7 * credit_score_factor)
    ).clip(0).round(0)

    # ── Compute derived features for target generation ──────────────
    monthly_income = annual_income / 12
    dti_ratio = np.where(
        monthly_income > 0,
        existing_monthly_debt / monthly_income,
        0,
    )
    lti_ratio = np.where(
        annual_income > 0,
        requested_loan_amount / annual_income,
        0,
    )
    monthly_installment = requested_loan_amount / loan_tenure_months
    debt_burden = np.where(
        monthly_income > 0,
        (existing_monthly_debt + monthly_installment) / monthly_income,
        0,
    )

    # ── Generate risk score (weighted combination) ──────────────────
    # Normalize features to 0-1 scale for scoring
    norm_credit = (credit_score - 300) / 600  # Higher = better
    norm_dti = np.clip(dti_ratio, 0, 1)
    norm_lti = np.clip(lti_ratio / 5, 0, 1)
    norm_debt_burden = np.clip(debt_burden, 0, 1)
    norm_stability = np.clip(employment_stability_years / 15, 0, 1)
    norm_savings = np.clip(savings_balance / annual_income, 0, 1)

    # Risk score: 0 (low risk) to 1 (high risk)
    risk_score = (
        0.30 * (1 - norm_credit)          # Low credit score = high risk
        + 0.20 * norm_debt_burden          # High debt burden = high risk
        + 0.15 * norm_dti                  # High DTI = high risk
        + 0.15 * norm_lti                  # High LTI = high risk
        + 0.10 * (1 - norm_stability)      # Low stability = high risk
        + 0.10 * (1 - norm_savings)        # Low savings = high risk
    )

    # Add controlled noise (keeps relationships non-trivial)
    noise = rng.normal(0, 0.08, size=n_samples)
    risk_score = np.clip(risk_score + noise, 0, 1)

    # Map to categories using thresholds
    risk_label = np.where(
        risk_score < 0.35, "LOW",
        np.where(risk_score < 0.60, "MEDIUM", "HIGH"),
    )

    # Approved: strongly correlated with risk (but not identical)
    approve_prob = np.where(
        risk_label == "LOW", rng.uniform(0.80, 0.98, size=n_samples),
        np.where(
            risk_label == "MEDIUM", rng.uniform(0.30, 0.55, size=n_samples),
            rng.uniform(0.02, 0.15, size=n_samples),
        ),
    )
    approved = (rng.random(n_samples) < approve_prob).astype(int)

    # ── Build DataFrame ─────────────────────────────────────────────
    df = pd.DataFrame({
        "age": age,
        "annual_income": annual_income,
        "employment_type": employment_type,
        "employment_stability_years": employment_stability_years,
        "credit_score": credit_score,
        "existing_loans_count": existing_loans_count,
        "existing_monthly_debt": existing_monthly_debt,
        "requested_loan_amount": requested_loan_amount,
        "loan_tenure_months": loan_tenure_months,
        "dependents": dependents,
        "savings_balance": savings_balance,
        "risk_label": risk_label,
        "approved": approved,
    })

    return df


def generate_transaction_data(
    n_samples: int = TRANSACTION_SAMPLES,
    seed: int = RANDOM_SEED,
    fraud_ratio: float = FRAUD_RATIO,
) -> pd.DataFrame:
    """Generate synthetic transaction dataset with fraud labels.

    Normal and fraudulent transactions have distinctly different
    behavioral patterns to ensure the model learns meaningful signals.

    Args:
        n_samples: Total number of transactions.
        seed: Random seed for reproducibility.
        fraud_ratio: Fraction of fraudulent transactions.

    Returns:
        DataFrame with transaction features and is_fraud label.
    """
    rng = np.random.RandomState(seed)

    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    # ── Normal transactions ─────────────────────────────────────────
    normal = _generate_normal_transactions(n_normal, rng)

    # ── Fraudulent transactions ─────────────────────────────────────
    fraud = _generate_fraud_transactions(n_fraud, rng)

    # ── Combine and shuffle ─────────────────────────────────────────
    df = pd.concat([normal, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Add transaction IDs
    df.insert(0, "transaction_id", [f"TXN{str(i).zfill(6)}" for i in range(len(df))])

    # Add customer IDs (simulate ~500 customers)
    df["customer_id"] = [f"CUST{str(rng.randint(1, 501)).zfill(4)}" for _ in range(len(df))]

    return df


def _generate_normal_transactions(n: int, rng: np.random.RandomState) -> pd.DataFrame:
    """Generate normal (non-fraudulent) transaction records."""
    return pd.DataFrame({
        "amount": rng.lognormal(mean=7.5, sigma=1.0, size=n).clip(50, 50_000).round(2),
        "transaction_type": rng.choice(TRANSACTION_TYPES, size=n, p=[0.30, 0.35, 0.15, 0.20]),
        "hour_of_day": rng.choice(range(24), size=n, p=_normal_hour_distribution()),
        "day_of_week": rng.randint(0, 7, size=n),
        "location_change": (rng.random(n) < 0.05).astype(int),       # 5% chance
        "new_device": (rng.random(n) < 0.03).astype(int),            # 3% chance
        "transaction_frequency_24h": rng.poisson(2.5, size=n).clip(0, 20),
        "customer_avg_amount": rng.lognormal(mean=7.5, sigma=0.8, size=n).clip(100, 30_000).round(2),
        "distance_from_home_km": rng.exponential(5, size=n).clip(0, 50).round(1),
        "account_age_days": rng.randint(30, 3650, size=n),
        "is_fraud": 0,
    })


def _generate_fraud_transactions(n: int, rng: np.random.RandomState) -> pd.DataFrame:
    """Generate fraudulent transaction records with suspicious patterns."""
    return pd.DataFrame({
        "amount": rng.lognormal(mean=9.5, sigma=1.2, size=n).clip(5_000, 500_000).round(2),
        "transaction_type": rng.choice(TRANSACTION_TYPES, size=n, p=[0.45, 0.15, 0.30, 0.10]),
        "hour_of_day": rng.choice(range(24), size=n, p=_fraud_hour_distribution()),
        "day_of_week": rng.randint(0, 7, size=n),
        "location_change": (rng.random(n) < 0.65).astype(int),       # 65% chance
        "new_device": (rng.random(n) < 0.55).astype(int),            # 55% chance
        "transaction_frequency_24h": rng.poisson(8, size=n).clip(3, 30),
        "customer_avg_amount": rng.lognormal(mean=7.0, sigma=0.6, size=n).clip(100, 15_000).round(2),
        "distance_from_home_km": rng.exponential(80, size=n).clip(10, 2000).round(1),
        "account_age_days": rng.randint(1, 365, size=n),
        "is_fraud": 1,
    })


def _normal_hour_distribution() -> list[float]:
    """Create realistic hour-of-day distribution for normal transactions."""
    probs = np.array([
        0.5, 0.3, 0.2, 0.1, 0.1, 0.2,   # 0-5 (night, low)
        0.8, 1.5, 3.0, 5.0, 6.0, 5.5,   # 6-11 (morning peak)
        5.0, 4.5, 4.0, 4.5, 5.0, 5.5,   # 12-17 (afternoon)
        5.0, 4.0, 3.0, 2.0, 1.0, 0.8,   # 18-23 (evening decline)
    ])
    return (probs / probs.sum()).tolist()


def _fraud_hour_distribution() -> list[float]:
    """Create hour distribution for fraudulent transactions (skewed to night)."""
    probs = np.array([
        5.0, 5.5, 6.0, 6.5, 5.5, 4.0,   # 0-5 (night, HIGH)
        3.0, 2.0, 1.5, 1.5, 1.5, 1.5,   # 6-11 (morning, low)
        2.0, 2.5, 2.5, 2.0, 2.0, 2.5,   # 12-17 (afternoon)
        3.0, 3.5, 4.0, 4.5, 5.0, 5.5,   # 18-23 (evening ramp up)
    ])
    return (probs / probs.sum()).tolist()


def generate_customer_transactions(
    n_months: int = 6,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate customer transaction history for financial insights.

    Creates realistic monthly spending across categories for demo
    customers.

    Args:
        n_months: Number of months of history.
        seed: Random seed.

    Returns:
        DataFrame with date, amount, category, description columns.
    """
    rng = np.random.RandomState(seed)

    categories_config = {
        "Salary": {"monthly_count": 1, "amount_range": (40000, 120000), "is_income": True},
        "Groceries": {"monthly_count": (4, 8), "amount_range": (500, 3000), "is_income": False},
        "Dining": {"monthly_count": (3, 10), "amount_range": (200, 2500), "is_income": False},
        "Transport": {"monthly_count": (5, 15), "amount_range": (50, 800), "is_income": False},
        "Shopping": {"monthly_count": (2, 6), "amount_range": (500, 8000), "is_income": False},
        "Utilities": {"monthly_count": (2, 4), "amount_range": (500, 3000), "is_income": False},
        "Entertainment": {"monthly_count": (1, 5), "amount_range": (200, 2000), "is_income": False},
        "Healthcare": {"monthly_count": (0, 2), "amount_range": (300, 5000), "is_income": False},
        "Education": {"monthly_count": (0, 2), "amount_range": (1000, 10000), "is_income": False},
        "Rent": {"monthly_count": 1, "amount_range": (8000, 25000), "is_income": False},
    }

    records = []
    base_date = pd.Timestamp("2024-07-01")

    for month_offset in range(n_months):
        month_start = base_date + pd.DateOffset(months=month_offset)

        for category, config in categories_config.items():
            # Determine number of transactions this month
            if isinstance(config["monthly_count"], tuple):
                n_txns = rng.randint(config["monthly_count"][0], config["monthly_count"][1] + 1)
            else:
                n_txns = config["monthly_count"]

            for _ in range(n_txns):
                day = rng.randint(1, 29)
                date = month_start.replace(day=day)
                low, high = config["amount_range"]
                amount = round(rng.uniform(low, high), 2)

                if config["is_income"]:
                    amount_signed = amount
                    desc = f"{category} credit"
                else:
                    amount_signed = -amount
                    desc = f"{category} expense"

                records.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "amount": amount_signed,
                    "category": category,
                    "description": desc,
                })

    df = pd.DataFrame(records)
    df = df.sort_values("date").reset_index(drop=True)
    return df
