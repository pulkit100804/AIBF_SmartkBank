"""
Centralized configuration for SmartBank AI.

All paths, model parameters, and environment settings are defined here.
Uses pathlib for cross-platform compatibility.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
APP_DIR = BASE_DIR / "app"
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
SCRIPTS_DIR = BASE_DIR / "scripts"

# Ensure critical directories exist
ARTIFACTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# ─── Database ────────────────────────────────────────────────────────────────
DB_NAME = os.getenv("DB_NAME", "smartbank_ai.db")
DB_PATH = BASE_DIR / DB_NAME

# ─── Random Seed ─────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ─── Model Artifacts ────────────────────────────────────────────────────────
CREDIT_MODEL_PATH = ARTIFACTS_DIR / "credit_model.joblib"
CREDIT_PREPROCESSOR_PATH = ARTIFACTS_DIR / "credit_preprocessor.joblib"
FRAUD_MODEL_PATH = ARTIFACTS_DIR / "fraud_model.joblib"
FRAUD_PREPROCESSOR_PATH = ARTIFACTS_DIR / "fraud_preprocessor.joblib"

# ─── Dataset Paths ──────────────────────────────────────────────────────────
CREDIT_DATA_PATH = DATA_DIR / "credit_applications.csv"
TRANSACTION_DATA_PATH = DATA_DIR / "transactions.csv"

# ─── Dataset Generation Parameters ──────────────────────────────────────────
CREDIT_SAMPLES = 5000
TRANSACTION_SAMPLES = 10000
FRAUD_RATIO = 0.03  # 3% fraud rate

# ─── Model Hyperparameters ──────────────────────────────────────────────────
CREDIT_MODEL_PARAMS = {
    "n_estimators": 150,
    "max_depth": 5,
    "learning_rate": 0.1,
    "random_state": RANDOM_SEED,
}

FRAUD_MODEL_PARAMS = {
    "n_estimators": 150,
    "max_depth": 6,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
}

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ─── Optional LLM ───────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)
LLM_MODEL_NAME = "gemini-2.0-flash"

# ─── Application Metadata ───────────────────────────────────────────────────
APP_NAME = "SmartBank AI"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = (
    "AI-Powered Banking Risk, Fraud & Personalized Finance Assistant"
)

# ─── Feature Names (for explainability display) ─────────────────────────────
CREDIT_FEATURE_LABELS = {
    "age": "Age",
    "annual_income": "Annual Income",
    "employment_type_encoded": "Employment Type",
    "employment_stability_years": "Employment Stability (Years)",
    "credit_score": "Credit Score",
    "existing_loans_count": "Existing Loans Count",
    "existing_monthly_debt": "Existing Monthly Debt",
    "requested_loan_amount": "Requested Loan Amount",
    "loan_tenure_months": "Loan Tenure (Months)",
    "dependents": "Dependents",
    "savings_balance": "Savings Balance",
    "dti_ratio": "Debt-to-Income Ratio",
    "lti_ratio": "Loan-to-Income Ratio",
    "monthly_installment": "Estimated Monthly Installment",
    "debt_burden": "Total Debt Burden Ratio",
    "savings_ratio": "Savings-to-Income Ratio",
    "income_per_dependent": "Income per Dependent",
}

FRAUD_FEATURE_LABELS = {
    "amount": "Transaction Amount",
    "transaction_type_encoded": "Transaction Type",
    "hour_of_day": "Hour of Day",
    "day_of_week": "Day of Week",
    "location_change": "Location Changed",
    "new_device": "New Device Used",
    "transaction_frequency_24h": "Transactions in Last 24h",
    "customer_avg_amount": "Customer Average Amount",
    "distance_from_home_km": "Distance from Home (km)",
    "account_age_days": "Account Age (Days)",
    "amount_ratio": "Amount vs Average Ratio",
    "high_amount_flag": "High Amount Flag",
    "is_night": "Night Transaction",
    "is_weekend": "Weekend Transaction",
    "risk_composite": "Risk Composite Score",
}

# ─── Risk Thresholds ────────────────────────────────────────────────────────
FRAUD_LOW_THRESHOLD = 0.3
FRAUD_HIGH_THRESHOLD = 0.6

# Employment type mapping
EMPLOYMENT_TYPES = ["Salaried", "Self-Employed", "Business", "Freelance"]
TRANSACTION_TYPES = ["Transfer", "Payment", "Withdrawal", "Purchase"]
