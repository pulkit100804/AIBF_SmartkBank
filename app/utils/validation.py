"""
Input validation for SmartBank AI.

Validates user inputs for credit applications, transactions, and CSV uploads.
Returns structured errors instead of raising raw exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.config.settings import EMPLOYMENT_TYPES, TRANSACTION_TYPES


# ─── Validation Result ──────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.is_valid = False
        self.errors.append(message)


# ─── Credit Application Validation ──────────────────────────────────────────

def validate_credit_input(data: dict = None, **kwargs) -> ValidationResult:
    """Validate credit application input fields."""
    if data is None:
        data = kwargs
    elif isinstance(data, dict):
        data = {**data, **kwargs}

    result = ValidationResult()

    age = data.get("age")
    annual_income = data.get("annual_income")
    employment_type = data.get("employment_type")
    employment_stability_years = data.get("employment_stability_years")
    credit_score = data.get("credit_score")
    existing_loans_count = data.get("existing_loans_count")
    existing_monthly_debt = data.get("existing_monthly_debt")
    requested_loan_amount = data.get("requested_loan_amount")
    loan_tenure_months = data.get("loan_tenure_months")
    dependents = data.get("dependents")
    savings_balance = data.get("savings_balance")

    if age is None or not isinstance(age, (int, float)) or age < 18 or age > 100:
        result.add_error("Age must be between 18 and 100.")

    if annual_income is None or not isinstance(annual_income, (int, float)) or annual_income <= 0:
        result.add_error("Annual income must be a positive number.")
    elif annual_income > 100_000_000:
        result.add_error("Annual income exceeds maximum allowed value.")

    if employment_type not in EMPLOYMENT_TYPES:
        result.add_error(
            f"Employment type must be one of: {', '.join(EMPLOYMENT_TYPES)}."
        )

    if (
        employment_stability_years is None
        or not isinstance(employment_stability_years, (int, float))
        or employment_stability_years < 0
        or employment_stability_years > 50
    ):
        result.add_error("Employment stability must be between 0 and 50 years.")

    if credit_score is None or not isinstance(credit_score, (int, float)) or credit_score < 300 or credit_score > 900:
        result.add_error("Credit score must be between 300 and 900.")

    if existing_loans_count is None or not isinstance(existing_loans_count, (int, float)) or existing_loans_count < 0:
        result.add_error("Existing loans count cannot be negative.")

    if existing_monthly_debt is None or not isinstance(existing_monthly_debt, (int, float)) or existing_monthly_debt < 0:
        result.add_error("Existing monthly debt cannot be negative.")

    if (
        requested_loan_amount is None
        or not isinstance(requested_loan_amount, (int, float))
        or requested_loan_amount <= 0
    ):
        result.add_error("Requested loan amount must be a positive number.")

    if (
        loan_tenure_months is None
        or not isinstance(loan_tenure_months, (int, float))
        or loan_tenure_months < 6
        or loan_tenure_months > 360
    ):
        result.add_error("Loan tenure must be between 6 and 360 months.")

    if dependents is None or not isinstance(dependents, (int, float)) or dependents < 0 or dependents > 15:
        result.add_error("Dependents must be between 0 and 15.")

    if savings_balance is None or not isinstance(savings_balance, (int, float)) or savings_balance < 0:
        result.add_error("Savings balance cannot be negative.")

    return result


# ─── Transaction Validation ─────────────────────────────────────────────────

def validate_transaction_input(data: dict = None, **kwargs) -> ValidationResult:
    """Validate transaction input fields."""
    if data is None:
        data = kwargs
    elif isinstance(data, dict):
        data = {**data, **kwargs}

    result = ValidationResult()

    amount = data.get("amount")
    transaction_type = data.get("transaction_type")
    hour_of_day = data.get("hour_of_day")
    day_of_week = data.get("day_of_week")
    transaction_frequency_24h = data.get("transaction_frequency_24h")
    customer_avg_amount = data.get("customer_avg_amount")
    distance_from_home_km = data.get("distance_from_home_km")
    account_age_days = data.get("account_age_days")

    if amount is None or not isinstance(amount, (int, float)) or amount <= 0:
        result.add_error("Transaction amount must be a positive number.")

    if transaction_type not in TRANSACTION_TYPES:
        result.add_error(
            f"Transaction type must be one of: {', '.join(TRANSACTION_TYPES)}."
        )

    if hour_of_day is None or not isinstance(hour_of_day, (int, float)) or hour_of_day < 0 or hour_of_day > 23:
        result.add_error("Hour of day must be between 0 and 23.")

    if day_of_week is None or not isinstance(day_of_week, (int, float)) or day_of_week < 0 or day_of_week > 6:
        result.add_error("Day of week must be between 0 (Monday) and 6 (Sunday).")

    if transaction_frequency_24h is None or not isinstance(transaction_frequency_24h, (int, float)) or transaction_frequency_24h < 0:
        result.add_error("Transaction frequency cannot be negative.")

    if customer_avg_amount is None or not isinstance(customer_avg_amount, (int, float)) or customer_avg_amount < 0:
        result.add_error("Customer average amount cannot be negative.")

    if distance_from_home_km is None or not isinstance(distance_from_home_km, (int, float)) or distance_from_home_km < 0:
        result.add_error("Distance from home cannot be negative.")

    if account_age_days is None or not isinstance(account_age_days, (int, float)) or account_age_days < 0:
        result.add_error("Account age cannot be negative.")

    return result


# ─── CSV Validation ─────────────────────────────────────────────────────────

REQUIRED_TRANSACTION_CSV_COLUMNS = {
    "date", "amount", "category", "description",
}


def validate_transaction_csv(df) -> ValidationResult:
    """Validate an uploaded transaction CSV for financial insights.

    Args:
        df: pandas DataFrame from uploaded CSV.

    Returns:
        ValidationResult with is_valid flag and any error messages.
    """
    result = ValidationResult()

    if df is None or df.empty:
        result.add_error("Uploaded CSV is empty.")
        return result

    missing = REQUIRED_TRANSACTION_CSV_COLUMNS - set(df.columns.str.lower())
    if missing:
        result.add_error(
            f"Missing required columns: {', '.join(sorted(missing))}. "
            f"Required: {', '.join(sorted(REQUIRED_TRANSACTION_CSV_COLUMNS))}."
        )

    if len(df) > 50_000:
        result.add_error("CSV exceeds maximum of 50,000 rows.")

    return result
