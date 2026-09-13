import pytest
from app.utils.validation import validate_credit_input, validate_transaction_input, validate_transaction_csv
import pandas as pd

def test_validate_credit_input_valid():
    valid_data = {
        "age": 35,
        "annual_income": 800000.0,
        "employment_type": "Salaried",
        "employment_stability_years": 5.0,
        "credit_score": 750,
        "existing_loans_count": 1,
        "existing_monthly_debt": 10000.0,
        "requested_loan_amount": 500000.0,
        "loan_tenure_months": 36,
        "dependents": 1,
        "savings_balance": 200000.0
    }
    result = validate_credit_input(valid_data)
    assert result.is_valid is True
    assert len(result.errors) == 0

def test_validate_credit_input_invalid_age():
    invalid_data = {
        "age": 15,  # Below 18
        "annual_income": 800000.0,
        "employment_type": "Salaried",
        "employment_stability_years": 5.0,
        "credit_score": 750,
        "existing_loans_count": 1,
        "existing_monthly_debt": 10000.0,
        "requested_loan_amount": 500000.0,
        "loan_tenure_months": 36,
        "dependents": 1,
        "savings_balance": 200000.0
    }
    result = validate_credit_input(invalid_data)
    assert result.is_valid is False
    assert any("age" in err.lower() for err in result.errors)

def test_validate_transaction_input_valid():
    valid_tx = {
        "amount": 1500.0,
        "transaction_type": "Payment",
        "hour_of_day": 14,
        "day_of_week": 2,
        "location_change": False,
        "new_device": False,
        "transaction_frequency_24h": 3,
        "customer_avg_amount": 2000.0,
        "distance_from_home_km": 10.0,
        "account_age_days": 365
    }
    result = validate_transaction_input(valid_tx)
    assert result.is_valid is True

def test_validate_transaction_csv():
    df = pd.DataFrame([
        {"date": "2026-01-01", "amount": 100.0, "category": "Food", "description": "Grocery"},
        {"date": "2026-01-02", "amount": -50.0, "category": "Shopping", "description": "Mall"}
    ])
    result = validate_transaction_csv(df)
    assert result.is_valid is True
