import pytest
import pandas as pd
from app.models.model_loader import load_credit_model, load_fraud_model

def test_credit_model_prediction():
    model = load_credit_model()
    assert model is not None
    assert model.is_trained is True

    sample_input = pd.DataFrame([{
        "age": 35,
        "annual_income": 1200000.0,
        "employment_type": "Salaried",
        "employment_stability_years": 8.0,
        "credit_score": 780,
        "existing_loans_count": 1,
        "existing_monthly_debt": 15000.0,
        "requested_loan_amount": 500000.0,
        "loan_tenure_months": 60,
        "dependents": 1,
        "savings_balance": 800000.0
    }])

    pred = model.predict(sample_input)
    assert "risk_label" in pred
    assert pred["risk_label"] in ["LOW", "MEDIUM", "HIGH"]
    assert "risk_score" in pred
    assert 0.0 <= pred["risk_score"] <= 1.0
    assert "probabilities" in pred

def test_fraud_model_prediction():
    model = load_fraud_model()
    assert model is not None
    assert model.is_trained is True

    sample_tx = pd.DataFrame([{
        "amount": 2500.0,
        "transaction_type": "Payment",
        "hour_of_day": 14,
        "day_of_week": 2,
        "location_change": False,
        "new_device": False,
        "transaction_frequency_24h": 2,
        "customer_avg_amount": 3000.0,
        "distance_from_home_km": 5.0,
        "account_age_days": 730
    }])

    pred = model.predict(sample_tx)
    assert "is_fraud" in pred
    assert isinstance(pred["is_fraud"], bool)
    assert "fraud_probability" in pred
    assert 0.0 <= pred["fraud_probability"] <= 1.0
