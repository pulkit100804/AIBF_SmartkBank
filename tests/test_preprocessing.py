import pytest
import pandas as pd
from app.data.preprocessing import (
    engineer_credit_features,
    get_credit_model_features,
    engineer_fraud_features,
    get_fraud_model_features,
    CREDIT_MODEL_FEATURES,
    FRAUD_MODEL_FEATURES
)

def test_credit_feature_engineering():
    df = pd.DataFrame([{
        "age": 30,
        "annual_income": 600000.0,
        "employment_type": "Salaried",
        "employment_stability_years": 4.0,
        "credit_score": 700,
        "existing_loans_count": 1,
        "existing_monthly_debt": 15000.0,
        "requested_loan_amount": 300000.0,
        "loan_tenure_months": 36,
        "dependents": 1,
        "savings_balance": 100000.0
    }])
    df_eng = engineer_credit_features(df)
    assert "dti_ratio" in df_eng.columns
    assert "lti_ratio" in df_eng.columns
    assert "savings_ratio" in df_eng.columns
    assert df_eng.loc[0, "dti_ratio"] == pytest.approx((15000 * 12) / 600000.0)

def test_credit_model_features_extraction():
    df = pd.DataFrame([{
        "age": 30,
        "annual_income": 600000.0,
        "employment_type": "Salaried",
        "employment_stability_years": 4.0,
        "credit_score": 700,
        "existing_loans_count": 1,
        "existing_monthly_debt": 15000.0,
        "requested_loan_amount": 300000.0,
        "loan_tenure_months": 36,
        "dependents": 1,
        "savings_balance": 100000.0
    }])
    df_eng = engineer_credit_features(df)
    features_df = get_credit_model_features(df_eng)
    for feat in CREDIT_MODEL_FEATURES:
        assert feat in features_df.columns

def test_fraud_feature_engineering():
    df = pd.DataFrame([{
        "amount": 10000.0,
        "transaction_type": "Transfer",
        "hour_of_day": 3,
        "day_of_week": 6,
        "location_change": True,
        "new_device": True,
        "transaction_frequency_24h": 12,
        "customer_avg_amount": 2000.0,
        "distance_from_home_km": 100.0,
        "account_age_days": 45
    }])
    df_eng = engineer_fraud_features(df)
    assert "amount_ratio" in df_eng.columns
    assert "is_night" in df_eng.columns
    assert "risk_composite" in df_eng.columns
    assert df_eng.loc[0, "amount_ratio"] == pytest.approx(10000.0 / 2000.0)
    assert df_eng.loc[0, "is_night"] == 1
