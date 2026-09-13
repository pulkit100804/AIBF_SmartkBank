import pytest
from app.database.db import init_db
from app.services.credit_service import assess_credit
from app.services.fraud_service import assess_transaction
from app.services.insights_service import analyze_customer_finances

@pytest.fixture(autouse=True)
def setup_database():
    init_db()

def test_credit_service_assessment():
    input_data = {
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
    }
    result = assess_credit(input_data)
    assert result["success"] is True, f"Failed with errors: {result.get('errors')}"
    assert result["decision"] in ["APPROVE", "REVIEW", "REJECT"]
    assert "explanation" in result
    assert "top_factors" in result["explanation"]

def test_fraud_service_assessment():
    tx_data = {
        "amount": 85000.0,
        "transaction_type": "Transfer",
        "hour_of_day": 3,
        "day_of_week": 6,
        "location_change": True,
        "new_device": True,
        "transaction_frequency_24h": 14,
        "customer_avg_amount": 5000.0,
        "distance_from_home_km": 450.0,
        "account_age_days": 45
    }
    result = assess_transaction(tx_data)
    assert result["success"] is True, f"Failed with errors: {result.get('errors')}"
    assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert "action" in result
    assert "explanation" in result

def test_insights_service_demo_profile():
    result = analyze_customer_finances(customer_profile="Conservative Saver")
    assert result["success"] is True, f"Failed with errors: {result.get('errors')}"
    assert "analysis_result" in result
    analysis = result["analysis_result"]
    assert "total_income" in analysis
    assert "total_expenses" in analysis
    assert "savings_rate" in analysis
    assert "recommendations" in analysis
