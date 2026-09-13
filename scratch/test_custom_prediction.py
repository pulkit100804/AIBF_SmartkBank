import sys
from pathlib import Path
project_root = Path("c:/Users/pulki/OneDrive/Desktop/AIBF/smartbank-ai")
sys.path.insert(0, str(project_root))

import joblib
from app.models.model_loader import load_credit_model, load_fraud_model
from app.services.credit_service import assess_credit
from app.services.fraud_service import assess_transaction

print("="*70)
print("VERIFYING TRAINED ML MODELS & BACKEND INFERENCE ENGINE")
print("="*70)

# 1. Inspect loaded models
credit_model = load_credit_model()
fraud_model = load_fraud_model()

print(f"\n[1] Credit Risk Model Type : {type(credit_model.model).__name__}")
print(f"    Features Expected       : {len(credit_model.feature_names)} features")
print(f"    Scaler Class            : {type(credit_model.scaler).__name__}")
print(f"    Classes                 : {credit_model.model.classes_}")

print(f"\n[2] Fraud Detection Model Type: {type(fraud_model.model).__name__}")
print(f"    Features Expected         : {len(fraud_model.feature_names)} features")
print(f"    Trees in Random Forest    : {len(fraud_model.model.estimators_)}")

print("\n" + "="*70)
print("TESTING CUSTOM UNSEEN INPUT 1: HIGH INCOME, PERFECT CREDIT (CUSTOM APPLICANT)")
print("="*70)

custom_credit_1 = {
    "age": 45,
    "annual_income": 2500000.0,
    "employment_type": "Salaried",
    "employment_stability_years": 12.0,
    "credit_score": 820,
    "existing_loans_count": 0,
    "existing_monthly_debt": 5000.0,
    "requested_loan_amount": 1000000.0,
    "loan_tenure_months": 48,
    "dependents": 1,
    "savings_balance": 1500000.0
}

res1 = assess_credit(custom_credit_1)
print(f"Predicted Risk Label       : {res1['risk_label']}")
print(f"ML Default Risk Score      : {res1['risk_score']:.4f}")
print(f"Class Probabilities        : {res1['probabilities']}")
print(f"Underwriting Decision      : {res1['decision']}")
print(f"Recommendation             : {res1['recommendation']}")
print("Top Model Factors          :")
for f in res1['explanation']['top_factors'][:3]:
    print(f"  - {f['label']}: Value={f['value']}, Direction={f['direction']}, Impact={f['impact']}")

print("\n" + "="*70)
print("TESTING CUSTOM UNSEEN INPUT 2: HIGH DEBT, LOW CREDIT SCORE (RISKY APPLICANT)")
print("="*70)

custom_credit_2 = {
    "age": 24,
    "annual_income": 350000.0,
    "employment_type": "Freelance",
    "employment_stability_years": 1.0,
    "credit_score": 510,
    "existing_loans_count": 4,
    "existing_monthly_debt": 22000.0,
    "requested_loan_amount": 900000.0,
    "loan_tenure_months": 24,
    "dependents": 3,
    "savings_balance": 10000.0
}

res2 = assess_credit(custom_credit_2)
print(f"Predicted Risk Label       : {res2['risk_label']}")
print(f"ML Default Risk Score      : {res2['risk_score']:.4f}")
print(f"Class Probabilities        : {res2['probabilities']}")
print(f"Underwriting Decision      : {res2['decision']}")
print(f"Recommendation             : {res2['recommendation']}")
print("Top Model Factors          :")
for f in res2['explanation']['top_factors'][:3]:
    print(f"  - {f['label']}: Value={f['value']}, Direction={f['direction']}, Impact={f['impact']}")

print("\n" + "="*70)
print("TESTING CUSTOM UNSEEN INPUT 3: HIGH-RISK SUSPICIOUS TRANSACTION (FRAUD DETECTION)")
print("="*70)

custom_fraud_tx = {
    "amount": 150000.0,  # 30x customer average!
    "transaction_type": "Transfer",
    "hour_of_day": 2,     # 2 AM (Night)
    "day_of_week": 6,     # Sunday
    "location_change": True,
    "new_device": True,
    "transaction_frequency_24h": 18,
    "customer_avg_amount": 5000.0,
    "distance_from_home_km": 850.0,
    "account_age_days": 15
}

res3 = assess_transaction(custom_fraud_tx)
print(f"Fraud Detected by RF Model : {res3['is_fraud']}")
print(f"ML Fraud Probability       : {res3['fraud_probability']:.4f}")
print(f"Assessed Risk Tier         : {res3['risk_level']}")
print(f"Recommended Action         : {res3['action']}")
print("Top Fraud Factors          :")
for f in res3['explanation']['top_factors'][:3]:
    print(f"  - {f['label']}: Value={f['value']}, Direction={f['direction']}, Impact={f['impact']}")

print("\n" + "="*70)
print("VERIFICATION COMPLETE: REAL ML MODELS ARE ACTIVE AND INFERRING IN REAL TIME!")
print("="*70)
