"""
Business logic on top of ML prediction for credit decisions.
"""

def classify_credit_risk(risk_label: str, risk_score: float, input_data: dict) -> dict:
    """Map ML prediction to business decision."""
    annual_income = input_data.get('annual_income', 0)
    existing_monthly_debt = input_data.get('existing_monthly_debt', 0)
    credit_score = input_data.get('credit_score', 0)
    
    dti = (existing_monthly_debt * 12 / annual_income) if annual_income > 0 else 1.0
    
    reasons = []
    decision = 'REVIEW'
    
    if risk_label == 'LOW':
        if dti < 0.40 and credit_score >= 650:
            decision = 'APPROVE'
            reasons.append("Low risk model prediction.")
            reasons.append("Healthy debt-to-income ratio.")
            reasons.append("Good credit score.")
        else:
            if dti >= 0.40:
                reasons.append(f"High debt-to-income ratio: {dti:.2f}.")
            if credit_score < 650:
                reasons.append(f"Credit score ({credit_score}) below automatic approval threshold.")
    elif risk_label == 'HIGH':
        decision = 'REJECT'
        reasons.append("High risk model prediction.")
    else:
        # Default fallback
        if dti > 0.65 or credit_score < 550:
            decision = 'REJECT'
            reasons.append("Debt-to-income ratio critically high or credit score too low.")
        else:
            reasons.append("Requires manual review.")

    if dti > 0.65 or credit_score < 550:
        decision = 'REJECT'
        reasons.append("Credit score too low or DTI too high for approval.")
        
    return {
        "decision": decision,
        "reasons": list(set(reasons))
    }

def get_loan_recommendation(decision: str, input_data: dict) -> str:
    """Generate a short recommendation string based on decision and context."""
    if decision == 'APPROVE':
        return "Recommended for approval with standard terms."
    elif decision == 'REJECT':
        return "Not recommended for approval due to high risk factors."
    else:
        return "Manual review recommended. Please review borderline factors such as credit score and DTI."
