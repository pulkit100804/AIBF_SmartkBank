"""
Explainability service using feature importance.
"""
from app.config.settings import CREDIT_FEATURE_LABELS, FRAUD_FEATURE_LABELS

def explain_credit_prediction(model, input_data: dict, prediction: dict) -> dict:
    """Generate explanation for credit prediction."""
    importances = model.get_feature_importances()
    
    sorted_features = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)
    top_5 = sorted_features[:5]
    
    top_factors = []
    
    dti_ratio = 0
    if input_data.get('annual_income', 0) > 0:
        dti_ratio = (input_data.get('existing_monthly_debt', 0) * 12) / input_data.get('annual_income', 1)
        
    lti_ratio = 0
    if input_data.get('annual_income', 0) > 0:
        lti_ratio = input_data.get('requested_loan_amount', 0) / input_data.get('annual_income', 1)

    for feat, imp in top_5:
        val = input_data.get(feat, None)
        
        direction = "unknown"
        if feat == 'credit_score':
            direction = "increases risk" if val and val < 650 else "decreases risk"
        elif feat == 'dti_ratio':
            direction = "increases risk" if dti_ratio > 0.4 else "decreases risk"
            val = round(dti_ratio, 2)
        elif feat == 'lti_ratio':
            direction = "increases risk" if lti_ratio > 2.0 else "decreases risk"
            val = round(lti_ratio, 2)
        elif feat == 'employment_stability_years':
            direction = "increases risk" if val and val < 2 else "decreases risk"
        elif feat == 'savings_balance':
            direction = "decreases risk" if val and val > 5000 else "increases risk"
        else:
            direction = "increases risk" if imp > 0 else "decreases risk"
            
        label = CREDIT_FEATURE_LABELS.get(feat, feat.replace('_', ' ').title())
        top_factors.append({
            "feature": feat,
            "label": label,
            "importance": float(imp),
            "value": val,
            "direction": direction,
            "impact": "High"
        })
        
    plain_english = _generate_plain_english(top_factors, "credit", prediction)
    
    return {
        "top_factors": top_factors,
        "plain_english": plain_english,
        "feature_importances": {k: float(v) for k, v in importances.items()}
    }

def explain_fraud_prediction(model, input_data: dict, prediction: dict) -> dict:
    """Generate explanation for fraud prediction."""
    importances = model.get_feature_importances()
    
    sorted_features = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)
    top_5 = sorted_features[:5]
    
    top_factors = []
    
    for feat, imp in top_5:
        val = input_data.get(feat, None)
        
        direction = "unknown"
        if feat == 'amount_ratio':
            direction = "increases risk" if val and val > 3 else "decreases risk"
        elif feat == 'location_change':
            direction = "increases risk" if val == 1 or val is True else "decreases risk"
        elif feat == 'new_device':
            direction = "increases risk" if val == 1 or val is True else "decreases risk"
        elif feat == 'is_night':
            direction = "increases risk" if val == 1 or val is True else "decreases risk"
        elif feat == 'transaction_frequency_24h':
            direction = "increases risk" if val and val > 10 else "decreases risk"
        elif feat == 'distance_from_home_km':
            direction = "increases risk" if val and val > 50 else "decreases risk"
        elif feat == 'account_age_days':
            direction = "increases risk" if val and val < 90 else "decreases risk"
        else:
            direction = "increases risk" if imp > 0 else "decreases risk"
            
        label = FRAUD_FEATURE_LABELS.get(feat, feat.replace('_', ' ').title())
        top_factors.append({
            "feature": feat,
            "label": label,
            "importance": float(imp),
            "value": val,
            "direction": direction,
            "impact": "High"
        })
        
    plain_english = _generate_plain_english(top_factors, "fraud", prediction)
    
    return {
        "top_factors": top_factors,
        "plain_english": plain_english,
        "feature_importances": {k: float(v) for k, v in importances.items()}
    }

def _generate_plain_english(factors: list, context: str, prediction: dict) -> str:
    """Convert structured factors into a readable paragraph."""
    conf = prediction.get('confidence', 0) * 100
    
    if context == "credit":
        risk_label = prediction.get('risk_label', 'UNKNOWN')
        intro = f"The assessment indicates {risk_label} risk (confidence: {conf:.1f}%)."
    else:
        prob = prediction.get('fraud_probability', 0) * 100
        risk_level = "HIGH" if prob > 60 else "MEDIUM" if prob > 30 else "LOW"
        intro = f"The transaction indicates {risk_level} fraud risk (probability: {prob:.1f}%, confidence: {conf:.1f}%)."
        
    if not factors:
        return f"{intro} No significant factors were identified."
        
    factors_text = []
    for i, f in enumerate(factors, 1):
        factors_text.append(f"({i}) {f['label']} of {f['value']}, which {f['direction']}")
        
    body = " The primary factors are: " + "; ".join(factors_text) + "."
    
    return intro + body
