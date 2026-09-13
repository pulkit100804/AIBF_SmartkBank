"""
Fraud risk classification and action mapping.
"""

from app.config.settings import FRAUD_LOW_THRESHOLD, FRAUD_HIGH_THRESHOLD

def classify_fraud_risk(fraud_probability: float) -> str:
    """Return LOW / MEDIUM / HIGH based on thresholds."""
    if fraud_probability >= FRAUD_HIGH_THRESHOLD:
        return 'HIGH'
    elif fraud_probability >= FRAUD_LOW_THRESHOLD:
        return 'MEDIUM'
    return 'LOW'

def determine_action(risk_level: str, fraud_probability: float) -> str:
    """Map risk level to action: Allow / Monitor / Additional Verification / Block"""
    if risk_level == 'HIGH':
        return 'Block'
    elif risk_level == 'MEDIUM':
        if fraud_probability > (FRAUD_LOW_THRESHOLD + FRAUD_HIGH_THRESHOLD) / 2:
            return 'Additional Verification'
        return 'Monitor'
    return 'Allow'

def get_fraud_summary(risk_level: str, fraud_probability: float, action: str) -> str:
    """Generate a summary string for the fraud assessment."""
    return f"Risk Level: {risk_level} ({fraud_probability:.1%} fraud probability). Recommended Action: {action}."
