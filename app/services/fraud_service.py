"""
Orchestrates fraud detection pipeline.
"""
import pandas as pd
from app.utils.validation import validate_transaction_input
from app.models.model_loader import load_fraud_model
from app.modules.fraud_detection import classify_fraud_risk, determine_action, get_fraud_summary
from app.services.explanation_service import explain_fraud_prediction
from app.data.repository import save_fraud_alert
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

def assess_transaction(input_data: dict) -> dict:
    """Full fraud assessment pipeline."""
    try:
        validation_result = validate_transaction_input(input_data)
        if not validation_result.is_valid:
            return {"success": False, "errors": validation_result.errors}
            
        model = load_fraud_model()
        if not model:
            return {"success": False, "errors": ["Failed to load fraud model."]}
            
        df = pd.DataFrame([input_data])
        
        prediction = model.predict(df)
        fraud_probability = prediction.get('fraud_probability', 0.0)
        is_fraud = prediction.get('is_fraud', False)
        confidence = prediction.get('confidence', 0.0)
        
        risk_level = classify_fraud_risk(fraud_probability)
        action = determine_action(risk_level, fraud_probability)
        
        explanation = explain_fraud_prediction(model, input_data, prediction)
        
        alert_id = None
        if risk_level in ['MEDIUM', 'HIGH'] or is_fraud:
            alert_id = save_fraud_alert(
                input_data=input_data,
                fraud_probability=fraud_probability,
                risk_level=risk_level,
                recommended_action=action,
                explanation=explanation
            )
            
        summary = get_fraud_summary(risk_level, fraud_probability, action)
            
        return {
            "success": True,
            "alert_id": alert_id,
            "fraud_probability": fraud_probability,
            "risk_level": risk_level,
            "action": action,
            "confidence": confidence,
            "explanation": explanation,
            "is_fraud": is_fraud,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error in assess_transaction: {e}", exc_info=True)
        return {"success": False, "errors": [str(e)]}
