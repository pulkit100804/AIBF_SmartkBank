"""
Orchestrates the credit assessment pipeline.
"""
import pandas as pd
from app.utils.validation import validate_credit_input
from app.models.model_loader import load_credit_model
from app.modules.credit_decision import classify_credit_risk, get_loan_recommendation
from app.services.explanation_service import explain_credit_prediction
from app.data.repository import save_loan_application
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

def assess_credit(input_data: dict) -> dict:
    """Full credit assessment pipeline."""
    try:
        validation_result = validate_credit_input(input_data)
        if not validation_result.is_valid:
            return {"success": False, "errors": validation_result.errors}
            
        model = load_credit_model()
        if not model:
            return {"success": False, "errors": ["Failed to load credit model."]}
            
        df = pd.DataFrame([input_data])
        
        prediction = model.predict(df)
        
        risk_label = prediction.get('risk_label')
        risk_score = prediction.get('risk_score')
        confidence = prediction.get('confidence')
        probabilities = prediction.get('probabilities')
        
        decision_info = classify_credit_risk(risk_label, risk_score, input_data)
        decision = decision_info.get("decision")
        
        recommendation = get_loan_recommendation(decision, input_data)
        
        explanation = explain_credit_prediction(model, input_data, prediction)
        
        app_id = save_loan_application(
            input_data=input_data,
            risk_label=risk_label,
            risk_score=risk_score,
            decision=decision,
            confidence=confidence,
            explanation=explanation
        )
        
        return {
            "success": True,
            "application_id": app_id,
            "risk_label": risk_label,
            "risk_score": risk_score,
            "decision": decision,
            "confidence": confidence,
            "recommendation": recommendation,
            "explanation": explanation,
            "probabilities": probabilities,
            "reasons": decision_info.get("reasons")
        }
    except Exception as e:
        logger.error(f"Error in assess_credit: {e}", exc_info=True)
        return {"success": False, "errors": [str(e)]}
