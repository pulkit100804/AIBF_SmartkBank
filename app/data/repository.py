"""
Repository for database operations.
"""

import json
from sqlalchemy import func
from app.database.db import get_session, init_db
from app.database.schema import LoanApplication, FraudAlert, FinancialInsight
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

def save_loan_application(input_data: dict, risk_label: str, risk_score: float, decision: str, confidence: float, explanation: dict, applicant_name: str = 'Anonymous') -> int:
    """Save a loan assessment. Return the ID."""
    session = get_session()
    try:
        app = LoanApplication(
            applicant_name=applicant_name,
            input_data=json.dumps(input_data),
            risk_label=risk_label,
            risk_score=risk_score,
            decision=decision,
            confidence=confidence,
            explanation=json.dumps(explanation)
        )
        session.add(app)
        session.commit()
        session.refresh(app)
        return app.id
    except Exception as e:
        logger.error(f"Error saving loan application: {e}")
        session.rollback()
        raise
    finally:
        session.close()

def get_recent_loan_applications(limit: int = 10) -> list[dict]:
    """Get recent loan applications as list of dicts."""
    session = get_session()
    try:
        apps = session.query(LoanApplication).order_by(LoanApplication.created_at.desc()).limit(limit).all()
        return [
            {
                "id": a.id,
                "applicant_name": a.applicant_name,
                "input_data": json.loads(a.input_data) if a.input_data else {},
                "risk_label": a.risk_label,
                "risk_score": a.risk_score,
                "decision": a.decision,
                "confidence": a.confidence,
                "explanation": json.loads(a.explanation) if a.explanation else {},
                "created_at": a.created_at.isoformat()
            } for a in apps
        ]
    except Exception as e:
        logger.error(f"Error getting recent loan applications: {e}")
        return []
    finally:
        session.close()

def get_loan_application_count() -> int:
    """Count total loan applications."""
    session = get_session()
    try:
        return session.query(LoanApplication).count()
    except Exception as e:
        logger.error(f"Error counting loan applications: {e}")
        return 0
    finally:
        session.close()

def save_fraud_alert(input_data: dict, fraud_probability: float, risk_level: str, recommended_action: str, explanation: dict, transaction_id: str = 'N/A') -> int:
    """Save a fraud alert. Return the ID."""
    session = get_session()
    try:
        alert = FraudAlert(
            transaction_id=transaction_id,
            input_data=json.dumps(input_data),
            fraud_probability=fraud_probability,
            risk_level=risk_level,
            recommended_action=recommended_action,
            explanation=json.dumps(explanation)
        )
        session.add(alert)
        session.commit()
        session.refresh(alert)
        return alert.id
    except Exception as e:
        logger.error(f"Error saving fraud alert: {e}")
        session.rollback()
        raise
    finally:
        session.close()

def get_recent_fraud_alerts(limit: int = 10) -> list[dict]:
    """Get recent fraud alerts as list of dicts."""
    session = get_session()
    try:
        alerts = session.query(FraudAlert).order_by(FraudAlert.created_at.desc()).limit(limit).all()
        return [
            {
                "id": a.id,
                "transaction_id": a.transaction_id,
                "input_data": json.loads(a.input_data) if a.input_data else {},
                "fraud_probability": a.fraud_probability,
                "risk_level": a.risk_level,
                "recommended_action": a.recommended_action,
                "explanation": json.loads(a.explanation) if a.explanation else {},
                "created_at": a.created_at.isoformat()
            } for a in alerts
        ]
    except Exception as e:
        logger.error(f"Error getting recent fraud alerts: {e}")
        return []
    finally:
        session.close()

def get_fraud_alert_count() -> int:
    """Count total fraud alerts."""
    session = get_session()
    try:
        return session.query(FraudAlert).count()
    except Exception as e:
        logger.error(f"Error counting fraud alerts: {e}")
        return 0
    finally:
        session.close()

def get_high_risk_fraud_count() -> int:
    """Count HIGH risk fraud alerts."""
    session = get_session()
    try:
        return session.query(FraudAlert).filter(FraudAlert.risk_level == 'HIGH').count()
    except Exception as e:
        logger.error(f"Error counting high risk fraud alerts: {e}")
        return 0
    finally:
        session.close()

def save_financial_insight(customer_name: str, period: str, summary_data: dict) -> int:
    """Save financial insight. Return ID."""
    session = get_session()
    try:
        insight = FinancialInsight(
            customer_name=customer_name,
            period=period,
            summary_data=json.dumps(summary_data)
        )
        session.add(insight)
        session.commit()
        session.refresh(insight)
        return insight.id
    except Exception as e:
        logger.error(f"Error saving financial insight: {e}")
        session.rollback()
        raise
    finally:
        session.close()

def get_recent_insights(limit: int = 10) -> list[dict]:
    """Get recent insights."""
    session = get_session()
    try:
        insights = session.query(FinancialInsight).order_by(FinancialInsight.created_at.desc()).limit(limit).all()
        return [
            {
                "id": i.id,
                "customer_name": i.customer_name,
                "period": i.period,
                "summary_data": json.loads(i.summary_data) if i.summary_data else {},
                "created_at": i.created_at.isoformat()
            } for i in insights
        ]
    except Exception as e:
        logger.error(f"Error getting recent insights: {e}")
        return []
    finally:
        session.close()

def get_insight_count() -> int:
    """Count insights."""
    session = get_session()
    try:
        return session.query(FinancialInsight).count()
    except Exception as e:
        logger.error(f"Error counting insights: {e}")
        return 0
    finally:
        session.close()
