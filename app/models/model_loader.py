from app.models.credit_model import CreditModel
from app.models.fraud_model import FraudModel
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

_credit_model_instance = None
_fraud_model_instance = None

def load_credit_model() -> CreditModel:
    """Load or return cached CreditModel. If artifact not found, auto-train from data."""
    global _credit_model_instance
    if _credit_model_instance is not None:
        return _credit_model_instance
    model = CreditModel()
    try:
        model.load()
    except FileNotFoundError:
        # Auto-train fallback
        logger.warning('Credit model artifacts not found, training...')
        from app.data.generator import generate_credit_data
        df = generate_credit_data()
        model.train(df)
        model.save()
    _credit_model_instance = model
    return model

def load_fraud_model() -> FraudModel:
    """Load or return cached FraudModel. If artifact not found, auto-train from data."""
    global _fraud_model_instance
    if _fraud_model_instance is not None:
        return _fraud_model_instance
    model = FraudModel()
    try:
        model.load()
    except FileNotFoundError:
        # Auto-train fallback
        logger.warning('Fraud model artifacts not found, training...')
        from app.data.generator import generate_transaction_data
        df = generate_transaction_data()
        model.train(df)
        model.save()
    _fraud_model_instance = model
    return model
