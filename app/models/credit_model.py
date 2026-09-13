import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from app.config.settings import (
    CREDIT_MODEL_PATH, CREDIT_PREPROCESSOR_PATH,
    RANDOM_SEED, CREDIT_MODEL_PARAMS
)
from app.data.preprocessing import (
    engineer_credit_features, get_credit_model_features,
    CREDIT_MODEL_FEATURES
)
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

class CreditModel:
    def __init__(self):
        self.model = None  # GradientBoostingClassifier
        self.scaler = None  # StandardScaler
        self.label_encoder = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
        self.label_decoder = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}
        self.is_trained = False
        self.feature_names = CREDIT_MODEL_FEATURES
        self.evaluation_metrics = {}
    
    def train(self, df: pd.DataFrame) -> dict:
        """Train on credit data. df has raw features + risk_label column.
        1. Engineer features using engineer_credit_features
        2. Extract model features using get_credit_model_features
        3. Encode risk_label to 0,1,2
        4. Train-test split (80/20, stratified, random_state=RANDOM_SEED)
        5. Fit StandardScaler on train
        6. Train GradientBoostingClassifier with CREDIT_MODEL_PARAMS
        7. Evaluate: accuracy, precision, recall, f1 (macro), confusion matrix, roc_auc (ovr)
        8. Store metrics in self.evaluation_metrics
        9. Return metrics dict
        """
        logger.info("Training credit model...")
        df_engineered = engineer_credit_features(df.copy())
        
        y = df_engineered['risk_label'].map(self.label_encoder)
        X = get_credit_model_features(df_engineered)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
        )
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = GradientBoostingClassifier(**CREDIT_MODEL_PARAMS)
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)
        
        self.evaluation_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='macro'),
            'recall': recall_score(y_test, y_pred, average='macro'),
            'f1': f1_score(y_test, y_pred, average='macro'),
            'roc_auc': roc_auc_score(y_test, y_prob, multi_class='ovr'),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        self.is_trained = True
        logger.info(f"Credit model trained. Metrics: {self.evaluation_metrics}")
        return self.evaluation_metrics
    
    def predict(self, df: pd.DataFrame) -> dict:
        """Predict on a single row or batch.
        1. Engineer features
        2. Extract model features
        3. Scale
        4. Predict class + predict_proba
        5. Return dict with: risk_label (str), risk_score (float 0-1), probabilities (dict), confidence (float)
        risk_score should be probability of HIGH risk class.
        confidence is the max probability.
        """
        if not self.is_trained:
            raise ValueError("Model is not trained.")
        
        df_engineered = engineer_credit_features(df.copy())
        X = get_credit_model_features(df_engineered)
        X_scaled = self.scaler.transform(X)
        
        preds = self.model.predict(X_scaled)
        probs = self.model.predict_proba(X_scaled)
        
        results = []
        for pred, prob in zip(preds, probs):
            results.append({
                'risk_label': self.label_decoder[pred],
                'risk_score': float(prob[2]),
                'probabilities': {
                    'LOW': float(prob[0]),
                    'MEDIUM': float(prob[1]),
                    'HIGH': float(prob[2])
                },
                'confidence': float(np.max(prob))
            })
            
        if len(results) == 1:
            return results[0]
        return results
        
    def get_feature_importances(self) -> dict:
        """Return {feature_name: importance_value} from model.feature_importances_"""
        if not self.is_trained:
            return {}
        return dict(zip(self.feature_names, self.model.feature_importances_))
    
    def save(self, model_path=None, preprocessor_path=None):
        """Save model and scaler using joblib to CREDIT_MODEL_PATH and CREDIT_PREPROCESSOR_PATH"""
        model_path = model_path or CREDIT_MODEL_PATH
        preprocessor_path = preprocessor_path or CREDIT_PREPROCESSOR_PATH
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, preprocessor_path)
        logger.info(f"Credit model saved to {model_path} and {preprocessor_path}")
    
    def load(self, model_path=None, preprocessor_path=None):
        """Load model and scaler from joblib files"""
        model_path = model_path or CREDIT_MODEL_PATH
        preprocessor_path = preprocessor_path or CREDIT_PREPROCESSOR_PATH
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(preprocessor_path)
        self.is_trained = True
        logger.info(f"Credit model loaded from {model_path} and {preprocessor_path}")
