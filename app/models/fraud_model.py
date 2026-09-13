import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from app.config.settings import (
    FRAUD_MODEL_PATH, FRAUD_PREPROCESSOR_PATH,
    RANDOM_SEED, FRAUD_MODEL_PARAMS
)
from app.data.preprocessing import (
    engineer_fraud_features, get_fraud_model_features,
    FRAUD_MODEL_FEATURES
)
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

class FraudModel:
    def __init__(self):
        self.model = None  # RandomForestClassifier
        self.scaler = None  # StandardScaler
        self.is_trained = False
        self.feature_names = FRAUD_MODEL_FEATURES
        self.evaluation_metrics = {}
    
    def train(self, df: pd.DataFrame) -> dict:
        """Train on transaction data. df has raw features + is_fraud column.
        1. Engineer features using engineer_fraud_features
        2. Extract model features using get_fraud_model_features
        3. Train-test split (80/20, stratified, random_state=RANDOM_SEED)
        4. Fit StandardScaler on train
        5. Train RandomForestClassifier with FRAUD_MODEL_PARAMS
        6. Evaluate: accuracy, precision, recall, f1, confusion matrix, roc_auc
        7. Store metrics
        8. Return metrics dict
        """
        logger.info("Training fraud model...")
        df_engineered = engineer_fraud_features(df.copy())
        
        y = df_engineered['is_fraud']
        X = get_fraud_model_features(df_engineered)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
        )
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = RandomForestClassifier(**FRAUD_MODEL_PARAMS)
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)
        
        self.evaluation_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob[:, 1]),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        self.is_trained = True
        logger.info(f"Fraud model trained. Metrics: {self.evaluation_metrics}")
        return self.evaluation_metrics
    
    def predict(self, df: pd.DataFrame) -> dict:
        """Predict fraud probability.
        Return dict with: is_fraud (bool), fraud_probability (float), confidence (float)
        fraud_probability is predict_proba for class 1.
        """
        if not self.is_trained:
            raise ValueError("Model is not trained.")
        
        df_engineered = engineer_fraud_features(df.copy())
        X = get_fraud_model_features(df_engineered)
        X_scaled = self.scaler.transform(X)
        
        preds = self.model.predict(X_scaled)
        probs = self.model.predict_proba(X_scaled)
        
        results = []
        for pred, prob in zip(preds, probs):
            results.append({
                'is_fraud': bool(pred),
                'fraud_probability': float(prob[1]),
                'confidence': float(np.max(prob))
            })
            
        if len(results) == 1:
            return results[0]
        return results
    
    def get_feature_importances(self) -> dict:
        """Return {feature_name: importance_value}"""
        if not self.is_trained:
            return {}
        return dict(zip(self.feature_names, self.model.feature_importances_))
    
    def save(self, model_path=None, preprocessor_path=None):
        """Save to FRAUD_MODEL_PATH and FRAUD_PREPROCESSOR_PATH"""
        model_path = model_path or FRAUD_MODEL_PATH
        preprocessor_path = preprocessor_path or FRAUD_PREPROCESSOR_PATH
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, preprocessor_path)
        logger.info(f"Fraud model saved to {model_path} and {preprocessor_path}")
    
    def load(self, model_path=None, preprocessor_path=None):
        """Load from joblib files"""
        model_path = model_path or FRAUD_MODEL_PATH
        preprocessor_path = preprocessor_path or FRAUD_PREPROCESSOR_PATH
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(preprocessor_path)
        self.is_trained = True
        logger.info(f"Fraud model loaded from {model_path} and {preprocessor_path}")
