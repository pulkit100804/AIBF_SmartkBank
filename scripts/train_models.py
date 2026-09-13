#!/usr/bin/env python
"""Train all ML models for SmartBank AI."""
import sys
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.models.credit_model import CreditModel
from app.models.fraud_model import FraudModel
from app.config.settings import CREDIT_DATA_PATH, TRANSACTION_DATA_PATH

def main():
    print("="*50)
    print("Starting SmartBank AI Model Training")
    print("="*50)
    
    print("\n[1] Training Credit Risk Model...")
    credit_df = pd.read_csv(CREDIT_DATA_PATH)
    credit_model = CreditModel()
    credit_metrics = credit_model.train(credit_df)
    credit_model.save()
    
    print("Credit Model Metrics:")
    for metric, value in credit_metrics.items():
        print(f"  {metric}: {value}")
        
    print("\n[2] Training Fraud Detection Model...")
    fraud_df = pd.read_csv(TRANSACTION_DATA_PATH)
    fraud_model = FraudModel()
    fraud_metrics = fraud_model.train(fraud_df)
    fraud_model.save()
    
    print("Fraud Model Metrics:")
    for metric, value in fraud_metrics.items():
        print(f"  {metric}: {value}")
        
    print("\n" + "="*50)
    print("Model Training Completed Successfully!")
    print("="*50)

if __name__ == '__main__':
    main()
