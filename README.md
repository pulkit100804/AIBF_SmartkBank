# SmartBank AI: AI-Powered Banking Risk, Fraud & Personalized Finance Assistant

**CA-2 Coding Assignment: Artificial Intelligence in Banking & Finance (AIBF)**  
*Educational Production-Grade FinTech Prototype*

---

## 📌 Executive Summary

**SmartBank AI** is an enterprise-grade, full-stack FinTech application integrating Machine Learning, Explainable AI (XAI), and automated decision engines to solve critical challenges in modern digital banking:

1. **Credit Risk Assessment & Loan Underwriting**: Uses a multi-class Gradient Boosting Classifier to predict applicant credit risk (`LOW`, `MEDIUM`, `HIGH`), compute default probabilities, and enforce automated business underwriting rules (`APPROVE`, `REVIEW`, `REJECT`).
2. **Real-time Fraud Detection**: Employs a weighted Random Forest Classifier with imbalance handling to evaluate transaction risk in real time, flag anomalous high-risk transactions, and recommend instant defense actions (`ALLOW`, `MONITOR`, `STEP-UP AUTH`, `BLOCK`).
3. **Personalized Financial Insights & Anomaly Detection**: Analyzes monthly transaction histories, calculates savings rates, detects spending anomalies (>2 standard deviations from category means), and delivers customized financial health recommendations.
4. **Explainable AI (XAI) Engine**: Provides transparent model explanations by computing top feature importances and translating complex mathematical model weights into clear human-readable factors.

---

## 🚀 Machine Learning Models & Performance Architecture

SmartBank AI avoids hardcoded dummy logic by training robust machine learning models on synthetic domain-specific datasets:

| Model Task | Algorithm | Features | Metrics | Artifact Path |
| :--- | :--- | :--- | :--- | :--- |
| **Credit Risk Scoring** | `GradientBoostingClassifier` (3-Class) | 10 engineered features | **Accuracy**: 74.1%<br>**ROC-AUC**: 0.861<br>**F1-Score**: 0.635 | `artifacts/credit_model.joblib`<br>`artifacts/credit_preprocessor.joblib` |
| **Fraud Detection** | `RandomForestClassifier` (Binary, Balanced) | 10 engineered features | **Accuracy**: 99.95%<br>**Recall**: 100.0%<br>**ROC-AUC**: 0.9999 | `artifacts/fraud_model.joblib`<br>`artifacts/fraud_preprocessor.joblib` |

### Key Feature Engineering:
- **Credit Risk**: Debt-to-Income (DTI) ratio (`existing_monthly_debt * 12 / annual_income`), Loan-to-Income (LTI) ratio (`requested_loan / annual_income`), Savings-to-Loan ratio (`savings / requested_loan`).
- **Fraud Detection**: Amount Ratio (`amount / customer_avg_amount`), Night Transaction flag (`is_night` between 11 PM - 5 AM), High Frequency flag (`transaction_frequency_24h > 10`).

---

## 🛠️ Project Structure

```text
smartbank-ai/
│
├── app/
│   ├── config/
│   │   └── settings.py           # Core system configuration & hyperparameters
│   ├── data/
│   │   ├── generator.py          # Synthetic dataset generator
│   │   ├── preprocessing.py      # Feature engineering pipelines
│   │   └── repository.py         # SQLAlchemy DB CRUD repository
│   ├── database/
│   │   ├── db.py                 # SQLite database engine & session maker
│   │   └── schema.py             # ORM entities (LoanApplication, FraudAlert, etc.)
│   ├── models/
│   │   ├── credit_model.py       # Gradient Boosting credit classifier
│   │   ├── fraud_model.py        # Random Forest fraud detector
│   │   └── model_loader.py       # Cached singleton model loader
│   ├── modules/
│   │   ├── credit_decision.py    # Business decision mapping
│   │   ├── fraud_detection.py    # Fraud risk tiering & action mapping
│   │   └── financial_insights.py # Anomaly detection & analytics
│   ├── services/
│   │   ├── credit_service.py     # Credit assessment pipeline coordinator
│   │   ├── fraud_service.py      # Fraud evaluation pipeline coordinator
│   │   ├── insights_service.py   # Insights pipeline coordinator
│   │   └── explanation_service.py# XAI feature importance engine
│   ├── ui/
│   │   ├── components.py         # Reusable Streamlit & Plotly widgets
│   │   ├── dashboard.py          # Executive overview page
│   │   ├── credit_page.py        # Loan underwriting interactive UI
│   │   ├── fraud_page.py         # Fraud monitoring & simulator UI
│   │   ├── insights_page.py      # Spending & anomaly dashboard UI
│   │   └── explainability_page.py# XAI & Responsible AI audit UI
│   ├── utils/
│   │   ├── helpers.py            # Formatting & styling helpers
│   │   ├── logging_config.py     # Centralized logging setup
│   │   └── validation.py         # Pydantic-like input validation
│   └── main.py                   # Streamlit app main entrypoint
│
├── data/
│   ├── credit_applications.csv   # Synthetic credit dataset (5,000 records)
│   └── transactions.csv          # Synthetic transaction dataset (10,000 records)
│
├── artifacts/                     # Saved joblib models & scalers
├── scripts/
│   ├── generate_data.py          # Dataset creation runner script
│   └── train_models.py           # ML training & evaluation script
├── tests/                        # Comprehensive PyTest suite (10 tests)
├── Dockerfile                    # Containerization configuration
├── docker-compose.yml            # Multi-container orchestration
├── requirements.txt              # Production dependency manifest
└── README.md                     # Technical documentation
```

---

## ⚡ Quick Start & Setup Guide

### 1. Installation

Clone/navigate to the project folder and install dependencies:

```bash
cd smartbank-ai
python -m pip install -r requirements.txt
```

### 2. Generate Synthetic Datasets

Generate 5,000 credit records and 10,000 transaction records:

```bash
python scripts/generate_data.py
```

### 3. Train Machine Learning Models

Train the Gradient Boosting credit model and Random Forest fraud model:

```bash
python scripts/train_models.py
```

### 4. Run Unit Tests

Execute the 10-test validation suite:

```bash
python -m pytest tests/ -v
```

### 5. Launch the Web Interface

Launch the interactive Streamlit dashboard:

```bash
python -m streamlit run app/main.py
```

Open your browser at `http://localhost:8501`.

---

## 🐳 Docker Deployment

To build and run SmartBank AI inside a Docker container:

```bash
# Build the Docker image
docker build -t smartbank-ai .

# Run the container
docker run -p 8501:8501 smartbank-ai
```

Alternatively using `docker-compose`:

```bash
docker-compose up --build
```

---

## 🛡️ Responsible AI & Compliance Notice

SmartBank AI incorporates Responsible AI principles:
1. **Explainability**: Every decision includes non-black-box feature importance analysis.
2. **Human Oversight**: Decisions labeled `REVIEW` require manual underwriter verification.
3. **Data Privacy**: No real PII is collected or processed; all data is synthetic.
4. **Educational Scope**: Designed strictly for academic demonstration and CA-2 project presentation.

---
*Developed for College CA-2 Coding Assignment: Artificial Intelligence in Banking & Finance.*
