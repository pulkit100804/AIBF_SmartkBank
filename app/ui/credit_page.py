import streamlit as st
import pandas as pd
import plotly.express as px
from app.services import credit_service
from app.ui.components import gauge_chart, factor_chart, decision_banner, workflow_stepper
from app.config.settings import EMPLOYMENT_TYPES
from app.data import repository

def render_credit_page():
    """Credit Decision page."""
    st.title("Credit Decision Engine")
    st.write("AI-powered assessment for loan applications.")

    demo_profiles = {
        'Custom Input': {},
        'Healthy Credit Profile': {
            'age': 35, 'annual_income': 1200000, 'employment_type': 'Salaried',
            'employment_stability_years': 8, 'credit_score': 780, 'existing_loans_count': 1,
            'existing_monthly_debt': 15000, 'requested_loan_amount': 500000, 'loan_tenure_months': 60,
            'dependents': 1, 'savings_balance': 800000
        },
        'Borderline Credit Profile': {
            'age': 42, 'annual_income': 600000, 'employment_type': 'Self-Employed',
            'employment_stability_years': 3, 'credit_score': 640, 'existing_loans_count': 3,
            'existing_monthly_debt': 22000, 'requested_loan_amount': 400000, 'loan_tenure_months': 36,
            'dependents': 3, 'savings_balance': 120000
        },
        'High Risk Credit Profile': {
            'age': 28, 'annual_income': 300000, 'employment_type': 'Freelance',
            'employment_stability_years': 1, 'credit_score': 520, 'existing_loans_count': 4,
            'existing_monthly_debt': 18000, 'requested_loan_amount': 800000, 'loan_tenure_months': 24,
            'dependents': 2, 'savings_balance': 25000
        }
    }

    profile = st.selectbox('Demo Profile Selector', list(demo_profiles.keys()))
    defaults = demo_profiles[profile]

    with st.form("credit_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=defaults.get('age', 30), help="Applicant's age")
            annual_income = st.number_input("Annual Income", min_value=1, value=defaults.get('annual_income', 500000))
            emp_type_idx = EMPLOYMENT_TYPES.index(defaults['employment_type']) if defaults.get('employment_type') in EMPLOYMENT_TYPES else 0
            employment_type = st.selectbox("Employment Type", EMPLOYMENT_TYPES, index=emp_type_idx)
            employment_stability_years = st.number_input("Years of Employment", min_value=0, max_value=50, value=defaults.get('employment_stability_years', 2))
        with col2:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=defaults.get('credit_score', 650))
            existing_loans_count = st.number_input("Existing Loans Count", min_value=0, value=defaults.get('existing_loans_count', 0))
            existing_monthly_debt = st.number_input("Monthly Debt Payments", min_value=0, value=defaults.get('existing_monthly_debt', 5000))
            dependents = st.number_input("Dependents", min_value=0, max_value=15, value=defaults.get('dependents', 0))
        with col3:
            requested_loan_amount = st.number_input("Requested Loan Amount", min_value=1000, value=defaults.get('requested_loan_amount', 100000))
            loan_tenure_months = st.number_input("Loan Tenure (Months)", min_value=6, max_value=360, value=defaults.get('loan_tenure_months', 24))
            savings_balance = st.number_input("Savings Balance", min_value=0, value=defaults.get('savings_balance', 50000))

        submitted = st.form_submit_button("Assess Credit")

    if submitted:
        input_data = {
            'age': int(age),
            'annual_income': float(annual_income),
            'employment_type': employment_type,
            'employment_stability_years': float(employment_stability_years),
            'credit_score': int(credit_score),
            'existing_loans_count': int(existing_loans_count),
            'existing_monthly_debt': float(existing_monthly_debt),
            'requested_loan_amount': float(requested_loan_amount),
            'loan_tenure_months': int(loan_tenure_months),
            'dependents': int(dependents),
            'savings_balance': float(savings_balance)
        }

        with st.spinner("Analyzing profile with ML models..."):
            try:
                res = credit_service.assess_credit(input_data)
                if not res.get('success'):
                    st.error(f"Error analyzing credit: {res.get('errors')}")
                else:
                    st.subheader("Assessment Results")
                    steps = [
                        {'name': 'Data Collection', 'status': 'done', 'detail': 'Received form input'},
                        {'name': 'Risk Modeling', 'status': 'done', 'detail': 'Evaluated by Gradient Boosting ML'},
                        {'name': 'Decision Engine', 'status': 'done', 'detail': 'Underwriting rule check complete'}
                    ]
                    workflow_stepper(steps)
                    
                    decision = res.get('decision', 'REVIEW')
                    confidence = res.get('confidence', 0.0) * 100
                    decision_banner(decision, confidence)

                    c1, c2 = st.columns(2)
                    with c1:
                        gauge_chart(res.get('risk_score', 0), title="Default Risk Score")
                    with c2:
                        probs = res.get('probabilities', {})
                        if probs:
                            prob_df = pd.DataFrame(list(probs.items()), columns=['Risk Level', 'Probability'])
                            fig = px.bar(prob_df, x='Risk Level', y='Probability', color='Risk Level', 
                                         color_discrete_map={'LOW':'#28a745', 'MEDIUM':'#ffc107', 'HIGH':'#dc3545'},
                                         title="Risk Probability Distribution")
                            st.plotly_chart(fig, width="stretch")

                    explanation_obj = res.get('explanation', {})
                    if isinstance(explanation_obj, dict):
                        factors = explanation_obj.get('top_factors', [])
                        plain_text = explanation_obj.get('plain_english', '')
                    else:
                        factors = []
                        plain_text = str(explanation_obj)

                    if factors:
                        factor_chart(factors, title="Key Contributing Factors (XAI)")

                    if decision == 'APPROVE':
                        st.success(f"**Explanation:** {plain_text}")
                    elif decision == 'REJECT':
                        st.error(f"**Explanation:** {plain_text}")
                    else:
                        st.warning(f"**Explanation:** {plain_text}")

                    st.info(f"**Recommendation:** {res.get('recommendation', 'N/A')}")
            except Exception as e:
                st.error(f"Exception during assessment: {str(e)}")

    st.divider()
    st.subheader("Recent Assessments History")
    recent = repository.get_recent_loan_applications(10)
    if recent:
        st.dataframe(pd.DataFrame(recent), width="stretch")
    else:
        st.write("No recent history.")
