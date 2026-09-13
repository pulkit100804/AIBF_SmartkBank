import streamlit as st
import pandas as pd
from app.services import fraud_service
from app.ui.components import gauge_chart, factor_chart, risk_badge, workflow_stepper
from app.config.settings import TRANSACTION_TYPES
from app.data import repository

def render_fraud_page():
    """Fraud Detection page."""
    st.title("Fraud Detection System")
    st.write("Real-time transaction monitoring and anomaly detection.")

    demo_scenarios = {
        'Custom Input': {},
        'Normal Transaction': {
            'amount': 2500, 'transaction_type': 'Payment', 'hour_of_day': 14, 'day_of_week': 2,
            'location_change': False, 'new_device': False, 'transaction_frequency_24h': 2,
            'customer_avg_amount': 3000, 'distance_from_home_km': 5, 'account_age_days': 730
        },
        'Suspicious Transaction': {
            'amount': 85000, 'transaction_type': 'Transfer', 'hour_of_day': 3, 'day_of_week': 6,
            'location_change': True, 'new_device': True, 'transaction_frequency_24h': 14,
            'customer_avg_amount': 5000, 'distance_from_home_km': 450, 'account_age_days': 45
        }
    }

    scenario = st.selectbox('Demo Scenario Selector', list(demo_scenarios.keys()))
    defaults = demo_scenarios[scenario]

    with st.form("fraud_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            amount = st.number_input("Amount", min_value=1, value=defaults.get('amount', 1000))
            txn_type_idx = TRANSACTION_TYPES.index(defaults['transaction_type']) if defaults.get('transaction_type') in TRANSACTION_TYPES else 0
            transaction_type = st.selectbox("Transaction Type", TRANSACTION_TYPES, index=txn_type_idx)
            hour_of_day = st.number_input("Hour of Day (0-23)", min_value=0, max_value=23, value=defaults.get('hour_of_day', 12))
            day_of_week = st.number_input("Day of Week (0-6)", min_value=0, max_value=6, value=defaults.get('day_of_week', 0))
        with col2:
            location_change = st.checkbox("Location Change", value=defaults.get('location_change', False))
            new_device = st.checkbox("New Device", value=defaults.get('new_device', False))
            transaction_frequency_24h = st.number_input("Transactions (Last 24h)", min_value=0, value=defaults.get('transaction_frequency_24h', 1))
        with col3:
            customer_avg_amount = st.number_input("Average Txn Amount", min_value=0, value=defaults.get('customer_avg_amount', 1000))
            distance_from_home_km = st.number_input("Distance from Home (km)", min_value=0, value=defaults.get('distance_from_home_km', 0))
            account_age_days = st.number_input("Account Age (days)", min_value=0, value=defaults.get('account_age_days', 30))

        submitted = st.form_submit_button("Assess Transaction")

    if submitted:
        input_data = {
            'amount': float(amount),
            'transaction_type': transaction_type,
            'hour_of_day': int(hour_of_day),
            'day_of_week': int(day_of_week),
            'location_change': bool(location_change),
            'new_device': bool(new_device),
            'transaction_frequency_24h': int(transaction_frequency_24h),
            'customer_avg_amount': float(customer_avg_amount),
            'distance_from_home_km': float(distance_from_home_km),
            'account_age_days': int(account_age_days)
        }

        with st.spinner("Analyzing transaction with Fraud ML model..."):
            try:
                res = fraud_service.assess_transaction(input_data)
                if not res.get('success'):
                    st.error(f"Error checking fraud: {res.get('errors')}")
                else:
                    st.subheader("Analysis Results")
                    steps = [
                        {'name': 'Event Capture', 'status': 'done', 'detail': 'Transaction payload parsed'},
                        {'name': 'Feature Engineering', 'status': 'done', 'detail': 'Context & risk composite derived'},
                        {'name': 'Random Forest Scoring', 'status': 'done', 'detail': 'Scored by 150 Decision Trees'}
                    ]
                    workflow_stepper(steps)

                    c1, c2 = st.columns(2)
                    with c1:
                        gauge_chart(res.get('fraud_probability', 0.0), title="Fraud Probability")
                    with c2:
                        st.markdown("### Risk Assessment")
                        risk_badge(res.get('risk_level', 'LOW'))
                        st.markdown(f"**Action Required:** `{res.get('action', 'None')}`")
                    
                    explanation_obj = res.get('explanation', {})
                    if isinstance(explanation_obj, dict):
                        factors = explanation_obj.get('top_factors', [])
                        plain_text = explanation_obj.get('plain_english', '')
                    else:
                        factors = []
                        plain_text = str(explanation_obj)

                    if factors:
                        factor_chart(factors, title="Anomaly Indicators (XAI)")
                    
                    st.info(f"**Summary:** {plain_text}")
            except Exception as e:
                st.error(f"Exception during assessment: {str(e)}")

    st.divider()
    st.subheader("Recent Fraud Alerts History")
    alerts = repository.get_recent_fraud_alerts(10)
    if alerts:
        st.dataframe(pd.DataFrame(alerts), width="stretch")
    else:
        st.write("No recent alerts.")
