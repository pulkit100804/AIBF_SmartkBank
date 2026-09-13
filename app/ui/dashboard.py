import streamlit as st
import pandas as pd
from app.data import repository
from app.ui.components import metric_card

def render_dashboard():
    """Main dashboard page."""
    st.title('SmartBank AI Dashboard')
    
    try:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("Total Assessments", str(repository.get_loan_application_count()))
        with col2:
            metric_card("Fraud Alerts", str(repository.get_fraud_alert_count()))
        with col3:
            metric_card("High Risk Alerts", str(repository.get_high_risk_fraud_count()))
        with col4:
            metric_card("Insights Generated", str(repository.get_insight_count()))
            
        st.divider()
        
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Recent Loan Assessments")
            recent_loans = repository.get_recent_loan_applications(5)
            if recent_loans:
                st.dataframe(pd.DataFrame(recent_loans), width="stretch")
            else:
                st.info("No recent loan assessments.")
                
        with col_right:
            st.subheader("Recent Fraud Alerts")
            recent_frauds = repository.get_recent_fraud_alerts(5)
            if recent_frauds:
                st.dataframe(pd.DataFrame(recent_frauds), width="stretch")
            else:
                st.info("No recent fraud alerts.")
                
        st.divider()
        st.subheader("System Overview")
        st.write("SmartBank AI operates across three main modules to ensure financial security and operational efficiency:")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("### 🏦 Credit Decisioning")
            st.write("Leveraging machine learning to assess creditworthiness based on real-time factors.")
        with col_b:
            st.markdown("### 🛡️ Fraud Detection")
            st.write("Identifying suspicious activities through continuous transaction monitoring.")
        with col_c:
            st.markdown("### 📊 Financial Insights")
            st.write("Providing personalized financial management and savings recommendations.")
    except Exception as e:
        st.error(f"Error loading dashboard: {str(e)}")
