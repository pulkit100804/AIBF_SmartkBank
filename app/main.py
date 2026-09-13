import streamlit as st
from app.database.db import init_db

# Page config MUST be first Streamlit command
st.set_page_config(
    page_title='SmartBank AI',
    page_icon='🏦',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Initialize database
init_db()

# Sidebar navigation
st.sidebar.title('🏦 SmartBank AI')
st.sidebar.caption('AI-Powered Banking Assistant')

page = st.sidebar.radio(
    'Navigation',
    ['Dashboard', 'Credit Decision', 'Fraud Detection', 'Financial Insights', 'Model & Explainability'],
    index=0
)

# Footer in sidebar
st.sidebar.divider()
st.sidebar.caption('v1.0.0 | Educational Prototype')
st.sidebar.caption('Built with Streamlit + scikit-learn')

# Route to pages
if page == 'Dashboard':
    from app.ui.dashboard import render_dashboard
    render_dashboard()
elif page == 'Credit Decision':
    from app.ui.credit_page import render_credit_page
    render_credit_page()
elif page == 'Fraud Detection':
    from app.ui.fraud_page import render_fraud_page
    render_fraud_page()
elif page == 'Financial Insights':
    from app.ui.insights_page import render_insights_page
    render_insights_page()
elif page == 'Model & Explainability':
    from app.ui.explainability_page import render_explainability_page
    render_explainability_page()
