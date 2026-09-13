import streamlit as st
import pandas as pd
import plotly.express as px
from app.services import insights_service
from app.ui.components import metric_card
from app.utils import validation

def render_insights_page():
    """Financial Insights page."""
    st.title("Financial Insights Engine")
    st.write("Personalized analytics and recommendations based on transaction history.")

    sources = ['Demo - Conservative Saver', 'Demo - Heavy Spender', 'Demo - Unusual Activity', 'Upload CSV']
    source = st.selectbox("Data Source Profile", sources)
    
    df = None
    if source == 'Upload CSV':
        uploaded = st.file_uploader("Upload Transaction CSV", type=['csv'])
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                val_res = validation.validate_transaction_csv(df)
                if not val_res.is_valid:
                    st.error(f"Validation errors: {', '.join(val_res.errors)}")
                    df = None
            except Exception as e:
                st.error(f"Failed to read CSV: {str(e)}")
                df = None
    
    if st.button("Analyze Finances"):
        with st.spinner("Analyzing spending trends and detecting anomalies..."):
            try:
                customer_profile = source.replace('Demo - ', '') if source != 'Upload CSV' else 'Custom'
                res = insights_service.analyze_customer_finances(df=df, customer_profile=customer_profile)
                if not res.get('success'):
                    st.error(f"Error analyzing finances: {res.get('errors')}")
                else:
                    analysis = res.get('analysis_result', {})
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        metric_card("Total Income", f"₹{analysis.get('total_income', 0):,.2f}")
                    with c2:
                        metric_card("Total Expenses", f"₹{analysis.get('total_expenses', 0):,.2f}")
                    with c3:
                        metric_card("Net Savings", f"₹{analysis.get('net_savings', 0):,.2f}")
                    with c4:
                        metric_card("Savings Rate", f"{(analysis.get('savings_rate', 0) * 100):.1f}%")
                    
                    st.divider()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        cat_data = analysis.get('spending_by_category', {})
                        if cat_data:
                            cat_df = pd.DataFrame(list(cat_data.items()), columns=['Category', 'Amount'])
                            fig1 = px.pie(cat_df, values='Amount', names='Category', title='Spending by Category')
                            st.plotly_chart(fig1, width="stretch")
                    
                    with col2:
                        monthly_list = analysis.get('monthly_summary', [])
                        if monthly_list:
                            rows = []
                            for m_dict in monthly_list:
                                month_str = m_dict.get('month', 'N/A')
                                rows.append({'Month': month_str, 'Type': 'Income', 'Amount': m_dict.get('income', 0)})
                                rows.append({'Month': month_str, 'Type': 'Expenses', 'Amount': m_dict.get('expenses', 0)})
                            monthly_df = pd.DataFrame(rows)
                            if not monthly_df.empty:
                                fig2 = px.line(monthly_df, x='Month', y='Amount', color='Type', title='Monthly Income vs Expenses')
                                st.plotly_chart(fig2, width="stretch")
                                
                    anomalies = analysis.get('anomalies', [])
                    if anomalies:
                        st.warning("⚠️ Anomalous Spending Detected (>2 Std Devs from Mean)")
                        st.dataframe(pd.DataFrame(anomalies), width="stretch")
                    else:
                        st.success("✅ No unusual spending anomalies detected in the period.")
                        
                    recs = analysis.get('recommendations', [])
                    if recs:
                        st.subheader("Personalized Financial Advice")
                        for i, rec in enumerate(recs, 1):
                            st.info(f"💡 **Tip {i}:** {rec}")
                            
            except Exception as e:
                st.error(f"Exception during analysis: {str(e)}")
