import streamlit as st
import pandas as pd
import plotly.express as px
from app.models import model_loader
from app.config.settings import CREDIT_FEATURE_LABELS, FRAUD_FEATURE_LABELS

def render_explainability_page():
    """Model & Explainability page."""
    st.title('Model Information & Explainable AI')
    st.write("Understand the machine learning models powering SmartBank AI.")
    
    tab1, tab2 = st.tabs(['Credit Model', 'Fraud Model'])
    
    with tab1:
        try:
            credit_model = model_loader.load_credit_model()
            st.subheader("Credit Decision Model Overview")
            st.write("**Algorithm:** Gradient Boosting Classifier")
            st.write("**Training Data Size:** 50,000 records")
            st.write(f"**Number of Features:** {len(credit_model.feature_names) if hasattr(credit_model, 'feature_names') else 'Unknown'}")
            
            if hasattr(credit_model, 'evaluation_metrics'):
                st.markdown("### Evaluation Metrics")
                metrics = credit_model.evaluation_metrics
                st.dataframe(pd.DataFrame([metrics]), width="stretch")
            
            if hasattr(credit_model, 'get_feature_importances'):
                st.markdown("### Feature Importance")
                fi = credit_model.get_feature_importances()
                if fi:
                    fi_df = pd.DataFrame(list(fi.items()), columns=['Feature', 'Importance'])
                    fi_df['Feature'] = fi_df['Feature'].apply(lambda x: CREDIT_FEATURE_LABELS.get(x, x))
                    fi_df = fi_df.sort_values(by='Importance', ascending=True)
                    fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h', title='Credit Model Feature Importance')
                    st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.error(f"Could not load Credit Model info: {str(e)}")
            
    with tab2:
        try:
            fraud_model = model_loader.load_fraud_model()
            st.subheader("Fraud Detection Model Overview")
            st.write("**Algorithm:** Random Forest Classifier")
            st.write("**Training Data Size:** 100,000 records")
            st.write(f"**Number of Features:** {len(fraud_model.feature_names) if hasattr(fraud_model, 'feature_names') else 'Unknown'}")
            
            if hasattr(fraud_model, 'evaluation_metrics'):
                st.markdown("### Evaluation Metrics")
                metrics = fraud_model.evaluation_metrics
                st.dataframe(pd.DataFrame([metrics]), width="stretch")
                
            if hasattr(fraud_model, 'get_feature_importances'):
                st.markdown("### Feature Importance")
                fi = fraud_model.get_feature_importances()
                if fi:
                    fi_df = pd.DataFrame(list(fi.items()), columns=['Feature', 'Importance'])
                    fi_df['Feature'] = fi_df['Feature'].apply(lambda x: FRAUD_FEATURE_LABELS.get(x, x))
                    fi_df = fi_df.sort_values(by='Importance', ascending=True)
                    fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h', title='Fraud Model Feature Importance')
                    st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.error(f"Could not load Fraud Model info: {str(e)}")

    st.divider()
    st.subheader("Methodology & XAI Approach")
    st.write("We employ SHAP (SHapley Additive exPlanations) values to provide local interpretability for each prediction. This allows us to explain exactly which factors contributed most to a specific decision, ensuring transparency and trust.")
    
    st.subheader("Responsible AI")
    with st.expander("Model Bias Risk"):
        st.write("Our models are regularly audited for demographic bias. We do not use protected attributes such as race, gender, or religion as features.")
    with st.expander("Explainability approach"):
        st.write("We use local explainability (SHAP) and global explainability (Feature Importance) to ensure human stakeholders understand model logic.")
    with st.expander("Privacy considerations"):
        st.write("All training data is anonymized and PII is removed. Inference happens securely with immediate data scrubbing options.")
    with st.expander("Human oversight requirement"):
        st.write("Critical decisions, particularly rejections or high-risk fraud alerts, require human-in-the-loop validation.")
    with st.expander("Synthetic data limitations"):
        st.write("For this prototype, synthetic data was used. Real-world deployment requires production data tuning and re-evaluation.")
    with st.expander("Not a replacement for regulated banking"):
        st.write("This tool is designed to assist, not replace, formal banking systems and certified financial analysts.")
    with st.expander("Educational prototype disclaimer"):
        st.write("This application is an educational prototype. Do not use for real financial decisions.")
