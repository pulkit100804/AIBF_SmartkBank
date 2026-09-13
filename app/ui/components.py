import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

def metric_card(label: str, value: str, delta: str = None, color: str = '#1f77b4'):
    """Display a styled metric card using st.metric or custom HTML."""
    st.metric(label=label, value=value, delta=delta)

def risk_badge(level: str):
    """Display colored risk badge (LOW=green, MEDIUM=amber, HIGH=red).
    Use st.markdown with HTML/CSS styling."""
    colors = {'LOW': '#28a745', 'MEDIUM': '#ffc107', 'HIGH': '#dc3545'}
    color = colors.get(level.upper(), '#6c757d')
    st.markdown(f"""
    <div style="background-color: {color}; padding: 5px 10px; border-radius: 15px; color: white; text-align: center; font-weight: bold; width: fit-content; margin: auto;">
        {level.upper()} RISK
    </div>
    """, unsafe_allow_html=True)

def decision_banner(decision: str, confidence: float):
    """Large colored banner showing APPROVE/REVIEW/REJECT with confidence %."""
    colors = {'APPROVE': '#d4edda', 'REVIEW': '#fff3cd', 'REJECT': '#f8d7da'}
    text_colors = {'APPROVE': '#155724', 'REVIEW': '#856404', 'REJECT': '#721c24'}
    bg_color = colors.get(decision.upper(), '#e2e3e5')
    text_color = text_colors.get(decision.upper(), '#383d41')
    
    st.markdown(f"""
    <div style="background-color: {bg_color}; color: {text_color}; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
        <h2 style="margin:0;">Decision: {decision.upper()}</h2>
        <p style="margin:0; font-size: 1.1em;">Confidence: {confidence:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

def factor_chart(factors: list, title: str = 'Top Contributing Factors'):
    """Horizontal bar chart of feature importance factors.
    factors is a list of dicts with keys: label, importance, direction.
    Color bars green for decreases_risk, red for increases_risk.
    Use plotly horizontal bar chart."""
    if not factors:
        st.write("No factors available.")
        return
    
    labels = [f.get('label', '') for f in factors]
    importances = [f.get('importance', 0) for f in factors]
    directions = [f.get('direction', 'increases_risk') for f in factors]
    
    colors = ['#dc3545' if d == 'increases_risk' else '#28a745' for d in directions]
    
    fig = go.Figure(go.Bar(
        x=importances,
        y=labels,
        orientation='h',
        marker_color=colors
    ))
    fig.update_layout(
        title=title,
        yaxis={'autorange': 'reversed'},
        margin=dict(l=20, r=20, t=40, b=20),
        height=300
    )
    st.plotly_chart(fig, width="stretch")

def gauge_chart(value: float, title: str = 'Risk Score', max_val: float = 1.0):
    """Plotly gauge chart for risk scores. Green-yellow-red color scheme."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [0, max_val]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, max_val * 0.33], 'color': "#28a745"},
                {'range': [max_val * 0.33, max_val * 0.66], 'color': "#ffc107"},
                {'range': [max_val * 0.66, max_val], 'color': "#dc3545"}
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, width="stretch")

def workflow_stepper(steps: list[dict]):
    """Display the automation workflow as a visual stepper.
    Each step dict has: name, status ('done', 'active', 'pending'), detail.
    Use markdown with emojis: ✅ done, 🔄 active, ⏳ pending."""
    icons = {'done': '✅', 'active': '🔄', 'pending': '⏳'}
    
    cols = st.columns(len(steps))
    for i, step in enumerate(steps):
        with cols[i]:
            status = step.get('status', 'pending')
            icon = icons.get(status, '⏳')
            st.markdown(f"**{icon} {step.get('name', '')}**")
            st.caption(step.get('detail', ''))
