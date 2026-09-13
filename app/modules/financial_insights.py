"""
Transaction analysis and personalized insights.
"""

import pandas as pd
import numpy as np

def analyze_transactions(df: pd.DataFrame) -> dict:
    """Comprehensive transaction analysis."""
    if df.empty:
        return {
            "total_income": 0.0,
            "total_expenses": 0.0,
            "net_savings": 0.0,
            "savings_rate": 0.0,
            "spending_by_category": {},
            "top_spending_categories": [],
            "monthly_summary": [],
            "anomalies": [],
            "recommendations": []
        }
        
    df = df.copy()
    if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    income_mask = df['amount'] > 0
    expense_mask = df['amount'] < 0
    
    total_income = df.loc[income_mask, 'amount'].sum()
    total_expenses = abs(df.loc[expense_mask, 'amount'].sum())
    net_savings = total_income - total_expenses
    savings_rate = (net_savings / total_income) if total_income > 0 else 0.0
    
    expenses_df = df[expense_mask].copy()
    expenses_df['abs_amount'] = expenses_df['amount'].abs()
    
    spending_by_category = expenses_df.groupby('category')['abs_amount'].sum().to_dict()
    top_spending_categories = sorted(spending_by_category.items(), key=lambda x: x[1], reverse=True)[:5]
    
    df['month'] = df['date'].dt.to_period('M')
    monthly_groups = df.groupby('month')
    monthly_summary = []
    for m, group in monthly_groups:
        m_inc = group.loc[group['amount'] > 0, 'amount'].sum()
        m_exp = abs(group.loc[group['amount'] < 0, 'amount'].sum())
        monthly_summary.append({
            "month": str(m),
            "income": float(m_inc),
            "expenses": float(m_exp),
            "savings": float(m_inc - m_exp)
        })
        
    anomalies = detect_spending_anomalies(df)
    
    analysis_result = {
        "total_income": float(total_income),
        "total_expenses": float(total_expenses),
        "net_savings": float(net_savings),
        "savings_rate": float(savings_rate),
        "spending_by_category": {k: float(v) for k, v in spending_by_category.items()},
        "top_spending_categories": [(k, float(v)) for k, v in top_spending_categories],
        "monthly_summary": monthly_summary,
        "anomalies": anomalies
    }
    
    analysis_result["recommendations"] = generate_recommendations(analysis_result)
    
    return analysis_result

def detect_spending_anomalies(df: pd.DataFrame) -> list[dict]:
    """Find transactions that are >2 standard deviations from category mean."""
    anomalies = []
    expense_df = df[df['amount'] < 0].copy()
    expense_df['abs_amount'] = expense_df['amount'].abs()
    
    for category, group in expense_df.groupby('category'):
        if len(group) > 2:
            mean_amt = group['abs_amount'].mean()
            std_amt = group['abs_amount'].std()
            if pd.isna(std_amt) or std_amt == 0:
                continue
            threshold = mean_amt + 2 * std_amt
            outliers = group[group['abs_amount'] > threshold]
            for _, row in outliers.iterrows():
                anomalies.append({
                    "date": row['date'].isoformat() if hasattr(row['date'], 'isoformat') else str(row['date']),
                    "amount": float(row['abs_amount']),
                    "category": category,
                    "reason": f"Unusually high amount for {category} (avg: {mean_amt:.2f})"
                })
    return anomalies

def generate_recommendations(analysis: dict) -> list[str]:
    """Generate personalized recommendations based on analysis results."""
    recs = []
    savings_rate = analysis.get("savings_rate", 0)
    
    if savings_rate < 0.1:
        recs.append(f"Your savings rate is below 10% ({savings_rate:.1%}). Consider reviewing your top expenses to find saving opportunities.")
    elif savings_rate > 0.3:
        recs.append("Great job! Your savings rate is excellent. Consider investing excess cash for higher returns.")
        
    total_exp = analysis.get("total_expenses", 0)
    top_cats = analysis.get("top_spending_categories", [])
    if total_exp > 0 and top_cats:
        top_cat, top_amt = top_cats[0]
        if top_amt / total_exp > 0.3:
            recs.append(f"'{top_cat}' accounts for {(top_amt/total_exp):.1%} of your total spending. See if you can reduce costs here.")
            
    anomalies = analysis.get("anomalies", [])
    if len(anomalies) > 0:
        recs.append(f"{len(anomalies)} unusual transactions detected. Please review your anomalies list to ensure no unauthorized charges.")
        
    monthly_summary = analysis.get("monthly_summary", [])
    if len(monthly_summary) >= 2:
        last_month = monthly_summary[-1]['expenses']
        prev_month = monthly_summary[-2]['expenses']
        if prev_month > 0 and (last_month - prev_month) / prev_month > 0.2:
            recs.append(f"Monthly expenses have increased by {((last_month - prev_month) / prev_month):.1%} compared to the previous month.")
            
    if not recs:
        recs.append("Your spending looks consistent and well-managed.")
        
    return recs
