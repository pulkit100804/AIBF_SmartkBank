"""
Orchestrates financial insights pipeline.
"""
import pandas as pd
from app.modules.financial_insights import analyze_transactions
from app.data.repository import save_financial_insight
from app.data.generator import generate_customer_transactions
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

def analyze_customer_finances(df: pd.DataFrame = None, customer_profile: str = None) -> dict:
    """Financial insights pipeline."""
    try:
        if df is None:
            if not customer_profile:
                return {"success": False, "errors": ["Must provide df or customer_profile"]}
                
            seed = 42
            if customer_profile == 'Conservative Saver':
                seed = 42
            elif customer_profile == 'Heavy Spender':
                seed = 123
            elif customer_profile == 'Unusual Activity':
                seed = 789
                
            df = generate_customer_transactions(n_months=3, seed=seed)
            
        if df.empty:
            return {"success": False, "errors": ["Transaction data is empty"]}
            
        analysis_result = analyze_transactions(df)
        
        name = customer_profile if customer_profile else "Custom Analysis"
        period = "Last 3 Months"
        
        insight_id = save_financial_insight(
            customer_name=name,
            period=period,
            summary_data=analysis_result
        )
        
        return {
            "success": True,
            "insight_id": insight_id,
            "analysis_result": analysis_result
        }
    except Exception as e:
        logger.error(f"Error in analyze_customer_finances: {e}", exc_info=True)
        return {"success": False, "errors": [str(e)]}
