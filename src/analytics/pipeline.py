"""
Real Data Analytics Pipeline Orchestrator.

Coordinates calculation and persistent export of:
1. Real Market Summary (technical indicators and price metrics)
2. Real News Summary (company-level NLP sentiment, relevance, triage)
3. Real News Daily Summary (daily aggregated news signals)
4. Real Data Coverage Audit (completeness, source counts, quota boundaries)

Writes machine-readable CSV outputs to `data/live/analytics/` with ISO 8601 UTC timestamps.
Does NOT modify academic datasets or interact with Decision Fusion / ML models.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from src.analytics.coverage_analytics import compute_real_data_coverage
from src.analytics.market_analytics import compute_real_market_summary
from src.analytics.news_analytics import compute_real_news_daily_summary, compute_real_news_summary

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LIVE_ANALYTICS_DIR = PROJECT_ROOT / "data" / "live" / "analytics"


def run_real_data_analytics(
    output_dir: Optional[Union[str, Path]] = None,
    save_outputs: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Execute the end-to-end real data analytics calculations and save results.
    
    Args:
        output_dir: Target directory for CSV outputs (default: data/live/analytics/)
        save_outputs: If True, writes CSVs to output_dir
        
    Returns:
        Dictionary of computed DataFrames:
        {
            'market_summary': pd.DataFrame,
            'news_summary': pd.DataFrame,
            'news_daily_summary': pd.DataFrame,
            'data_coverage': pd.DataFrame
        }
    """
    target_dir = Path(output_dir) if output_dir else LIVE_ANALYTICS_DIR
    logger.info("Executing Real Data Analytics Pipeline...")
    
    # 1. Market Analytics
    market_summary_df = compute_real_market_summary()
    logger.info(f"Computed market analytics for {len(market_summary_df)} tickers.")
    
    # 2. News Analytics
    news_summary_df = compute_real_news_summary()
    logger.info(f"Computed news summary for {len(news_summary_df)} tickers.")
    
    # 3. Daily News Signal Analytics
    news_daily_df = compute_real_news_daily_summary()
    logger.info(f"Computed daily news signals: {len(news_daily_df)} date-ticker sessions.")
    
    # 4. Coverage Audit
    coverage_df = compute_real_data_coverage()
    logger.info(f"Computed coverage audit for {len(coverage_df)} tickers.")
    
    results = {
        "market_summary": market_summary_df,
        "news_summary": news_summary_df,
        "news_daily_summary": news_daily_df,
        "data_coverage": coverage_df
    }
    
    if save_outputs:
        target_dir.mkdir(parents=True, exist_ok=True)
        
        file_map = {
            "real_market_summary.csv": market_summary_df,
            "real_news_summary.csv": news_summary_df,
            "real_news_daily_summary.csv": news_daily_df,
            "real_data_coverage.csv": coverage_df
        }
        
        for fname, df in file_map.items():
            if not df.empty:
                out_path = target_dir / fname
                df.to_csv(out_path, index=False)
                logger.info(f"Saved {fname} ({len(df)} rows) to {out_path}")
                
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_real_data_analytics()
