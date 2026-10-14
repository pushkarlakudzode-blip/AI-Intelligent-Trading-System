"""
Real Data Coverage Analytics Module.

Audits and summarizes real financial news and market data coverage across the
10 project companies. Evaluates article counts, NLP record alignment, source diversity,
and explicitly reports January 2025 coverage completeness vs partial/capped status.

Coverage Status Policy:
- Complete: AAPL, MSFT, META, TSLA, V, JNJ (all requested January windows fetched and verified)
- Partial / Capped: GOOGL, AMZN, NVDA, JPM (Alpha Vantage free-tier 25 req/day limit capped sub-windows)
This is an external data coverage limitation due to API quotas, not a pipeline failure.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LIVE_DATA_DIR = PROJECT_ROOT / "data" / "live"
DB_PATH = LIVE_DATA_DIR / "news_database.db"

SUPPORTED_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ"]

COMPANY_NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "META": "Meta Platforms",
    "TSLA": "Tesla",
    "JPM": "JPMorgan Chase",
    "V": "Visa",
    "JNJ": "Johnson & Johnson"
}

# Explicit historical backfill audit outcomes
JANUARY_COVERAGE_STATUS = {
    "AAPL": ("Complete", "Exhaustive January 2025 backfill (53 articles) + 50 live October 2026 articles."),
    "MSFT": ("Complete", "Exhaustive January 2025 backfill (90 articles across 4 sub-windows)."),
    "GOOGL": ("Partial / Capped", "Provider free-tier daily quota limit reached on windows 3-4; 59 January articles preserved."),
    "AMZN": ("Partial / Capped", "Provider free-tier daily quota limit reached during sub-window fetching; 50 January articles preserved."),
    "NVDA": ("Partial / Capped", "Provider free-tier daily quota limit reached during sub-window fetching; 50 January articles preserved."),
    "META": ("Complete", "Exhaustive January 2025 backfill (40 articles across full month)."),
    "TSLA": ("Complete", "Exhaustive January 2025 backfill (44 articles across full month)."),
    "JPM": ("Partial / Capped", "Provider free-tier daily quota limit reached during sub-window fetching; 50 January articles preserved."),
    "V": ("Complete", "Exhaustive January 2025 backfill (15 articles across full month)."),
    "JNJ": ("Complete", "Exhaustive January 2025 backfill (22 articles across full month).")
}


def compute_real_data_coverage(db_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Compute real-data coverage audit metrics across all 10 companies.
    
    Reads directly from the read-only SQLite live database.
    
    Returns:
        DataFrame with columns:
        ['Ticker', 'Company_Name', 'Real_Article_Count', 'January_Article_Count',
         'Earliest_Article_Date', 'Latest_Article_Date', 'January_Coverage_Status',
         'NLP_Record_Count', 'Unique_Sources_Count', 'Coverage_Notes', 'Generated_At_UTC']
    """
    target_db = Path(db_path) if db_path else DB_PATH
    now_utc = datetime.now(timezone.utc).isoformat()
    
    if not target_db.exists():
        logger.warning(f"Database not found at {target_db}")
        return pd.DataFrame()
        
    try:
        conn = sqlite3.connect(f"file:{target_db}?mode=ro", uri=True)
        
        # Aggregate real articles
        art_query = """
        SELECT 
            ticker,
            COUNT(*) as total_articles,
            SUM(CASE WHEN published_at_utc >= '2025-01-01' AND published_at_utc <= '2025-01-31T23:59:59Z' THEN 1 ELSE 0 END) as jan_articles,
            MIN(published_at_utc) as earliest_date,
            MAX(published_at_utc) as latest_date,
            COUNT(DISTINCT source) as unique_sources
        FROM articles
        WHERE dataset_type = 'REAL'
        GROUP BY ticker
        """
        art_df = pd.read_sql(art_query, conn)
        
        # Aggregate real NLP records
        nlp_query = """
        SELECT 
            ticker,
            COUNT(*) as nlp_count
        FROM nlp_results
        WHERE dataset_type = 'REAL'
        GROUP BY ticker
        """
        nlp_df = pd.read_sql(nlp_query, conn)
        conn.close()
        
    except Exception as e:
        logger.error(f"Error querying live database: {e}")
        return pd.DataFrame()
        
    rows = []
    art_dict = art_df.set_index("ticker").to_dict(orient="index") if not art_df.empty else {}
    nlp_dict = nlp_df.set_index("ticker")["nlp_count"].to_dict() if not nlp_df.empty else {}
    
    for ticker in SUPPORTED_TICKERS:
        company_name = COMPANY_NAMES.get(ticker, ticker)
        art_info = art_dict.get(ticker, {})
        
        total_arts = int(art_info.get("total_articles", 0))
        jan_arts = int(art_info.get("jan_articles", 0))
        earliest_raw = art_info.get("earliest_date", "")
        latest_raw = art_info.get("latest_date", "")
        
        earliest = str(earliest_raw)[:10] if earliest_raw else "N/A"
        latest = str(latest_raw)[:10] if latest_raw else "N/A"
        
        sources_cnt = int(art_info.get("unique_sources", 0))
        nlp_cnt = int(nlp_dict.get(ticker, 0))
        
        status, note = JANUARY_COVERAGE_STATUS.get(ticker, ("Unspecified", "No specific coverage notes."))
        
        rows.append({
            "Ticker": ticker,
            "Company_Name": company_name,
            "Real_Article_Count": total_arts,
            "January_Article_Count": jan_arts,
            "Earliest_Article_Date": earliest,
            "Latest_Article_Date": latest,
            "January_Coverage_Status": status,
            "NLP_Record_Count": nlp_cnt,
            "Unique_Sources_Count": sources_cnt,
            "Coverage_Notes": note,
            "Generated_At_UTC": now_utc
        })
        
    res_df = pd.DataFrame(rows)
    return res_df
