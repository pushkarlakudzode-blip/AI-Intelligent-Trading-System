"""
Real News Analytics Module.

Aggregates and computes company-level and daily financial news sentiment analytics
from the real NLP enrichment pipeline outputs in SQLite and CSV storage.

Metrics calculated:
- Article counts: Total, Positive, Neutral, Negative
- Loughran-McDonald sentiment scores: Mean sentiment, relevance, importance, triage
- High-impact news article count
- Dominant sentiment classification
- Daily aggregated news signals

Note: Uses our project's internal NLP analysis, NOT Alpha Vantage's provider scores.
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
CSV_NLP_PATH = LIVE_DATA_DIR / "nlp" / "news_nlp_results.csv"
CSV_DAILY_SIGNAL_PATH = LIVE_DATA_DIR / "nlp" / "daily_news_signal.csv"

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


def load_real_nlp_results(db_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load real NLP records from SQLite storage, falling back to CSV if needed."""
    target_db = Path(db_path) if db_path else DB_PATH
    if target_db.exists():
        try:
            conn = sqlite3.connect(f"file:{target_db}?mode=ro", uri=True)
            # Only select genuine records for supported tickers
            query = """
            SELECT n.*, a.source, a.headline, a.url
            FROM nlp_results n
            LEFT JOIN articles a ON n.news_id = a.news_id
            WHERE n.dataset_type = 'REAL'
            """
            df = pd.read_sql(query, conn)
            conn.close()
            # Standardize column casing
            col_map = {
                "news_id": "News_ID", "ticker": "Ticker", "published_at_utc": "Published_At_UTC",
                "sentiment_label": "Sentiment_Label", "sentiment_score": "Sentiment_Score",
                "relevance_score": "Relevance_Score", "event_type": "Event_Type",
                "importance_score": "Importance_Score", "recency_score": "Recency_Score",
                "triage_score": "Triage_Score", "processed_at_utc": "Processed_At_UTC",
                "dataset_type": "Dataset_Type", "source": "Source", "headline": "Headline", "url": "URL"
            }
            df = df.rename(columns=col_map)
            return df
        except Exception as e:
            logger.warning(f"Failed to read from SQLite ({e}), falling back to CSV.")
            
    if CSV_NLP_PATH.exists():
        return pd.read_csv(CSV_NLP_PATH)
        
    return pd.DataFrame()


def load_real_daily_signals(csv_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load real daily news signals from CSV storage."""
    target_csv = Path(csv_path) if csv_path else CSV_DAILY_SIGNAL_PATH
    if target_csv.exists():
        return pd.read_csv(target_csv)
    return pd.DataFrame()


def compute_real_news_summary(nlp_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compute aggregate sentiment and impact analytics for each supported company.
    
    Returns:
        DataFrame with columns:
        ['Ticker', 'Company_Name', 'Article_Count', 'Positive_Count', 'Neutral_Count',
         'Negative_Count', 'Average_Sentiment', 'Average_Relevance', 'Average_Importance',
         'Average_Triage_Score', 'High_Impact_Article_Count', 'Dominant_Sentiment',
         'Latest_News_Signal', 'Generated_At_UTC']
    """
    if nlp_df is None or nlp_df.empty:
        nlp_df = load_real_nlp_results()
        
    daily_signals_df = load_real_daily_signals()
    now_utc = datetime.now(timezone.utc).isoformat()
    
    summary_rows = []
    
    for ticker in SUPPORTED_TICKERS:
        t_nlp = nlp_df[nlp_df["Ticker"] == ticker] if not nlp_df.empty else pd.DataFrame()
        company_name = COMPANY_NAMES.get(ticker, ticker)
        
        art_count = len(t_nlp)
        if art_count > 0:
            pos_cnt = int((t_nlp["Sentiment_Label"] == "Positive").sum())
            neu_cnt = int((t_nlp["Sentiment_Label"] == "Neutral").sum())
            neg_cnt = int((t_nlp["Sentiment_Label"] == "Negative").sum())
            
            avg_sent = float(t_nlp["Sentiment_Score"].mean())
            avg_rel = float(t_nlp["Relevance_Score"].mean())
            avg_imp = float(t_nlp["Importance_Score"].mean())
            avg_triage = float(t_nlp["Triage_Score"].mean())
            
            # High impact articles (Triage score >= 0.50 or Importance score >= 0.70)
            high_impact_cnt = int(((t_nlp["Triage_Score"] >= 0.50) | (t_nlp["Importance_Score"] >= 0.70)).sum())
            
            # Determine dominant sentiment
            if pos_cnt >= neu_cnt and pos_cnt >= neg_cnt:
                dom_sent = "Positive"
            elif neg_cnt >= neu_cnt and neg_cnt >= pos_cnt:
                dom_sent = "Negative"
            else:
                dom_sent = "Neutral"
        else:
            pos_cnt = neu_cnt = neg_cnt = high_impact_cnt = 0
            avg_sent = avg_rel = avg_imp = avg_triage = 0.0
            dom_sent = "Neutral"
            
        # Latest news signal from daily signals
        latest_signal = 0.0
        if not daily_signals_df.empty and "Ticker" in daily_signals_df.columns:
            t_signals = daily_signals_df[daily_signals_df["Ticker"] == ticker].sort_values(by="Date")
            if not t_signals.empty and "News_Signal" in t_signals.columns:
                latest_signal = float(t_signals["News_Signal"].iloc[-1])
                
        summary_rows.append({
            "Ticker": ticker,
            "Company_Name": company_name,
            "Article_Count": art_count,
            "Positive_Count": pos_cnt,
            "Neutral_Count": neu_cnt,
            "Negative_Count": neg_cnt,
            "Average_Sentiment": round(avg_sent, 4),
            "Average_Relevance": round(avg_rel, 4),
            "Average_Importance": round(avg_imp, 4),
            "Average_Triage_Score": round(avg_triage, 4),
            "High_Impact_Article_Count": high_impact_cnt,
            "Dominant_Sentiment": dom_sent,
            "Latest_News_Signal": round(latest_signal, 4),
            "Generated_At_UTC": now_utc
        })
        
    return pd.DataFrame(summary_rows)


def compute_real_news_daily_summary(daily_signals_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Format and validate daily aggregated news signals across all tickers and trading sessions.
    """
    if daily_signals_df is None or daily_signals_df.empty:
        daily_signals_df = load_real_daily_signals()
        
    if daily_signals_df.empty:
        return pd.DataFrame()
        
    now_utc = datetime.now(timezone.utc).isoformat()
    df = daily_signals_df.copy()
    
    # Ensure supported tickers only
    if "Ticker" in df.columns:
        df = df[df["Ticker"].isin(SUPPORTED_TICKERS)].copy()
        
    # Sort and deduplicate
    df = df.drop_duplicates(subset=["Ticker", "Date"]).sort_values(by=["Ticker", "Date"]).reset_index(drop=True)
    df["Generated_At_UTC"] = now_utc
    
    return df
