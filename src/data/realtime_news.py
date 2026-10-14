"""
Real-Time & Breaking Financial News Ingestion Module.

Fetches the latest available news snapshots from financial news providers,
normalizes publication timestamps, enforces deduplication, and streams records
into the storage and NLP processing pipeline.

API Key is read strictly from environment variable: ALPHA_VANTAGE_API_KEY.
"""

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Union
import pandas as pd

from dotenv import load_dotenv

from src.data.data_storage import (
    store_articles,
    store_nlp_results,
    get_stored_articles
)
from src.data.historical_news import (
    query_alpha_vantage_news,
    normalize_av_article,
    SUPPORTED_TICKERS,
    COMPANY_NAMES
)
from src.nlp.news_nlp_pipeline import process_articles_nlp

load_dotenv()
logger = logging.getLogger(__name__)


def fetch_latest_news_for_ticker(
    ticker: str,
    limit: int = 25,
    lookback_hours: Optional[int] = None,
    run_nlp: bool = True
) -> Dict[str, Union[str, int]]:
    """
    Fetch the latest available news articles for a single ticker over the recent window.
    
    Returns:
        Dict: Telemetry summary with keys:
              ['ticker', 'status', 'raw_fetched', 'added', 'duplicates', 'rejected']
    """
    clean_ticker = ticker.strip().upper()
    if clean_ticker not in SUPPORTED_TICKERS:
        raise ValueError(f"Ticker '{clean_ticker}' is not supported.")

    time_from_str = None
    time_to_str = None
    if lookback_hours is not None:
        now_utc = datetime.now(timezone.utc)
        from_utc = now_utc - timedelta(hours=lookback_hours)
        time_from_str = from_utc.strftime("%Y-%m-%d")
        time_to_str = now_utc.strftime("%Y-%m-%d")

    feed, status = query_alpha_vantage_news(
        ticker=clean_ticker,
        time_from=time_from_str,
        time_to=time_to_str,
        sort="LATEST",
        limit=limit
    )

    if not feed:
        return {
            "ticker": clean_ticker,
            "status": status,
            "raw_fetched": 0,
            "added": 0,
            "duplicates": 0,
            "rejected": 0
        }

    # Normalize
    normalized = []
    for item in feed:
        norm = normalize_av_article(item, clean_ticker)
        if norm:
            normalized.append(norm)

    # Persist and deduplicate
    added, dups, rejs = store_articles(normalized)

    # Execute NLP pipeline on newly added records
    if run_nlp and added > 0:
        new_df = pd.DataFrame(normalized)
        nlp_df, daily_df = process_articles_nlp(new_df)
        store_nlp_results(nlp_df, daily_df)

    return {
        "ticker": clean_ticker,
        "status": status,
        "raw_fetched": len(feed),
        "added": added,
        "duplicates": dups,
        "rejected": rejs
    }


def fetch_latest_news_all(
    tickers: Optional[List[str]] = None,
    limit_per_ticker: int = 20,
    request_delay_seconds: float = 12.5,
    run_nlp: bool = True
) -> Dict[str, Dict]:
    """
    Fetch latest news for all supported tickers while strictly respecting the
    Alpha Vantage 5-requests-per-minute rate limit (12.5 second delay between tickers).
    """
    target_tickers = tickers or SUPPORTED_TICKERS
    results = {}

    for idx, t in enumerate(target_tickers):
        logger.info(f"[{idx+1}/{len(target_tickers)}] Fetching latest news for {t}...")
        res = fetch_latest_news_for_ticker(t, limit=limit_per_ticker, run_nlp=run_nlp)
        results[t] = res

        # Rate limit protection delay between calls
        if idx < len(target_tickers) - 1:
            time_to_wait = request_delay_seconds
            logger.info(f"Rate-limit pacing: sleeping {time_to_wait:.1f}s...")
            time.sleep(time_to_wait)

    return results
