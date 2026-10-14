"""
Historical Financial News Ingestion Module (Alpha Vantage).

Fetches historical financial news articles via Alpha Vantage's NEWS_SENTIMENT API,
normalizes records to UTC timestamps, applies deterministic deduplication,
validates quality, and executes the NLP enrichment pipeline.

API Key is read strictly from environment variable: ALPHA_VANTAGE_API_KEY.
"""

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import urllib.parse
import urllib.request
import json
import pandas as pd

from dotenv import load_dotenv

from src.data.data_storage import (
    generate_news_id,
    store_articles,
    store_nlp_results,
    log_validation_issue,
    get_existing_news_ids,
    validate_article
)
from src.nlp.news_nlp_pipeline import process_articles_nlp

# Load environment variables from .env if present
load_dotenv()

logger = logging.getLogger(__name__)

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

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


def get_api_key() -> Optional[str]:
    """Retrieve Alpha Vantage API key from environment variable."""
    key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if key:
        clean = key.strip()
        if clean and not clean.startswith("<") and clean != "your_api_key_here":
            return clean
    return None


def format_av_timestamp(dt_str: str, is_end_of_day: bool = False) -> str:
    """Convert YYYY-MM-DD to Alpha Vantage format YYYYMMDDTHHMM."""
    dt = pd.to_datetime(dt_str)
    if is_end_of_day and dt.hour == 0 and dt.minute == 0:
        dt = dt.replace(hour=23, minute=59)
    return dt.strftime("%Y%m%dT%H%M")


def parse_av_time_published(time_str: str) -> str:
    """Parse Alpha Vantage time string YYYYMMDDTHHMMSS into ISO 8601 UTC string."""
    try:
        # Expected: 20240415T133000
        dt = datetime.strptime(time_str[:15], "%Y%m%dT%H%M%S")
        dt_utc = dt.replace(tzinfo=timezone.utc)
        return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        # Fallback to pandas parser
        dt = pd.to_datetime(time_str)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        else:
            dt = dt.tz_convert("UTC")
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_av_article(item: Dict, target_ticker: str) -> Optional[Dict]:
    """Normalize raw Alpha Vantage news JSON dictionary into our standardized schema."""
    title = item.get("title", "").strip()
    if not title:
        return None
        
    time_pub_raw = item.get("time_published", "")
    if not time_pub_raw:
        return None
        
    pub_utc = parse_av_time_published(time_pub_raw)
    url = item.get("url", "").strip()
    
    # Calculate deterministic News_ID
    news_id = generate_news_id(target_ticker, pub_utc, title, url)
    
    # Extract provider sentiment & relevance for this specific ticker if available
    provider_sentiment = float(item.get("overall_sentiment_score", 0.0) or 0.0)
    provider_relevance = 0.50
    for ts in item.get("ticker_sentiment", []):
        if ts.get("ticker", "").upper() == target_ticker.upper():
            try:
                provider_relevance = float(ts.get("relevance_score", 0.50) or 0.50)
                provider_sentiment = float(ts.get("ticker_sentiment_score", provider_sentiment) or provider_sentiment)
            except Exception:
                pass
            break

    return {
        "News_ID": news_id,
        "Ticker": target_ticker.upper(),
        "Company_Name": COMPANY_NAMES.get(target_ticker.upper(), target_ticker.upper()),
        "Published_At_UTC": pub_utc,
        "Headline": title,
        "Summary": item.get("summary", "").strip(),
        "Source": item.get("source", "Unknown"),
        "URL": url,
        "Provider": "Alpha Vantage",
        "Retrieved_At_UTC": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "Provider_Sentiment": round(provider_sentiment, 4),
        "Provider_Relevance": round(provider_relevance, 4),
        "Dataset_Type": "REAL"
    }


def query_alpha_vantage_news(
    ticker: str,
    time_from: Optional[str] = None,
    time_to: Optional[str] = None,
    sort: str = "LATEST",
    limit: int = 50,
    max_retries: int = 3,
    backoff_factor: float = 2.0
) -> Tuple[List[Dict], str]:
    """
    Execute rate-limited, retried REST query against Alpha Vantage NEWS_SENTIMENT endpoint.
    
    Returns: (raw_articles_list, status_message)
    """
    api_key = get_api_key()
    if not api_key:
        msg = "ALPHA_VANTAGE_API_KEY is not configured in environment. Skipping live network call."
        logger.warning(msg)
        return [], msg

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker.upper(),
        "sort": sort,
        "limit": str(limit),
        "apikey": api_key
    }
    if time_from:
        params["time_from"] = format_av_timestamp(time_from, is_end_of_day=False)
    if time_to:
        params["time_to"] = format_av_timestamp(time_to, is_end_of_day=True)

    url = f"{ALPHA_VANTAGE_BASE_URL}?{urllib.parse.urlencode(params)}"
    
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AI-Trading-System/2.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                
                # Check for rate limit notices
                if "Note" in data:
                    note = data["Note"]
                    logger.warning(f"Alpha Vantage Rate Limit Note (Attempt {attempt}): {note}")
                    # If daily quota limit reached, never retry repeatedly
                    if any(w in note.lower() for w in ["25 requests", "per day", "daily quota", "daily limit", "maximum of 25"]):
                        return [], f"Rate limit reached: {note}"
                    if attempt < max_retries and backoff_factor > 0:
                        sleep_time = backoff_factor ** attempt * 5
                        logger.info(f"Backing off for {sleep_time:.1f} seconds...")
                        time.sleep(sleep_time)
                        continue
                    return [], f"Rate limit reached: {note}"

                if "Information" in data:
                    info_msg = data["Information"]
                    logger.warning(f"Alpha Vantage Quota/Information Note: {info_msg}")
                    return [], f"Rate limit reached: {info_msg}"
                    
                if "Error Message" in data:
                    err = data["Error Message"]
                    logger.error(f"Alpha Vantage API Error: {err}")
                    return [], f"API Error: {err}"
                    
                feed = data.get("feed", [])
                logger.info(f"Retrieved {len(feed)} raw articles for {ticker} from Alpha Vantage.")
                return feed, "SUCCESS"
                
        except Exception as e:
            logger.error(f"HTTP request failed (Attempt {attempt}): {e}")
            if attempt < max_retries:
                time.sleep(backoff_factor ** attempt)
            else:
                return [], f"Network error after {max_retries} attempts: {e}"

    return [], "MAX_RETRIES_EXCEEDED"


def fetch_historical_news(
    ticker: str,
    start_date: str,
    end_date: str,
    limit: int = 100,
    run_nlp: bool = True
) -> Dict[str, Union[str, int, float]]:
    """
    Fetch historical news articles for a ticker over a date window,
    normalize, deduplicate, persist to storage, and run NLP pipeline.
    
    Returns telemetry dictionary:
    ['ticker', 'status', 'raw_fetched', 'added', 'duplicates', 'rejected']
    """
    clean_ticker = ticker.strip().upper()
    if clean_ticker not in SUPPORTED_TICKERS:
        raise ValueError(f"Ticker {clean_ticker} is not in the supported 10-company universe.")

    logger.info(f"Fetching historical news for {clean_ticker} from {start_date} to {end_date}...")
    
    feed, status = query_alpha_vantage_news(
        ticker=clean_ticker,
        time_from=start_date,
        time_to=end_date,
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

    # Normalize records
    normalized_list = []
    for item in feed:
        norm = normalize_av_article(item, clean_ticker)
        if norm:
            normalized_list.append(norm)

    # Determine genuinely new, valid items before storing to run NLP strictly on new records
    existing_ids = get_existing_news_ids()
    newly_added_items = []
    for art in normalized_list:
        is_valid, _ = validate_article(art)
        if is_valid and art["News_ID"] not in existing_ids:
            newly_added_items.append(art)
            existing_ids.add(art["News_ID"])

    # Store articles
    added, dups, rejs = store_articles(normalized_list)
    logger.info(f"Storage for {clean_ticker}: Added={added}, Duplicates={dups}, Rejected={rejs}")

    # Run NLP strictly on newly added articles
    if run_nlp and newly_added_items:
        new_df = pd.DataFrame(newly_added_items)
        nlp_df, daily_df = process_articles_nlp(new_df)
        store_nlp_results(nlp_df, daily_df)
        logger.info(f"NLP pipeline processed {len(nlp_df)} articles across {len(daily_df)} daily sessions.")

    return {
        "ticker": clean_ticker,
        "status": status,
        "raw_fetched": len(feed),
        "added": added,
        "duplicates": dups,
        "rejected": rejs
    }


def fetch_historical_news_windowed(
    ticker: str,
    windows: List[Tuple[str, str]],
    limit_per_window: int = 50,
    inter_window_delay: float = 12.5,
    run_nlp: bool = True
) -> List[Dict[str, Union[str, int, float, bool]]]:
    """
    Fetch historical news articles across multiple bounded date windows for a ticker.
    Enforces deduplication, tracks per-window capping, and executes NLP on newly added records.
    
    Args:
        ticker: Target company ticker.
        windows: List of (start_date, end_date) tuples, e.g. [('2025-01-01', '2025-01-07'), ...].
        limit_per_window: API limit per sub-window (default: 50).
        inter_window_delay: Rate-limit sleep duration between sub-windows (default: 12.5s).
        run_nlp: Whether to run Loughran-McDonald NLP pipeline on new articles.
        
    Returns:
        List of dicts containing per-window audit records.
    """
    clean_ticker = ticker.strip().upper()
    if clean_ticker not in SUPPORTED_TICKERS:
        raise ValueError(f"Ticker {clean_ticker} is not in the supported 10-company universe.")
        
    results = []
    total_windows = len(windows)
    
    for idx, (w_start, w_end) in enumerate(windows):
        logger.info(f"[{clean_ticker}] Sub-window {idx+1}/{total_windows}: {w_start} to {w_end} (limit={limit_per_window})...")
        
        feed, status = query_alpha_vantage_news(
            ticker=clean_ticker,
            time_from=w_start,
            time_to=w_end,
            sort="LATEST",
            limit=limit_per_window
        )
        
        raw_count = len(feed)
        is_capped = (raw_count >= limit_per_window)
        
        # Normalize articles
        normalized_list = []
        for item in feed:
            norm = normalize_av_article(item, clean_ticker)
            if norm:
                normalized_list.append(norm)
                
        # Determine actual range returned
        if normalized_list:
            actual_min = min(art["Published_At_UTC"] for art in normalized_list)
            actual_max = max(art["Published_At_UTC"] for art in normalized_list)
        else:
            actual_min = "N/A"
            actual_max = "N/A"
            
        # Determine genuinely new, valid items before storing
        existing_ids = get_existing_news_ids()
        newly_added_items = []
        for art in normalized_list:
            is_valid, _ = validate_article(art)
            if is_valid and art["News_ID"] not in existing_ids:
                newly_added_items.append(art)
                existing_ids.add(art["News_ID"])
                
        # Store articles
        added, dups, rejs = store_articles(normalized_list)
        logger.info(f"[{clean_ticker}] Window {w_start}->{w_end}: Raw={raw_count}, Added={added}, Dups={dups}, Rej={rejs}, Capped={is_capped}")
        
        # Run NLP strictly on newly added articles
        nlp_count = 0
        if run_nlp and newly_added_items:
            new_df = pd.DataFrame(newly_added_items)
            nlp_df, daily_df = process_articles_nlp(new_df)
            store_nlp_results(nlp_df, daily_df)
            nlp_count = len(nlp_df)
            logger.info(f"[{clean_ticker}] Processed {nlp_count} NLP records for newly added articles.")
            
        window_record = {
            "ticker": clean_ticker,
            "window_start": w_start,
            "window_end": w_end,
            "status": status,
            "raw_fetched": raw_count,
            "new_added": added,
            "duplicates": dups,
            "rejected": rejs,
            "nlp_records": nlp_count,
            "limit_reached": is_capped,
            "actual_earliest": actual_min,
            "actual_latest": actual_max
        }
        results.append(window_record)
        
        # Rate-limit pacing between sub-windows
        if idx < total_windows - 1:
            logger.info(f"Rate-limit pacing: sleeping {inter_window_delay:.1f}s...")
            time.sleep(inter_window_delay)
            
    return results
