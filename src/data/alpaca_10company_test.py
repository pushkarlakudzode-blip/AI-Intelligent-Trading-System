"""
10-Company Alpaca Historical Financial News Feasibility Module.

This module provides completely isolated functionality to evaluate Alpaca Markets
News API (Benzinga-powered) historical coverage across all 10 project target companies:
AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, V, JNJ.

IMPORTANT SAFETY & ISOLATION RULES:
- Read-only feasibility test.
- Does NOT write to data/live/news_database.db or any production CSVs.
- Does NOT replace or alter Alpha Vantage, MarketAux, or Finnhub implementations.
- API credentials are read strictly from environment variables:
  ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY.
- Never logs or exposes full API credentials.
"""

import os
import sys
import json
import time
import re
import html
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import urllib.parse
import urllib.request
import urllib.error

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Ensure project root is accessible
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_storage import generate_news_id
from src.nlp.news_nlp_pipeline import process_articles_nlp

load_dotenv()
logger = logging.getLogger(__name__)

ALPACA_NEWS_BASE_URL = "https://data.alpaca.markets/v1beta1/news"
DEFAULT_TEST_OUTPUT_DIR = PROJECT_ROOT / "data" / "live" / "alpaca_test"

TARGET_COMPANIES: Dict[str, Dict[str, str]] = {
    "AAPL": {"name": "Apple", "query_symbol": "AAPL"},
    "MSFT": {"name": "Microsoft", "query_symbol": "MSFT"},
    "GOOGL": {"name": "Alphabet", "query_symbol": "GOOGL"},
    "AMZN": {"name": "Amazon", "query_symbol": "AMZN"},
    "NVDA": {"name": "NVIDIA", "query_symbol": "NVDA"},
    "META": {"name": "Meta Platforms", "query_symbol": "META,FB"},
    "TSLA": {"name": "Tesla", "query_symbol": "TSLA"},
    "JPM": {"name": "JPMorgan Chase", "query_symbol": "JPM"},
    "V": {"name": "Visa", "query_symbol": "V"},
    "JNJ": {"name": "Johnson & Johnson", "query_symbol": "JNJ"}
}


def get_alpaca_credentials() -> Tuple[Optional[str], Optional[str]]:
    """
    Retrieve Alpaca API credentials from environment variables.
    
    Returns: (key_id, secret_key) or (None, None) if unconfigured.
    """
    key_id = os.getenv("ALPACA_API_KEY_ID")
    secret_key = os.getenv("ALPACA_API_SECRET_KEY")

    clean_key = key_id.strip() if key_id else None
    clean_secret = secret_key.strip() if secret_key else None

    if clean_key and (clean_key.startswith("<") or clean_key == "your_key_id_here"):
        clean_key = None
    if clean_secret and (clean_secret.startswith("<") or clean_secret == "your_secret_key_here"):
        clean_secret = None

    return clean_key, clean_secret


def mask_credential(cred: Optional[str]) -> str:
    """Safely mask an API key or secret so it is never displayed in logs or reports."""
    if not cred:
        return "None"
    if len(cred) <= 8:
        return "***"
    return f"{cred[:4]}...{cred[-4:]}"


def parse_alpaca_timestamp(raw_time: Union[str, int, float, None]) -> Optional[str]:
    """
    Parse Alpaca RFC-3339 / ISO 8601 timestamp string to standardized UTC string: YYYY-MM-DDTHH:MM:SSZ.
    """
    if raw_time is None:
        return None
    try:
        if isinstance(raw_time, str):
            dt = pd.to_datetime(raw_time.strip())
            if dt.tzinfo is None:
                dt = dt.tz_localize("UTC")
            else:
                dt = dt.tz_convert("UTC")
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        elif isinstance(raw_time, (int, float)):
            dt = datetime.fromtimestamp(float(raw_time), tz=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception as exc:
        logger.debug("Failed to parse Alpaca timestamp '%s': %s", raw_time, exc)
    return None


def clean_html_content(raw_html: str) -> str:
    """Strip HTML markup and unescape characters to produce clean plain text."""
    if not raw_html or not isinstance(raw_html, str):
        return ""
    clean = re.sub(r"<[^>]+>", " ", raw_html)
    clean = html.unescape(clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def normalize_alpaca_article(item: Dict[str, Any], target_ticker: str = "AAPL") -> Optional[Dict[str, Any]]:
    """
    Normalize raw Alpaca news JSON dictionary into standardized project schema.
    
    Standardized schema:
    - News_ID: Deterministic SHA-256 identifier
    - Ticker: Target company ticker (e.g. AAPL)
    - Company_Name: Resolved company name
    - Published_At_UTC: Standardized ISO 8601 UTC timestamp
    - Headline: Article headline
    - Summary: Cleaned summary or extracted content snippet
    - Source: Publishing source (Benzinga, etc.)
    - URL: Canonical article URL
    - Provider: 'Alpaca'
    - Retrieved_At_UTC: Ingestion timestamp
    - Provider_Sentiment: 0.0 (internal NLP pipeline strictly used instead)
    - Provider_Relevance: 0.50
    - Dataset_Type: 'REAL'
    """
    if not isinstance(item, dict):
        return None

    headline = (item.get("headline") or "").strip()
    if not headline:
        return None

    pub_raw = item.get("created_at") or item.get("updated_at")
    pub_utc = parse_alpaca_timestamp(pub_raw)
    if not pub_utc:
        return None

    # Check future-dated lookahead protection (beyond 15-minute clock drift)
    pub_dt = datetime.strptime(pub_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    now_utc = datetime.now(timezone.utc)
    if pub_dt > now_utc + timedelta(minutes=15):
        logger.warning("Rejected future-dated article: %s (%s)", headline, pub_utc)
        return None

    summary = (item.get("summary") or "").strip()
    if not summary and item.get("content"):
        cleaned_content = clean_html_content(item["content"])
        if cleaned_content:
            summary = cleaned_content[:250].rsplit(" ", 1)[0] + "..." if len(cleaned_content) > 250 else cleaned_content

    url = (item.get("url") or "").strip()
    source = (item.get("source") or "Benzinga").strip()
    ticker_upper = target_ticker.upper()
    if ticker_upper == "FB":
        ticker_upper = "META"
    comp_info = TARGET_COMPANIES.get(ticker_upper, {})
    comp_name = comp_info.get("name", ticker_upper)

    news_id = generate_news_id(ticker_upper, pub_utc, headline, url)

    return {
        "News_ID": news_id,
        "Ticker": ticker_upper,
        "Company_Name": comp_name,
        "Published_At_UTC": pub_utc,
        "Headline": headline,
        "Summary": summary,
        "Source": source,
        "URL": url,
        "Provider": "Alpaca",
        "Retrieved_At_UTC": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "Provider_Sentiment": 0.0,
        "Provider_Relevance": 0.50,
        "Dataset_Type": "REAL"
    }


def query_alpaca_news(
    params: Dict[str, Any],
    key_id: Optional[str] = None,
    secret_key: Optional[str] = None,
    timeout: int = 15,
    max_retries: int = 3,
    backoff: float = 1.5
) -> Tuple[Optional[Dict[str, Any]], str, int, Dict[str, str]]:
    """
    Execute rate-limited query against Alpaca /news endpoint.
    
    Returns: (parsed_json_dict, status_message, http_status_code, rate_limit_headers)
    """
    clean_key, clean_secret = get_alpaca_credentials()
    api_key_id = key_id or clean_key
    api_secret = secret_key or clean_secret

    if not api_key_id or not api_secret:
        msg = "ALPACA_API_KEY_ID or ALPACA_API_SECRET_KEY is not configured."
        return None, msg, 401, {}

    query_params = dict(params)
    encoded_params = urllib.parse.urlencode(query_params)
    url = f"{ALPACA_NEWS_BASE_URL}?{encoded_params}"

    headers = {
        "User-Agent": "AI-Trading-System-Alpaca-10Company-POC/1.0",
        "Accept": "application/json",
        "APCA-API-KEY-ID": api_key_id,
        "APCA-API-SECRET-KEY": api_secret
    }

    req = urllib.request.Request(url, headers=headers, method="GET")

    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status_code = response.getcode()
                raw_bytes = response.read()
                data = json.loads(raw_bytes.decode("utf-8"))

                rate_headers = {}
                for h_name, h_val in response.headers.items():
                    if "ratelimit" in h_name.lower() or "rate-limit" in h_name.lower():
                        rate_headers[h_name] = h_val

                return data, "OK", status_code, rate_headers
        except urllib.error.HTTPError as http_err:
            status_code = http_err.code
            try:
                err_body = http_err.read().decode("utf-8")
                err_json = json.loads(err_body)
                err_msg = err_json.get("message", err_body)
            except Exception:
                err_msg = str(http_err)

            rate_headers = {}
            for h_name, h_val in http_err.headers.items():
                if "ratelimit" in h_name.lower() or "rate-limit" in h_name.lower():
                    rate_headers[h_name] = h_val

            if status_code in (401, 403):
                return None, f"Alpaca Auth Error ({status_code}): {err_msg}", status_code, rate_headers
            elif status_code == 429:
                return None, f"Alpaca Rate Limit (429): {err_msg}", status_code, rate_headers
            elif status_code >= 500 and attempt < max_retries:
                time.sleep(backoff * attempt)
                continue
            return None, f"Alpaca HTTP Error {status_code}: {err_msg}", status_code, rate_headers
        except urllib.error.URLError as url_err:
            if attempt < max_retries:
                time.sleep(backoff * attempt)
                continue
            return None, f"Network connection failed: {url_err.reason}", 0, {}
        except Exception as exc:
            return None, f"Unexpected error querying Alpaca: {str(exc)}", 0, {}

    return None, "Max retries exceeded without response", 0, {}


def check_text_quality(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze text quality across raw or normalized Alpaca articles.
    
    Metrics:
    - total_returned
    - headline_available
    - summary_available
    - content_available
    - usable_text_available (headline OR summary/content present)
    - missing_usable_text
    - usable_text_pct
    - unique_sources
    """
    total = len(articles)
    if total == 0:
        return {
            "total_returned": 0,
            "headline_available": 0,
            "summary_available": 0,
            "content_available": 0,
            "usable_text_available": 0,
            "missing_usable_text": 0,
            "usable_text_pct": 0.0,
            "unique_sources": 0,
            "sources_list": []
        }

    hl_count = 0
    sm_count = 0
    cnt_count = 0
    usable_count = 0
    sources = set()

    for art in articles:
        hl = (art.get("headline") or art.get("Headline") or "").strip()
        sm = (art.get("summary") or art.get("Summary") or "").strip()
        cnt = (art.get("content") or "").strip()
        src = (art.get("source") or art.get("Source") or "").strip() or "Benzinga"
        sources.add(src)

        has_hl = bool(hl)
        has_sm = bool(sm)
        has_cnt = bool(cnt)

        if has_hl:
            hl_count += 1
        if has_sm:
            sm_count += 1
        if has_cnt:
            cnt_count += 1
        if has_hl or has_sm or has_cnt:
            usable_count += 1

    missing_usable = total - usable_count
    usable_pct = round((usable_count / total) * 100.0, 2)

    return {
        "total_returned": total,
        "headline_available": hl_count,
        "summary_available": sm_count,
        "content_available": cnt_count,
        "usable_text_available": usable_count,
        "missing_usable_text": missing_usable,
        "usable_text_pct": usable_pct,
        "unique_sources": len(sources),
        "sources_list": sorted(list(sources))
    }


def check_duplicates(articles: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Perform multi-level duplicate check on articles using:
    1. Article ID (Alpaca 'id' or News_ID)
    2. URL
    3. Normalized headline + publication date
    """
    seen_ids = set()
    seen_urls = set()
    seen_title_dates = set()

    dup_ids = 0
    dup_urls = 0
    dup_title_dates = 0

    for art in articles:
        art_id = str(art.get("News_ID") or art.get("id") or "").strip()
        if art_id:
            if art_id in seen_ids:
                dup_ids += 1
            else:
                seen_ids.add(art_id)

        url = str(art.get("URL") or art.get("url") or "").strip().lower()
        if url:
            if url in seen_urls:
                dup_urls += 1
            else:
                seen_urls.add(url)

        title = (art.get("Headline") or art.get("headline") or "").strip().lower()
        pub = str(art.get("Published_At_UTC") or art.get("created_at") or "").strip()
        date_part = pub[:10]
        if title and date_part:
            key = f"{date_part}|{title}"
            if key in seen_title_dates:
                dup_title_dates += 1
            else:
                seen_title_dates.add(key)

    return {
        "total_articles": len(articles),
        "duplicate_ids": dup_ids,
        "duplicate_urls": dup_urls,
        "duplicate_title_dates": dup_title_dates,
        "unique_ids": len(seen_ids),
        "unique_urls": len(seen_urls)
    }


def test_nlp_compatibility(
    normalized_articles: List[Dict[str, Any]],
    max_sample: int = 20
) -> Dict[str, Any]:
    """
    Run normalized Alpaca articles through existing internal NLP pipeline in-memory.
    
    STRICTLY NON-PERSISTENT:
    - Zero writes to database.
    - Zero writes to production CSVs.
    """
    sample = normalized_articles[:max_sample]
    if not sample:
        return {
            "sample_size": 0,
            "successfully_processed": 0,
            "failed": 0,
            "required_normalization": [
                "headline -> Headline",
                "summary / content -> Summary",
                "created_at (RFC-3339) -> Published_At_UTC (ISO 8601 UTC)",
                "id + metadata -> News_ID (SHA-256 deterministic digest)",
                "source -> Source",
                "url -> URL"
            ],
            "sample_results": []
        }

    df_in = pd.DataFrame(sample)
    try:
        enriched_df, daily_df = process_articles_nlp(df_in)
        success_count = len(enriched_df)
        failed_count = len(sample) - success_count

        sample_summary = []
        for _, row in enriched_df.head(5).iterrows():
            sample_summary.append({
                "News_ID": row.get("News_ID"),
                "Ticker": row.get("Ticker"),
                "Headline": row.get("Headline")[:60] + "..." if len(row.get("Headline", "")) > 60 else row.get("Headline"),
                "Sentiment_Label": row.get("Sentiment_Label"),
                "Sentiment_Score": row.get("Sentiment_Score"),
                "Relevance_Score": row.get("Relevance_Score"),
                "Importance_Score": row.get("Importance_Score"),
                "Triage_Score": row.get("Triage_Score")
            })

        return {
            "sample_size": len(sample),
            "successfully_processed": success_count,
            "failed": failed_count,
            "required_normalization": [
                "headline -> Headline",
                "summary / content -> Summary",
                "created_at (RFC-3339) -> Published_At_UTC (ISO 8601 UTC)",
                "id + metadata -> News_ID (SHA-256 deterministic digest)",
                "source -> Source",
                "url -> URL"
            ],
            "sample_results": sample_summary
        }
    except Exception as exc:
        logger.error("NLP compatibility execution error: %s", exc)
        return {
            "sample_size": len(sample),
            "successfully_processed": 0,
            "failed": len(sample),
            "error": str(exc),
            "required_normalization": [
                "headline -> Headline",
                "summary / content -> Summary",
                "created_at (RFC-3339) -> Published_At_UTC (ISO 8601 UTC)",
                "id + metadata -> News_ID (SHA-256 deterministic digest)",
                "source -> Source",
                "url -> URL"
            ],
            "sample_results": []
        }


def estimate_10company_scale() -> Dict[str, Any]:
    """
    Compute estimated dataset volume and API calls needed for a full 2019-2024 backfill.
    
    ESTIMATES based on empirical monthly sampling:
    - High-volume tech (AAPL, TSLA, NVDA, AMZN, MSFT, GOOGL, META): ~1,200 to 2,000 articles/year (~7,200 - 12,000 per company).
    - Mature / defensive (JPM, V, JNJ): ~400 to 750 articles/year (~2,400 - 4,500 per company).
    """
    company_estimates = {
        "AAPL": {"annual_est": 1800, "six_year_est": 10800, "category": "Mega-Cap Tech (High Volume)"},
        "MSFT": {"annual_est": 1600, "six_year_est": 9600, "category": "Mega-Cap Tech (High Volume)"},
        "GOOGL": {"annual_est": 1500, "six_year_est": 9000, "category": "Mega-Cap Tech (High Volume)"},
        "AMZN": {"annual_est": 1600, "six_year_est": 9600, "category": "Mega-Cap Tech (High Volume)"},
        "NVDA": {"annual_est": 1500, "six_year_est": 9000, "category": "Semiconductor / AI (High Volume)"},
        "META": {"annual_est": 1400, "six_year_est": 8400, "category": "Mega-Cap Tech / Social (High Volume)"},
        "TSLA": {"annual_est": 1800, "six_year_est": 10800, "category": "EV / Tech (High Volume)"},
        "JPM": {"annual_est": 700, "six_year_est": 4200, "category": "Financial Services (Medium Volume)"},
        "V": {"annual_est": 500, "six_year_est": 3000, "category": "Payments Network (Medium Volume)"},
        "JNJ": {"annual_est": 600, "six_year_est": 3600, "category": "Healthcare / Consumer (Medium Volume)"}
    }

    total_articles_est = sum(c["six_year_est"] for c in company_estimates.values())
    # With limit=50 articles per request
    requests_needed_est = int(np.ceil(total_articles_est / 50.0))
    # At 150 requests/min (under 200 rate limit)
    estimated_minutes = round(requests_needed_est / 150.0, 1)

    return {
        "company_breakdown": company_estimates,
        "total_estimated_articles": total_articles_est,
        "page_limit_used": 50,
        "total_estimated_requests": requests_needed_est,
        "estimated_duration_minutes": estimated_minutes,
        "rate_limit_headroom_pct": "75% (using 150 req/min vs 200 req/min limit)"
    }


def get_mock_10company_samples() -> List[Dict[str, Any]]:
    """
    Generate realistic multi-company mock articles across 2019-2024 for offline unit testing.
    """
    tickers = list(TARGET_COMPANIES.keys())
    sample_list = []
    base_id = 90000000

    for idx, ticker in enumerate(tickers):
        comp_name = TARGET_COMPANIES[ticker]["name"]
        for yr in [2019, 2021, 2024]:
            base_id += 1
            sample_list.append({
                "id": base_id,
                "headline": f"{comp_name} Reports Strategic Growth and Quarterly Milestone",
                "author": "Benzinga Newsdesk",
                "created_at": f"{yr}-06-15T14:30:00Z",
                "summary": f"{comp_name} announces expanded operating initiatives across enterprise segments.",
                "content": f"<p>{comp_name} executives reviewed quarterly performance.</p>",
                "url": f"https://www.benzinga.com/node/{base_id}",
                "symbols": [ticker],
                "source": "benzinga"
            })

    return sample_list
