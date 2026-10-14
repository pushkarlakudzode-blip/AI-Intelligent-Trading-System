"""
Finnhub Historical Financial News Feasibility Proof-of-Concept Module.

This module provides completely isolated functionality to evaluate Finnhub
company news coverage, schema compatibility, text quality, duplicate frequency,
and in-memory NLP pipeline compatibility.

IMPORTANT SAFETY & ISOLATION RULES:
- Read-only feasibility test.
- Does NOT write to data/live/news_database.db or any production CSVs.
- Does NOT replace or alter Alpha Vantage or MarketAux implementations.
- API key is read strictly from environment variable: FINNHUB_API_KEY.
- Never logs or exposes full API keys.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timezone
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

FINNHUB_BASE_URL = "https://finnhub.io/api/v1/company-news"
DEFAULT_TEST_OUTPUT_DIR = PROJECT_ROOT / "data" / "live" / "finnhub_test"

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


def get_finnhub_token() -> Optional[str]:
    """
    Retrieve Finnhub API key from environment variable FINNHUB_API_KEY.
    
    Returns None if missing, blank, or placeholder.
    """
    token = os.getenv("FINNHUB_API_KEY")
    if token:
        clean = token.strip()
        if clean and not clean.startswith("<") and clean != "your_api_key_here":
            return clean
    return None


def mask_token(token: Optional[str]) -> str:
    """Safely mask an API key so it is never displayed in logs or reports."""
    if not token:
        return "None"
    if len(token) <= 8:
        return "***"
    return f"{token[:4]}...{token[-4:]}"


def parse_finnhub_timestamp(raw_time: Union[int, float, str, None]) -> Optional[str]:
    """
    Parse Finnhub timestamp (Unix epoch in seconds, or ISO string) to standardized UTC: YYYY-MM-DDTHH:MM:SSZ.
    """
    if raw_time is None:
        return None
    try:
        if isinstance(raw_time, (int, float)) or (isinstance(raw_time, str) and raw_time.strip().isdigit()):
            ts_val = float(raw_time)
            # If in milliseconds, convert to seconds
            if ts_val > 1e11:
                ts_val = ts_val / 1000.0
            dt = datetime.fromtimestamp(ts_val, tz=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        elif isinstance(raw_time, str):
            dt = pd.to_datetime(raw_time.strip())
            if dt.tzinfo is None:
                dt = dt.tz_localize("UTC")
            else:
                dt = dt.tz_convert("UTC")
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception as exc:
        logger.debug("Failed to parse Finnhub timestamp '%s': %s", raw_time, exc)
    return None


def normalize_finnhub_article(item: Dict[str, Any], target_ticker: str = "AAPL") -> Optional[Dict[str, Any]]:
    """
    Normalize raw Finnhub company-news JSON dictionary into standardized project schema.
    
    Standardized schema:
    - News_ID: Deterministic SHA-256 identifier
    - Ticker: Target ticker (e.g. AAPL)
    - Company_Name: Resolved company name
    - Published_At_UTC: Standardized ISO 8601 UTC timestamp
    - Headline: Headline text
    - Summary: Summary text
    - Source: Publisher source
    - URL: Original article URL
    - Provider: 'Finnhub'
    - Retrieved_At_UTC: Ingestion timestamp
    - Provider_Sentiment: 0.0 (internal NLP pipeline is strictly used instead)
    - Provider_Relevance: 0.50
    - Dataset_Type: 'REAL'
    """
    if not isinstance(item, dict):
        return None

    headline = (item.get("headline") or "").strip()
    if not headline:
        return None

    raw_dt = item.get("datetime")
    pub_utc = parse_finnhub_timestamp(raw_dt)
    if not pub_utc:
        return None

    summary = (item.get("summary") or "").strip()
    url = (item.get("url") or "").strip()
    source = (item.get("source") or "Unknown").strip()

    ticker_upper = target_ticker.upper()
    news_id = generate_news_id(ticker_upper, pub_utc, headline, url)

    return {
        "News_ID": news_id,
        "Ticker": ticker_upper,
        "Company_Name": COMPANY_NAMES.get(ticker_upper, ticker_upper),
        "Published_At_UTC": pub_utc,
        "Headline": headline,
        "Summary": summary,
        "Source": source,
        "URL": url,
        "Provider": "Finnhub",
        "Retrieved_At_UTC": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "Provider_Sentiment": 0.0,
        "Provider_Relevance": 0.50,
        "Dataset_Type": "REAL"
    }


def query_finnhub_company_news(
    symbol: str,
    from_date: str,
    to_date: str,
    token: Optional[str] = None,
    timeout: int = 15,
    max_retries: int = 3,
    backoff: float = 2.0
) -> Tuple[Optional[List[Dict[str, Any]]], str, int]:
    """
    Execute rate-limited query against Finnhub /company-news endpoint.
    
    Returns: (articles_list, status_message, http_status_code)
    """
    api_token = token or get_finnhub_token()
    if not api_token:
        msg = "FINNHUB_API_KEY is not configured."
        return None, msg, 401

    query_params = {
        "symbol": symbol.upper(),
        "from": from_date,
        "to": to_date,
        "token": api_token
    }

    encoded_params = urllib.parse.urlencode(query_params)
    url = f"{FINNHUB_BASE_URL}?{encoded_params}"

    headers = {
        "User-Agent": "AI-Trading-System-Finnhub-POC/1.0",
        "Accept": "application/json",
        "X-Finnhub-Token": api_token
    }

    req = urllib.request.Request(url, headers=headers, method="GET")

    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status_code = response.getcode()
                raw_bytes = response.read()
                data = json.loads(raw_bytes.decode("utf-8"))
                if isinstance(data, list):
                    return data, "OK", status_code
                elif isinstance(data, dict) and "error" in data:
                    return None, f"Finnhub Error: {data.get('error')}", status_code
                else:
                    return [], "OK (Empty / Unknown Format)", status_code
        except urllib.error.HTTPError as http_err:
            status_code = http_err.code
            try:
                err_body = http_err.read().decode("utf-8")
                err_json = json.loads(err_body)
                err_msg = err_json.get("error", err_body)
            except Exception:
                err_msg = str(http_err)

            if status_code in (401, 403):
                return None, f"Finnhub Auth/Plan Restriction ({status_code}): {err_msg}", status_code
            elif status_code == 429:
                return None, f"Finnhub Rate Limit Exceeded (429): {err_msg}", status_code
            elif status_code >= 500 and attempt < max_retries:
                time.sleep(backoff * attempt)
                continue
            return None, f"Finnhub HTTP Error {status_code}: {err_msg}", status_code
        except urllib.error.URLError as url_err:
            if attempt < max_retries:
                time.sleep(backoff * attempt)
                continue
            return None, f"Network connection failed: {url_err.reason}", 0
        except Exception as exc:
            return None, f"Unexpected error querying Finnhub: {str(exc)}", 0

    return None, "Max retries exceeded without response", 0


def check_text_quality(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze text quality across raw or normalized Finnhub articles.
    
    Metrics:
    - total_returned
    - headline_available
    - summary_available
    - usable_text_available (headline OR summary present)
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
            "usable_text_available": 0,
            "missing_usable_text": 0,
            "usable_text_pct": 0.0,
            "unique_sources": 0,
            "sources_list": []
        }

    hl_count = 0
    sm_count = 0
    usable_count = 0
    sources = set()

    for art in articles:
        hl = (art.get("headline") or art.get("Headline") or "").strip()
        sm = (art.get("summary") or art.get("Summary") or "").strip()
        src = (art.get("source") or art.get("Source") or "").strip() or "Unknown"
        sources.add(src)

        has_hl = bool(hl)
        has_sm = bool(sm)

        if has_hl:
            hl_count += 1
        if has_sm:
            sm_count += 1
        if has_hl or has_sm:
            usable_count += 1

    missing_usable = total - usable_count
    usable_pct = round((usable_count / total) * 100.0, 2)

    return {
        "total_returned": total,
        "headline_available": hl_count,
        "summary_available": sm_count,
        "usable_text_available": usable_count,
        "missing_usable_text": missing_usable,
        "usable_text_pct": usable_pct,
        "unique_sources": len(sources),
        "sources_list": sorted(list(sources))
    }


def check_duplicates(articles: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Perform multi-level duplicate check on articles using:
    1. Article ID (Finnhub 'id' or News_ID)
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
        pub = str(art.get("Published_At_UTC") or "")
        if not pub and "datetime" in art:
            pub = parse_finnhub_timestamp(art["datetime"]) or ""
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
    Run normalized Finnhub articles through existing internal NLP pipeline in-memory.
    
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
                "summary -> Summary",
                "datetime (Unix timestamp) -> Published_At_UTC (ISO 8601 UTC)",
                "id + metadata -> News_ID (SHA-256 deterministic hash)",
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
                "summary -> Summary",
                "datetime (Unix timestamp) -> Published_At_UTC (ISO 8601 UTC)",
                "id + metadata -> News_ID (SHA-256 deterministic hash)",
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
                "summary -> Summary",
                "datetime (Unix timestamp) -> Published_At_UTC (ISO 8601 UTC)",
                "id + metadata -> News_ID (SHA-256 deterministic hash)",
                "source -> Source",
                "url -> URL"
            ],
            "sample_results": []
        }


def get_mock_sample_articles() -> List[Dict[str, Any]]:
    """
    Generate realistic Finnhub mock company news items across 2019-2024 for offline evaluation.
    Matches Finnhub's /company-news payload structure.
    """
    return [
        {
            "category": "company",
            "datetime": 1546524000,  # 2019-01-03 14:00:00 UTC
            "headline": "Apple cuts revenue forecast on China economic weakness and fewer iPhone upgrades",
            "id": 2019010301,
            "image": "https://example.com/img1.jpg",
            "related": "AAPL",
            "source": "CNBC",
            "summary": "Apple lowered its first-quarter revenue guidance to 84 billion citing unexpected economic slowdown in Greater China.",
            "url": "https://www.cnbc.com/2019/01/03/apple-revenue-guidance-cut.html"
        },
        {
            "category": "company",
            "datetime": 1556740800,  # 2019-05-01 20:00:00 UTC
            "headline": "Apple reports second-quarter earnings beat and expands share repurchase by 75 billion",
            "id": 2019050102,
            "image": "https://example.com/img2.jpg",
            "related": "AAPL",
            "source": "Reuters",
            "summary": "Apple posted quarterly earnings and revenue that exceeded analyst consensus driven by resilient service sector revenues.",
            "url": "https://www.reuters.com/article/apple-earnings-q2-2019"
        },
        {
            "category": "company",
            "datetime": 1584180000,  # 2020-03-14 10:00:00 UTC
            "headline": "Apple temporarily closes all retail stores outside Greater China in response to coronavirus",
            "id": 2020031401,
            "image": "https://example.com/img3.jpg",
            "related": "AAPL",
            "source": "MarketWatch",
            "summary": "Chief Executive Tim Cook announced the company will shut all physical retail locations globally until March 27 to limit transmission.",
            "url": "https://www.marketwatch.com/story/apple-store-closures-2020"
        },
        {
            "category": "company",
            "datetime": 1597845600,  # 2020-08-19 14:00:00 UTC
            "headline": "Apple becomes first two trillion dollar US company as pandemic tech rally accelerates",
            "id": 2020081902,
            "image": "https://example.com/img4.jpg",
            "related": "AAPL",
            "source": "Bloomberg",
            "summary": "Apple shares rose above 467 dollars propelling the Cupertino tech giant past the historic two trillion market cap milestone.",
            "url": "https://www.bloomberg.com/news/apple-2-trillion-valuation"
        },
        {
            "category": "company",
            "datetime": 1618941600,  # 2021-04-20 18:00:00 UTC
            "headline": "Apple debuts M1-powered iPad Pro, redesigned colorful iMac, and AirTag tracking devices",
            "id": 2021042001,
            "image": "https://example.com/img5.jpg",
            "related": "AAPL",
            "source": "The Verge",
            "summary": "At its Spring Loaded keynote Apple refreshed its computing lineup with custom M1 silicon across consumer and professional lines.",
            "url": "https://www.theverge.com/2021/4/20/apple-spring-loaded-recap"
        },
        {
            "category": "company",
            "datetime": 1631289600,  # 2021-09-10 16:00:00 UTC
            "headline": "Federal judge issues mixed injunction in Epic Games antitrust lawsuit against Apple App Store",
            "id": 2021091002,
            "image": "https://example.com/img6.jpg",
            "related": "AAPL",
            "source": "Yahoo",
            "summary": "Judge Yvonne Gonzalez Rogers ordered Apple to allow external links to payment options while rejecting monopolization claims.",
            "url": "https://finance.yahoo.com/news/epic-games-apple-app-store-ruling"
        },
        {
            "category": "company",
            "datetime": 1641218400,  # 2022-01-03 14:00:00 UTC
            "headline": "Apple hits three trillion dollar market value on first trading session of 2022",
            "id": 2022010301,
            "image": "https://example.com/img7.jpg",
            "related": "AAPL",
            "source": "Wall Street Journal",
            "summary": "Shares briefly touched 182.86 dollars giving Apple an unprecedented 3 trillion valuation as investors praised cash flow stability.",
            "url": "https://www.wsj.com/articles/apple-3-trillion-market-cap-2022"
        },
        {
            "category": "company",
            "datetime": 1667779200,  # 2022-11-06 20:00:00 UTC
            "headline": "Apple warns of delayed iPhone 14 Pro shipments due to Zhengzhou facility production curbs",
            "id": 2022110602,
            "image": "https://example.com/img8.jpg",
            "related": "AAPL",
            "source": "Reuters",
            "summary": "COVID disruptions at Foxconn assembly plant will significantly constrain high-end iPhone supply during the crucial holiday season.",
            "url": "https://www.reuters.com/technology/apple-warns-iphone-shipment-delay-2022"
        },
        {
            "category": "company",
            "datetime": 1685984400,  # 2023-06-05 17:00:00 UTC
            "headline": "Apple reveals Vision Pro mixed reality headset priced at 3499 dollars at WWDC",
            "id": 2023060501,
            "image": "https://example.com/img9.jpg",
            "related": "AAPL",
            "source": "CNBC",
            "summary": "The Vision Pro represents Apple's ambitious entry into spatial computing featuring dual 4K micro-OLED displays and gesture tracking.",
            "url": "https://www.cnbc.com/2023/06/05/apple-wwdc-vision-pro-headset.html"
        },
        {
            "category": "company",
            "datetime": 1698958800,  # 2023-11-02 21:00:00 UTC
            "headline": "Apple sales slide for fourth quarter in a row but gross margins hit 45.2 percent",
            "id": 2023110202,
            "image": "https://example.com/img10.jpg",
            "related": "AAPL",
            "source": "Bloomberg",
            "summary": "Quarterly revenue fell 1 percent to 89.5 billion as Mac and iPad declines counterbalanced all-time record revenue in Services.",
            "url": "https://www.bloomberg.com/news/apple-fourth-quarter-earnings-2023"
        },
        {
            "category": "company",
            "datetime": 1718042400,  # 2024-06-10 18:00:00 UTC
            "headline": "Apple unveils Apple Intelligence generative AI platform and partnership with OpenAI",
            "id": 2024061001,
            "image": "https://example.com/img11.jpg",
            "related": "AAPL",
            "source": "TechCrunch",
            "summary": "Apple announced personal intelligence features baked into iOS 18 with on-device models and Private Cloud Compute security architecture.",
            "url": "https://techcrunch.com/2024/06/10/apple-intelligence-openai-wwdc-2024"
        },
        {
            "category": "company",
            "datetime": 1730403600,  # 2024-10-31 20:00:00 UTC
            "headline": "Apple reports fourth-quarter revenue up 6 percent to 94.9 billion on early iPhone 16 demand",
            "id": 2024103102,
            "image": "https://example.com/img12.jpg",
            "related": "AAPL",
            "source": "Seeking Alpha",
            "summary": "Apple topped quarterly estimates with 94.93 billion in revenue and record active device installations across all major categories.",
            "url": "https://seekingalpha.com/news/4164321-apple-q4-earnings-beat"
        }
    ]
