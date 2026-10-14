"""
MarketAux Historical Financial News Feasibility Proof-of-Concept Module.

This module provides completely isolated functionality to evaluate MarketAux
financial news coverage, schema compatibility, text quality, duplicate frequency,
and in-memory NLP pipeline compatibility.

IMPORTANT SAFETY & ISOLATION RULES:
- Read-only feasibility test.
- Does NOT write to data/live/news_database.db or any production CSVs.
- Does NOT replace or alter Alpha Vantage ingestion.
- API token is read strictly from environment variable: MARKETAUX_API_TOKEN.
- Never logs or exposes full API tokens.
"""

import os
import sys
import json
import time
import logging
import hashlib
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

MARKETAUX_BASE_URL = "https://api.marketaux.com/v1/news/all"
DEFAULT_TEST_OUTPUT_DIR = PROJECT_ROOT / "data" / "live" / "marketaux_test"

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


def get_marketaux_token() -> Optional[str]:
    """
    Retrieve MarketAux API token from environment variable MARKETAUX_API_TOKEN.
    
    Returns None if missing, blank, or placeholder.
    """
    token = os.getenv("MARKETAUX_API_TOKEN")
    if token:
        clean = token.strip()
        if clean and not clean.startswith("<") and clean != "your_token_here":
            return clean
    return None


def mask_token(token: Optional[str]) -> str:
    """Safely mask an API token so it is never displayed in logs or reports."""
    if not token:
        return "None"
    if len(token) <= 8:
        return "***"
    return f"{token[:4]}...{token[-4:]}"


def parse_marketaux_timestamp(time_str: str) -> Optional[str]:
    """
    Parse MarketAux ISO 8601 timestamp string into standardized UTC string: YYYY-MM-DDTHH:MM:SSZ.
    
    Handles:
    - 2024-10-31T20:30:00.000000Z
    - 2024-10-31T20:30:00Z
    - 2024-10-31 20:30:00
    """
    if not time_str or not isinstance(time_str, str):
        return None
    time_clean = time_str.strip()
    try:
        # Fast path with pandas to handle variable microsecond precision
        dt = pd.to_datetime(time_clean)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        else:
            dt = dt.tz_convert("UTC")
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception as exc:
        logger.debug("Failed to parse timestamp '%s': %s", time_str, exc)
        return None


def normalize_marketaux_article(item: Dict[str, Any], target_ticker: str = "AAPL") -> Optional[Dict[str, Any]]:
    """
    Normalize raw MarketAux article JSON dictionary into our standardized project schema.
    
    Standardized fields:
    - News_ID: Deterministic SHA-256 identifier
    - Ticker: Target company ticker (e.g. AAPL)
    - Company_Name: Resolved company name
    - Published_At_UTC: Standardized ISO 8601 UTC timestamp
    - Headline: Article title
    - Summary: Combined description / snippet
    - Source: Publishing outlet / domain
    - URL: Original article URL
    - Provider: 'MarketAux'
    - Retrieved_At_UTC: Ingestion timestamp
    - Provider_Sentiment: Entity-specific sentiment if provided
    - Provider_Relevance: Entity match / relevance score
    - Dataset_Type: 'REAL'
    """
    if not isinstance(item, dict):
        return None

    title = (item.get("title") or "").strip()
    if not title:
        return None

    pub_raw = item.get("published_at") or ""
    pub_utc = parse_marketaux_timestamp(pub_raw)
    if not pub_utc:
        return None

    url = (item.get("url") or "").strip()

    # Extract summary from description and/or snippet
    desc = (item.get("description") or "").strip()
    snip = (item.get("snippet") or "").strip()
    summary = desc if desc else snip

    # Extract source domain or name
    source_val = item.get("source")
    if isinstance(source_val, dict):
        source = source_val.get("domain") or source_val.get("name") or "Unknown"
    elif isinstance(source_val, str) and source_val.strip():
        source = source_val.strip()
    else:
        source = "Unknown"

    ticker_upper = target_ticker.upper()
    news_id = generate_news_id(ticker_upper, pub_utc, title, url)

    # Extract provider sentiment & relevance for this specific ticker if available
    provider_sentiment = 0.0
    provider_relevance = 0.50
    entities = item.get("entities", [])
    if isinstance(entities, list):
        for ent in entities:
            if isinstance(ent, dict) and ent.get("symbol", "").upper() == ticker_upper:
                try:
                    raw_sent = ent.get("sentiment_score")
                    if raw_sent is not None:
                        provider_sentiment = float(raw_sent)
                except (ValueError, TypeError):
                    pass
                try:
                    raw_match = ent.get("match_score")
                    if raw_match is not None:
                        # Normalize match_score to [0.0, 1.0] if needed
                        val = float(raw_match)
                        provider_relevance = min(1.0, val / 100.0) if val > 1.0 else max(0.0, val)
                except (ValueError, TypeError):
                    pass
                break

    return {
        "News_ID": news_id,
        "Ticker": ticker_upper,
        "Company_Name": COMPANY_NAMES.get(ticker_upper, ticker_upper),
        "Published_At_UTC": pub_utc,
        "Headline": title,
        "Summary": summary,
        "Source": source,
        "URL": url,
        "Provider": "MarketAux",
        "Retrieved_At_UTC": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "Provider_Sentiment": round(provider_sentiment, 4),
        "Provider_Relevance": round(provider_relevance, 4),
        "Dataset_Type": "REAL"
    }


def query_marketaux(
    params: Dict[str, Any],
    token: Optional[str] = None,
    timeout: int = 15,
    max_retries: int = 3,
    backoff: float = 2.0
) -> Tuple[Optional[Dict[str, Any]], str, int]:
    """
    Execute rate-limited query against MarketAux news endpoint.
    
    Returns: (parsed_json_dict, status_message, http_status_code)
    """
    api_token = token or get_marketaux_token()
    if not api_token:
        msg = "MARKETAUX_API_TOKEN is not configured."
        return None, msg, 401

    query_params = dict(params)
    query_params["api_token"] = api_token

    encoded_params = urllib.parse.urlencode(query_params)
    url = f"{MARKETAUX_BASE_URL}?{encoded_params}"

    headers = {
        "User-Agent": "AI-Trading-System-MarketAux-POC/1.0",
        "Accept": "application/json"
    }

    req = urllib.request.Request(url, headers=headers, method="GET")

    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status_code = response.getcode()
                raw_bytes = response.read()
                data = json.loads(raw_bytes.decode("utf-8"))
                return data, "OK", status_code
        except urllib.error.HTTPError as http_err:
            status_code = http_err.code
            try:
                err_body = http_err.read().decode("utf-8")
                err_json = json.loads(err_body)
                err_msg = err_json.get("error", {}).get("message", err_body)
            except Exception:
                err_msg = str(http_err)

            if status_code in (401, 403):
                return None, f"MarketAux Auth Error ({status_code}): {err_msg}", status_code
            elif status_code in (402, 429):
                return None, f"MarketAux Quota/Rate Limit ({status_code}): {err_msg}", status_code
            elif status_code >= 500 and attempt < max_retries:
                time.sleep(backoff * attempt)
                continue
            return None, f"MarketAux HTTP Error {status_code}: {err_msg}", status_code
        except urllib.error.URLError as url_err:
            if attempt < max_retries:
                time.sleep(backoff * attempt)
                continue
            return None, f"Network connection failed: {url_err.reason}", 0
        except Exception as exc:
            return None, f"Unexpected error querying MarketAux: {str(exc)}", 0

    return None, "Max retries exceeded without response", 0


def check_text_quality(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze text quality across a list of raw or normalized articles.
    
    Metrics:
    - total_returned
    - title_available
    - description_available
    - usable_text_available (title OR description non-empty)
    - missing_usable_text
    - usable_text_pct
    - unique_sources
    """
    total = len(articles)
    if total == 0:
        return {
            "total_returned": 0,
            "title_available": 0,
            "description_available": 0,
            "usable_text_available": 0,
            "missing_usable_text": 0,
            "usable_text_pct": 0.0,
            "unique_sources": 0,
            "sources_list": []
        }

    title_count = 0
    desc_count = 0
    usable_count = 0
    sources = set()

    for art in articles:
        # Support both raw MarketAux format and normalized schema
        title = (art.get("title") or art.get("Headline") or "").strip()
        desc = (art.get("description") or art.get("snippet") or art.get("Summary") or "").strip()
        source = (art.get("source") or art.get("Source") or "")
        if isinstance(source, dict):
            source_name = source.get("domain") or source.get("name") or "Unknown"
        else:
            source_name = str(source).strip() or "Unknown"
        sources.add(source_name)

        has_title = bool(title)
        has_desc = bool(desc)

        if has_title:
            title_count += 1
        if has_desc:
            desc_count += 1
        if has_title or has_desc:
            usable_count += 1

    missing_usable = total - usable_count
    usable_pct = round((usable_count / total) * 100.0, 2)

    return {
        "total_returned": total,
        "title_available": title_count,
        "description_available": desc_count,
        "usable_text_available": usable_count,
        "missing_usable_text": missing_usable,
        "usable_text_pct": usable_pct,
        "unique_sources": len(sources),
        "sources_list": sorted(list(sources))
    }


def check_duplicates(articles: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Perform multi-level duplicate check on articles using:
    1. Article ID (News_ID or uuid)
    2. URL
    3. Normalized title + publication date
    """
    seen_ids = set()
    seen_urls = set()
    seen_title_dates = set()

    dup_ids = 0
    dup_urls = 0
    dup_title_dates = 0

    for art in articles:
        # 1. ID
        art_id = str(art.get("News_ID") or art.get("uuid") or "").strip()
        if art_id:
            if art_id in seen_ids:
                dup_ids += 1
            else:
                seen_ids.add(art_id)

        # 2. URL
        url = str(art.get("URL") or art.get("url") or "").strip().lower()
        if url:
            if url in seen_urls:
                dup_urls += 1
            else:
                seen_urls.add(url)

        # 3. Normalized Title + Date
        title = (art.get("Headline") or art.get("title") or "").strip().lower()
        pub = str(art.get("Published_At_UTC") or art.get("published_at") or "").strip()
        date_part = pub[:10]  # YYYY-MM-DD
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
    Run normalized MarketAux articles through the existing NLP pipeline in-memory.
    
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
                "title -> Headline",
                "description/snippet -> Summary",
                "published_at -> Published_At_UTC (ISO 8601 UTC)",
                "uuid -> News_ID (SHA-256 deterministic hash)",
                "entities.sentiment_score -> Provider_Sentiment",
                "entities.match_score -> Provider_Relevance"
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
                "title -> Headline",
                "description/snippet -> Summary",
                "published_at -> Published_At_UTC (ISO 8601 UTC)",
                "uuid -> News_ID (SHA-256 deterministic hash)",
                "entities.sentiment_score -> Provider_Sentiment",
                "entities.match_score -> Provider_Relevance"
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
                "title -> Headline",
                "description/snippet -> Summary",
                "published_at -> Published_At_UTC (ISO 8601 UTC)",
                "uuid -> News_ID (SHA-256 deterministic hash)",
                "entities.sentiment_score -> Provider_Sentiment",
                "entities.match_score -> Provider_Relevance"
            ],
            "sample_results": []
        }


def get_mock_sample_articles() -> List[Dict[str, Any]]:
    """
    Generate realistic MarketAux mock articles across 2019-2024 for offline validation.
    Used for unit testing and offline feasibility evaluation when API token is unconfigured.
    """
    return [
        {
            "uuid": "maux-2019-01-03-aapl",
            "title": "Apple Cuts Revenue Guidance Citing China Slowdown and iPhone Weakness",
            "description": "Apple Inc unexpectedly slashed its quarterly revenue forecast, blaming economic weakness in China and lower iPhone upgrade demand.",
            "snippet": "Tim Cook addressed investors warning of lower iPhone revenue in Greater China.",
            "url": "https://www.reuters.com/article/us-apple-revenue-guidance-2019",
            "published_at": "2019-01-03T14:30:00.000000Z",
            "source": "reuters.com",
            "entities": [{"symbol": "AAPL", "name": "Apple Inc.", "match_score": 95.0, "sentiment_score": -0.65}]
        },
        {
            "uuid": "maux-2019-05-01-aapl",
            "title": "Apple Reports Record Service Revenue and Authorizes 75B Share Buyback",
            "description": "Apple beat Wall Street expectations with strong growth in Services and Wearables, announcing an expanded capital return program.",
            "snippet": "Quarterly earnings beat propelled Apple shares higher in post-market trading.",
            "url": "https://www.bloomberg.com/news/articles/2019-05-01/apple-service-growth",
            "published_at": "2019-05-01T20:45:00.000000Z",
            "source": "bloomberg.com",
            "entities": [{"symbol": "AAPL", "name": "Apple Inc.", "match_score": 98.0, "sentiment_score": 0.72}]
        },
        {
            "uuid": "maux-2020-03-14-aapl",
            "title": "Apple Closes All Retail Stores Outside Greater China Amid Pandemic Headwinds",
            "description": "Apple announced the temporary closure of all global retail stores outside of Greater China in response to the COVID-19 pandemic.",
            "snippet": "Precautionary measures taken to protect employees and customer communities.",
            "url": "https://www.cnbc.com/2020/03/14/apple-closes-retail-stores-globally.html",
            "published_at": "2020-03-14T08:15:00.000000Z",
            "source": "cnbc.com",
            "entities": [{"symbol": "AAPL", "name": "Apple Inc.", "match_score": 92.0, "sentiment_score": -0.45}]
        },
        {
            "uuid": "maux-2020-08-19-aapl",
            "title": "Apple Becomes First US Public Company to Reach Two Trillion Dollar Market Valuation",
            "description": "Apple crossed the historic two trillion market cap milestone driven by strong Mac, iPad, and App Store momentum.",
            "snippet": "Shares surged to all-time highs following resilient remote-work demand.",
            "url": "https://www.wsj.com/articles/apple-hits-two-trillion-market-cap-2020",
            "published_at": "2020-08-19T15:20:00.000000Z",
            "source": "wsj.com",
            "entities": [{"symbol": "AAPL", "name": "Apple Inc.", "match_score": 100.0, "sentiment_score": 0.88}]
        },
        {
            "uuid": "maux-2021-04-20-aapl",
            "title": "Apple Unveils M1 iPad Pro, AirTag Tracking Accessories, and Purple iPhone 12",
            "description": "At its Spring Loaded event, Apple debuted the custom silicon M1 chip in iPad Pro and launched AirTag item finders.",
            "snippet": "Hardware expansion showcases vertical silicon integration advantages.",
            "url": "https://www.theverge.com/2021/4/20/apple-spring-loaded-event-recap",
            "published_at": "2021-04-20T18:00:00.000000Z",
            "source": "theverge.com",
            "entities": [{"symbol": "AAPL", "name": "Apple Inc.", "match_score": 90.0, "sentiment_score": 0.55}]
        },
        {
            "uuid": "maux-2021-09-10-aapl",
            "title": "Judge Issues Injunction in Epic Games Antitrust Case Against Apple App Store",
            "description": "Federal judge ruled Apple must allow developers to steer users to outside payment methods, rejecting monopolization claims.",
            "snippet": "Mixed ruling impacts App Store anti-steering provisions while upholding core business model.",
            "url": "https://www.reuters.com/legal/apple-epic-games-antitrust-ruling-2021",
            "published_at": "2021-09-10T17:30:00.000000Z",
            "source": "reuters.com",
            "entities": [{"symbol": "AAPL", "name": "Apple Inc.", "match_score": 96.0, "sentiment_score": -0.30}]
        },
        {
            "uuid": "maux-2022-01-03-aapl",
            "title": "Apple Briefly Crosses Three Trillion Dollar Threshold on iPhone Demand Optimism",
            "description": "Apple briefly touched the historic three trillion valuation mark during intraday trading on investor optimism over future growth drivers.",
            "snippet": "Milestone reflects market confidence in hardware ecosystem and software services cash flows.",
            "url": "https://www.ft.com/content/apple-three-trillion-market-cap-2022",
            "published_at": "2022-01-03T19:10:00.000000Z",
            "source": "ft.com",
            "entities": [{"symbol": "AAPL", "name": "Apple Inc.", "match_score": 98.0, "sentiment_score": 0.80}]
        },
        {
            "uuid": "maux-2022-11-06-aapl",
            "title": "Apple Warns of Lower iPhone 14 Pro Shipments Due to Supply Chain Bottlenecks in Zhengzhou",
            "description": "Apple issued a rare production warning indicating COVID-19 restrictions at its primary Zhengzhou assembly plant will delay holiday shipments.",
            "snippet": "Higher-tier iPhone 14 Pro and Pro Max supply severely constrained ahead of holiday quarter.",
            "url": "https://www.bloomberg.com/news/articles/2022-11-06/apple-iphone-shipment-warning",
            "published_at": "2022-11-06T23:00:00.000000Z",
            "source": "bloomberg.com",
            "entities": [{"symbol": "AAPL", "name": "Apple Inc.", "match_score": 94.0, "sentiment_score": -0.58}]
        },
        {
            "uuid": "maux-2023-06-05-aapl",
            "title": "Apple Introduces Vision Pro Spatial Computing Headset at WWDC 2023",
            "description": "Apple unveiled the Vision Pro mixed reality headset, marking its first major new product category entrance since the Apple Watch in 2015.",
            "snippet": "Spatial computing platform combines eye tracking, gesture control, and visionOS.",
            "url": "https://www.theverge.com/2023/6/5/apple-vision-pro-spatial-computing-wwdc",
            "published_at": "2023-06-05T18:45:00.000000Z",
            "source": "theverge.com",
            "entities": [{"symbol": "AAPL", "name": "Apple Inc.", "match_score": 95.0, "sentiment_score": 0.60}]
        },
        {
            "uuid": "maux-2023-11-02-aapl",
            "title": "Apple Revenue Declines for Fourth Straight Quarter Despite Record Services Growth",
            "description": "Apple recorded a fourth consecutive quarter of year-over-year revenue contraction, though gross margins expanded to 45.2%.",
            "snippet": "Hardware headwinds in China and PC market softness offset by record active installed device base.",
            "url": "https://www.wsj.com/articles/apple-q4-earnings-revenue-decline-2023",
            "published_at": "2023-11-02T20:30:00.000000Z",
            "source": "wsj.com",
            "entities": [{"symbol": "AAPL", "name": "Apple Inc.", "match_score": 96.0, "sentiment_score": -0.22}]
        },
        {
            "uuid": "maux-2024-06-10-aapl",
            "title": "Apple Announces Apple Intelligence Generative AI System and OpenAI Partnership at WWDC",
            "description": "Apple unveiled Apple Intelligence, integrating on-device and private cloud generative models across iOS 18, iPadOS 18, and macOS Sequoia.",
            "snippet": "Siri overhaul with deep context awareness and ChatGPT integration highlights AI strategy.",
            "url": "https://www.cnbc.com/2024/06/10/apple-wwdc-2024-apple-intelligence-ai.html",
            "published_at": "2024-06-10T17:15:00.000000Z",
            "source": "cnbc.com",
            "entities": [{"symbol": "AAPL", "name": "Apple Inc.", "match_score": 100.0, "sentiment_score": 0.85}]
        },
        {
            "uuid": "maux-2024-10-31-aapl",
            "title": "Apple Reports Strong Fourth Quarter Revenue of 94.9 Billion Driven by iPhone 16",
            "description": "Apple beat analyst forecasts with 94.9 billion in quarterly revenue, an increase of 6 percent year over year, while Services set an all-time revenue record.",
            "snippet": "iPhone 16 customer demand showed positive early momentum alongside record Services expansion.",
            "url": "https://www.reuters.com/technology/apple-q4-earnings-beat-forecasts-2024",
            "published_at": "2024-10-31T20:30:00.000000Z",
            "source": "reuters.com",
            "entities": [{"symbol": "AAPL", "name": "Apple Inc.", "match_score": 98.0, "sentiment_score": 0.70}]
        }
    ]
