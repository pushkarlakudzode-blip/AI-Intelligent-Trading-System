"""
Controlled Real-Data Manual Refresh Manager.

Coordinates on-demand, user-triggered refresh of:
1. Real Market Data (via yfinance snapshots / daily bars)
2. Latest Financial News (via Alpha Vantage NEWS_SENTIMENT)
3. Internal Loughran-McDonald NLP Pipeline Enrichment
4. Real-Data SQLite & CSV Storage
5. Real-Data Analytics Pipeline Rebuild
6. Audit Logging to data/live/analytics/refresh_log.csv

Strict Safety Rules:
- Manual execution only: Never runs automatically at import or page load.
- Quota Safety: Detects rate limits / daily quota exhaustion and aborts further calls.
- Non-interference: Does NOT touch academic datasets (Assignments 1–8) or Decision Fusion.
"""

import logging
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from src.analytics.pipeline import run_real_data_analytics
from src.data.data_storage import SUPPORTED_TICKERS
from src.data.realtime_market import refresh_market_data
from src.data.realtime_news import fetch_latest_news_for_ticker

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LIVE_ANALYTICS_DIR = PROJECT_ROOT / "data" / "live" / "analytics"
REFRESH_LOG_PATH = LIVE_ANALYTICS_DIR / "refresh_log.csv"

_ACTIVE_REFRESH_LOG_PATH = REFRESH_LOG_PATH

# Minimum cooldown per ticker to prevent rapid spamming (seconds)
NEWS_REFRESH_COOLDOWN_SECONDS = 300  # 5 minutes

# Ticker-level timestamp tracking for cooldown enforcement
_TICKER_LAST_SUCCESSFUL_REFRESH: Dict[str, datetime] = {}


def record_ticker_refresh(ticker: str, ts: Optional[datetime] = None) -> None:
    """Record timestamp of a successful news refresh for a ticker."""
    _TICKER_LAST_SUCCESSFUL_REFRESH[str(ticker).strip().upper()] = ts or datetime.now(timezone.utc)


def get_ticker_last_refresh(ticker: str) -> Optional[datetime]:
    """Get timestamp of last successful news refresh for a ticker."""
    return _TICKER_LAST_SUCCESSFUL_REFRESH.get(str(ticker).strip().upper())


def reset_cooldown_tracking() -> None:
    """Reset in-memory cooldown timestamps."""
    _TICKER_LAST_SUCCESSFUL_REFRESH.clear()


def get_deterministic_tickers(
    tickers: Optional[Union[List[str], Set[str], Tuple[str, ...]]] = None
) -> List[str]:
    """
    Return a deterministic, sliceable, ordered list of tickers.
    - If tickers is provided: preserves caller's explicit order, filters empty/blank values,
      and removes duplicates while keeping first occurrence.
    - If tickers is None: returns sorted list of SUPPORTED_TICKERS.
    - Does not mutate global SUPPORTED_TICKERS.
    """
    if tickers is not None:
        cleaned: List[str] = []
        for t in tickers:
            t_str = str(t).strip().upper()
            if t_str and t_str not in cleaned:
                cleaned.append(t_str)
        return cleaned
    return sorted(list(SUPPORTED_TICKERS))


def is_quota_or_rate_limit(status_str: str) -> bool:
    """Detect whether status string indicates provider rate limit or quota exhaustion."""
    if not status_str:
        return False
    lower = status_str.lower()
    keywords = [
        "rate limit", "quota", "25 requests", "exceeded",
        "note", "call frequency", "calls per minute", "maximum of 25"
    ]
    return any(kw in lower for kw in keywords)


def configure_refresh_manager(log_path: Optional[Union[str, Path]] = None) -> None:
    """Configure or reset the active refresh log path for testing or custom output."""
    global _ACTIVE_REFRESH_LOG_PATH
    if log_path is not None:
        _ACTIVE_REFRESH_LOG_PATH = Path(log_path)
    else:
        _ACTIVE_REFRESH_LOG_PATH = REFRESH_LOG_PATH
    reset_cooldown_tracking()


def log_refresh_event(
    log_entry: Dict[str, Union[str, bool, int]],
    log_path: Optional[Union[str, Path]] = None
) -> None:
    """Append a structured audit row to refresh_log.csv."""
    target_path = Path(log_path) if log_path else _ACTIVE_REFRESH_LOG_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    expected_cols = [
        "timestamp_utc",
        "market_requested",
        "news_requested",
        "market_success",
        "news_success",
        "articles_added",
        "articles_duplicate",
        "articles_rejected",
        "quota_limited",
        "analytics_rebuilt",
        "status",
        "error_message"
    ]
    
    row_data = {col: log_entry.get(col, "") for col in expected_cols}
    df_row = pd.DataFrame([row_data])
    
    if not target_path.exists():
        df_row.to_csv(target_path, index=False)
    else:
        df_row.to_csv(target_path, mode="a", header=False, index=False)
    logger.info(f"Logged refresh event to {target_path} with status: {log_entry.get('status')}")


def get_last_refresh_info(log_path: Optional[Union[str, Path]] = None) -> Dict[str, Union[str, bool, int]]:
    """Retrieve the most recent refresh record from the refresh log."""
    target_path = Path(log_path) if log_path else _ACTIVE_REFRESH_LOG_PATH
    if not target_path.exists():
        return {
            "timestamp_utc": "Never",
            "status": "READY",
            "market_success": False,
            "news_success": False,
            "quota_limited": False,
            "summary": "No manual refresh has been performed yet."
        }
        
    try:
        df = pd.read_csv(target_path)
        if df.empty:
            return {
                "timestamp_utc": "Never",
                "status": "READY",
                "market_success": False,
                "news_success": False,
                "quota_limited": False,
                "summary": "No manual refresh records found."
            }
        last_row = df.iloc[-1]
        return {
            "timestamp_utc": str(last_row.get("timestamp_utc", "Never")),
            "status": str(last_row.get("status", "READY")),
            "market_success": bool(last_row.get("market_success", False)),
            "news_success": bool(last_row.get("news_success", False)),
            "quota_limited": bool(last_row.get("quota_limited", False)),
            "articles_added": int(last_row.get("articles_added", 0)),
            "articles_duplicate": int(last_row.get("articles_duplicate", 0)),
            "summary": f"Last Status: {last_row.get('status')} | Added: {last_row.get('articles_added')} arts."
        }
    except Exception as e:
        logger.warning(f"Error reading refresh log: {e}")
        return {
            "timestamp_utc": "Unknown",
            "status": "READY",
            "market_success": False,
            "news_success": False,
            "quota_limited": False,
            "summary": "Could not parse refresh log."
        }


def execute_real_data_refresh(
    scope: str = "Market + News",
    tickers: Optional[Union[List[str], Set[str], Tuple[str, ...]]] = None,
    limit_per_ticker: int = 15,
    rebuild_analytics: bool = True,
    log_path: Optional[Union[str, Path]] = None,
    market_save_path: Optional[Union[str, Path]] = None,
    analytics_output_dir: Optional[Union[str, Path]] = None,
    cooldown_seconds: int = 0,
    request_delay_seconds: float = 0.0
) -> Dict[str, Union[str, bool, int, Dict]]:
    """
    Execute controlled manual real data refresh.
    
    Args:
        scope: "Market only", "News only", or "Market + News"
        tickers: Target equities (defaults to sorted project tickers)
        limit_per_ticker: Maximum articles requested per ticker
        rebuild_analytics: If True, re-runs real-data analytics pipeline
        log_path: Optional override for audit log path
        market_save_path: Optional override for market CSV path
        analytics_output_dir: Optional override for analytics output directory
        cooldown_seconds: Minimum cooldown seconds before refreshing same ticker (0 disables)
        request_delay_seconds: Optional pacing delay in seconds between ticker queries
        
    Returns:
        Dict: Full execution telemetry and status summary.
    """
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    target_tickers = get_deterministic_tickers(tickers)
    
    market_requested = scope in ["Market only", "Market + News"]
    news_requested = scope in ["News only", "Market + News"]
    
    market_success = False
    news_success = False
    quota_limited = False
    analytics_rebuilt = False
    
    articles_added = 0
    articles_duplicate = 0
    articles_rejected = 0
    news_refreshed_count = 0
    
    error_messages = []
    market_details = {}
    news_details = {}
    
    logger.info(f"Starting controlled manual refresh. Scope: {scope}, Tickers: {len(target_tickers)}")
    
    # 1. MARKET DATA REFRESH
    if market_requested:
        try:
            m_res = refresh_market_data(tickers=target_tickers, save_path=market_save_path)
            market_success = m_res.get("success", False)
            market_details = m_res.get("details", {})
            m_count = m_res.get("refreshed_count", 0)
            logger.info(f"Market refresh completed: {m_count}/{len(target_tickers)} succeeded.")
            if m_count < len(target_tickers):
                error_messages.append(f"Market refresh succeeded for {m_count}/{len(target_tickers)} tickers.")
        except Exception as e:
            logger.error(f"Market refresh exception: {e}")
            market_success = False
            error_messages.append(f"Market refresh failed: {e}")

    # 2. NEWS DATA REFRESH (With Quota Protection & Pacing)
    if news_requested:
        try:
            for idx, t in enumerate(target_tickers):
                # Pacing delay between requests (if configured and idx > 0)
                if idx > 0 and request_delay_seconds > 0:
                    time.sleep(request_delay_seconds)

                # Cooldown enforcement
                if cooldown_seconds > 0:
                    last_ref = get_ticker_last_refresh(t)
                    if last_ref is not None:
                        elapsed = (datetime.now(timezone.utc) - last_ref).total_seconds()
                        if elapsed < cooldown_seconds:
                            logger.info(f"Ticker {t} skipped due to cooldown ({elapsed:.1f}s < {cooldown_seconds}s).")
                            news_details[t] = {
                                "ticker": t,
                                "status": "SKIPPED_DUE_TO_COOLDOWN",
                                "raw_fetched": 0, "added": 0, "duplicates": 0, "rejected": 0
                            }
                            continue

                # Fetch latest news with per-ticker exception isolation
                try:
                    n_res = fetch_latest_news_for_ticker(t, limit=limit_per_ticker, run_nlp=True)
                except Exception as e_fetch:
                    logger.error(f"Error fetching news for {t}: {e_fetch}")
                    news_details[t] = {
                        "ticker": t,
                        "status": f"ERROR: {e_fetch}",
                        "raw_fetched": 0, "added": 0, "duplicates": 0, "rejected": 0
                    }
                    error_messages.append(f"News fetch failed on {t}: {e_fetch}")
                    continue

                status_str = n_res.get("status", "")
                news_details[t] = n_res
                
                # Check for rate limit or quota exhaustion
                if is_quota_or_rate_limit(status_str):
                    quota_limited = True
                    logger.warning(f"Alpha Vantage quota limit reached while refreshing {t}: {status_str}")
                    error_messages.append(f"Alpha Vantage quota limit reached on {t}.")
                    # Abort further ticker requests to avoid burning API quota
                    for remaining_t in target_tickers[idx+1:]:
                        news_details[remaining_t] = {
                            "ticker": remaining_t,
                            "status": "SKIPPED_DUE_TO_QUOTA_LIMIT",
                            "raw_fetched": 0, "added": 0, "duplicates": 0, "rejected": 0
                        }
                    break
                elif status_str == "SUCCESS":
                    news_refreshed_count += 1
                    articles_added += n_res.get("added", 0)
                    articles_duplicate += n_res.get("duplicates", 0)
                    articles_rejected += n_res.get("rejected", 0)
                    record_ticker_refresh(t)
                else:
                    logger.warning(f"News refresh non-success for {t}: {status_str}")
                    error_messages.append(f"News refresh failed on {t}: {status_str}")

            if not quota_limited and news_refreshed_count == len(target_tickers):
                news_success = True
            elif news_refreshed_count > 0:
                news_success = False  # Partial
            else:
                news_success = False
        except Exception as e:
            logger.error(f"News refresh exception: {e}")
            news_success = False
            error_messages.append(f"News refresh failed: {e}")

    # 3. REBUILD REAL-DATA ANALYTICS
    if rebuild_analytics:
        try:
            logger.info("Rebuilding Real-Data Analytics artifacts...")
            run_real_data_analytics(output_dir=analytics_output_dir, save_outputs=True)
            analytics_rebuilt = True
        except Exception as e:
            logger.error(f"Analytics rebuild failed: {e}")
            analytics_rebuilt = False
            error_messages.append(f"Analytics rebuild failed: {e}")

    # 4. DETERMINE OVERALL STATUS
    # Rules:
    # SUCCESS: all requested components completely succeeded without errors or quota limits.
    # PARTIAL: at least one component succeeded/partially succeeded, but another failed or hit quota.
    # FAILED: all requested components failed.
    if market_requested and news_requested:
        if market_success and news_success and not quota_limited:
            overall_status = "SUCCESS"
        elif (market_success or news_refreshed_count > 0 or articles_added > 0) and (quota_limited or not news_success or not market_success):
            overall_status = "PARTIAL"
        else:
            overall_status = "FAILED"
    elif market_requested and not news_requested:
        if market_success:
            overall_status = "SUCCESS"
        elif any(v == "SUCCESS" for v in market_details.values()):
            overall_status = "PARTIAL"
        else:
            overall_status = "FAILED"
    elif news_requested and not market_requested:
        if news_success and not quota_limited:
            overall_status = "SUCCESS"
        elif news_refreshed_count > 0 or articles_added > 0:
            overall_status = "PARTIAL"
        else:
            overall_status = "FAILED"
    else:
        overall_status = "SKIPPED"

    err_str = "; ".join(error_messages) if error_messages else ""

    telemetry = {
        "timestamp_utc": now_utc,
        "market_requested": market_requested,
        "news_requested": news_requested,
        "market_success": market_success,
        "news_success": news_success,
        "articles_added": articles_added,
        "articles_duplicate": articles_duplicate,
        "articles_rejected": articles_rejected,
        "quota_limited": quota_limited,
        "analytics_rebuilt": analytics_rebuilt,
        "status": overall_status,
        "error_message": err_str,
        "market_refreshed_count": sum(1 for v in market_details.values() if v == "SUCCESS"),
        "news_refreshed_count": news_refreshed_count,
        "total_tickers": len(target_tickers),
        "market_details": market_details,
        "news_details": news_details
    }

    # 5. LOG EVENT
    log_refresh_event(telemetry, log_path=log_path)

    return telemetry
