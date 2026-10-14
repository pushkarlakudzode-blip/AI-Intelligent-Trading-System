"""
Real-Data Analytics Layer Package.

Provides standalone, reusable analytics modules for the live/real-data layer:
- market_analytics: Technical indicators and volatility estimators
- news_analytics: NLP sentiment, relevance, importance, triage aggregation
- coverage_analytics: January 2025 completeness and provider quota audits
- pipeline: End-to-end execution and persistent artifact generation
"""

from src.analytics.coverage_analytics import compute_real_data_coverage
from src.analytics.market_analytics import (
    compute_real_market_summary,
    compute_ticker_market_analytics,
    load_real_market_data,
)
from src.analytics.news_analytics import (
    compute_real_news_daily_summary,
    compute_real_news_summary,
    load_real_daily_signals,
    load_real_nlp_results,
)
from src.analytics.pipeline import run_real_data_analytics
from src.analytics.refresh_manager import (
    configure_refresh_manager,
    execute_real_data_refresh,
    get_last_refresh_info,
    log_refresh_event,
)

__all__ = [
    "load_real_market_data",
    "compute_ticker_market_analytics",
    "compute_real_market_summary",
    "load_real_nlp_results",
    "load_real_daily_signals",
    "compute_real_news_summary",
    "compute_real_news_daily_summary",
    "compute_real_data_coverage",
    "run_real_data_analytics",
    "execute_real_data_refresh",
    "get_last_refresh_info",
    "log_refresh_event",
    "configure_refresh_manager",
]
