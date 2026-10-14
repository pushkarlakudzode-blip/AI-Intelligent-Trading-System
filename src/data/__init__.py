"""
Data Ingestion Package for AI-Based Intelligent Trading System.

Includes historical market ingestion, real-time snapshot/intraday polling,
Alpha Vantage news collection, deterministic deduplication, and SQLite/CSV storage.
"""

from src.data.market_data import validate_market_data
from src.data.realtime_market import (
    get_current_quote,
    get_latest_market_data,
    get_intraday_market_data,
    stream_market_data,
    fetch_historical_market_extension
)
from src.data.historical_news import (
    fetch_historical_news,
    fetch_historical_news_windowed,
    query_alpha_vantage_news
)
from src.data.realtime_news import fetch_latest_news_for_ticker, fetch_latest_news_all
from src.data.data_storage import (
    generate_news_id,
    store_articles,
    store_nlp_results,
    get_stored_articles,
    get_stored_nlp_results,
    init_storage,
    configure_storage
)

__all__ = [
    "validate_market_data",
    "get_current_quote",
    "get_latest_market_data",
    "get_intraday_market_data",
    "stream_market_data",
    "fetch_historical_market_extension",
    "fetch_historical_news",
    "fetch_historical_news_windowed",
    "query_alpha_vantage_news",
    "fetch_latest_news_for_ticker",
    "fetch_latest_news_all",
    "generate_news_id",
    "store_articles",
    "store_nlp_results",
    "get_stored_articles",
    "get_stored_nlp_results",
    "init_storage",
    "configure_storage"
]
