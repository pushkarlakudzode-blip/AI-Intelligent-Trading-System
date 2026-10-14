"""
Fetch Latest Financial News CLI Script.

Requests latest articles from Alpha Vantage, normalizes timestamps and schema,
enforces duplicate prevention, stores in SQLite/CSV, executes the NLP enrichment pipeline,
and reports ingestion metrics.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.historical_news import get_api_key, SUPPORTED_TICKERS
from src.data.realtime_news import fetch_latest_news_for_ticker, fetch_latest_news_all
from src.data.data_storage import get_stored_articles, get_stored_nlp_results

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Fetch latest financial news via Alpha Vantage.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Target ticker (e.g. AAPL, NVDA, TSLA).")
    parser.add_argument("--all", action="store_true", help="Fetch latest news for all 10 project tickers sequentially.")
    parser.add_argument("--limit", type=int, default=25, help="Maximum articles to request per ticker.")
    args = parser.parse_args()

    print("=" * 65)
    print("LIVE FINANCIAL NEWS INGESTION (ALPHA VANTAGE)")
    print("=" * 65)

    api_key = get_api_key()
    if not api_key:
        print("\n[WARNING] ALPHA_VANTAGE_API_KEY environment variable is not configured.")
        print("Please obtain a free API key at: https://www.alphavantage.co/support/#api-key")
        print("Add it to .env or export ALPHA_VANTAGE_API_KEY='your_key'")
        print("\nSkipping live HTTP call safely without error.")
        return

    if args.all:
        print(f"\n[INFO] Initiating sequential fetch for all 10 tickers (limit={args.limit})...")
        results = fetch_latest_news_all(limit_per_ticker=args.limit)
        print("\nIngestion Summary Across Universe:")
        total_added = 0
        total_dups = 0
        for t, res in results.items():
            print(f"  {t:5s} | Status: {res['status']:12s} | Fetched: {res['raw_fetched']:2d} | Added: {res['added']:2d} | Skipped Dups: {res['duplicates']:2d}")
            total_added += res['added']
            total_dups += res['duplicates']
        print(f"\nTotal New Articles Added: {total_added} | Total Duplicates Skipped: {total_dups}")
    else:
        ticker = args.ticker.strip().upper()
        if ticker not in SUPPORTED_TICKERS:
            print(f"[ERROR] Unsupported ticker '{ticker}'. Supported: {SUPPORTED_TICKERS}")
            sys.exit(1)

        print(f"\n[INFO] Fetching latest news for {ticker} (limit={args.limit})...")
        res = fetch_latest_news_for_ticker(ticker, limit=args.limit)
        print(f"\nResult for {ticker}:")
        print(f"  Status:             {res['status']}")
        print(f"  Raw Fetched:        {res['raw_fetched']}")
        print(f"  New Articles Added: {res['added']}")
        print(f"  Duplicates Skipped: {res['duplicates']}")
        print(f"  Rejected (Quality): {res['rejected']}")

    # Report total database state
    stored_df = get_stored_articles()
    nlp_df = get_stored_nlp_results()
    print(f"\nDatabase State:")
    print(f"  Total Articles in DB:    {len(stored_df)}")
    print(f"  Total NLP Records in DB: {len(nlp_df)}")
    print("=" * 65)

if __name__ == "__main__":
    main()
