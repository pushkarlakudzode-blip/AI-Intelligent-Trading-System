"""
Historical Financial News Ingestion CLI Script.

Fetches historical financial news over a user-defined date range from Alpha Vantage,
normalizes publication timestamps, enforces deduplication, executes the NLP pipeline,
and saves records into the SQLite database and CSV files.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.historical_news import get_api_key, fetch_historical_news, SUPPORTED_TICKERS
from src.data.data_storage import get_stored_articles, get_stored_nlp_results

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Fetch historical financial news from Alpha Vantage.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Target ticker (e.g. AAPL, NVDA, TSLA).")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date (YYYY-MM-DD).")
    parser.add_argument("--limit", type=int, default=100, help="Maximum articles to request per query.")
    parser.add_argument("--all", action="store_true", help="Fetch historical news for all 10 project tickers.")
    parser.add_argument("--delay", type=float, default=12.5, help="Delay in seconds between ticker queries for rate-limit protection.")
    parser.add_argument("--windows", type=str, default=None, help="Explicit semicolon-separated sub-windows, e.g. '2025-01-01:2025-01-07;2025-01-08:2025-01-14'")
    args = parser.parse_args()

    print("=" * 65)
    print("HISTORICAL FINANCIAL NEWS INGESTION (ALPHA VANTAGE)")
    print("=" * 65)

    api_key = get_api_key()
    if not api_key:
        print("\n[WARNING] ALPHA_VANTAGE_API_KEY environment variable is not configured.")
        print("Please obtain a free API key at: https://www.alphavantage.co/support/#api-key")
        print("Add it to .env or export ALPHA_VANTAGE_API_KEY='your_key'")
        print("\nSkipping live HTTP call safely without error.")
        return

    if args.all:
        targets = SUPPORTED_TICKERS
    elif "," in args.ticker:
        targets = [t.strip().upper() for t in args.ticker.split(",") if t.strip()]
    else:
        targets = [args.ticker.strip().upper()]

    parsed_windows = None
    if args.windows:
        parsed_windows = []
        for w in args.windows.split(";"):
            if ":" in w:
                s, e = w.split(":")
                parsed_windows.append((s.strip(), e.strip()))
        print(f"Executing with {len(parsed_windows)} configured sub-windows:")
        for sw, ew in parsed_windows:
            print(f"  - {sw} to {ew}")

    all_window_records = []

    for idx, ticker in enumerate(targets):
        if ticker not in SUPPORTED_TICKERS:
            print(f"[ERROR] Unsupported ticker '{ticker}'. Supported: {SUPPORTED_TICKERS}")
            continue

        if parsed_windows:
            print(f"\n[{idx+1}/{len(targets)}] Fetching windowed historical news for {ticker} across {len(parsed_windows)} windows...")
            from src.data.historical_news import fetch_historical_news_windowed
            w_results = fetch_historical_news_windowed(
                ticker=ticker,
                windows=parsed_windows,
                limit_per_window=args.limit,
                inter_window_delay=args.delay,
                run_nlp=True
            )
            all_window_records.extend(w_results)
            print(f"\n  Sub-Window Results for {ticker}:")
            for wr in w_results:
                capped_str = "YES (Capped)" if wr['limit_reached'] else "No"
                print(f"    [{wr['window_start']} -> {wr['window_end']}] Raw: {wr['raw_fetched']}, Added: {wr['new_added']}, Dups: {wr['duplicates']}, Rej: {wr['rejected']}, NLP: {wr['nlp_records']}, Capped: {capped_str}")
                print(f"      Actual Range: {wr['actual_earliest']} to {wr['actual_latest']}")
        else:
            print(f"\n[{idx+1}/{len(targets)}] Fetching historical news for {ticker} ({args.start} to {args.end})...")
            res = fetch_historical_news(
                ticker=ticker,
                start_date=args.start,
                end_date=args.end,
                limit=args.limit,
                run_nlp=True
            )
            print(f"  Status:             {res['status']}")
            print(f"  Raw Fetched:        {res['raw_fetched']}")
            print(f"  New Articles Added: {res['added']}")
            print(f"  Duplicates Skipped: {res['duplicates']}")
            print(f"  Rejected:           {res['rejected']}")

        if idx < len(targets) - 1 and not parsed_windows:
            print(f"  Rate-limit pacing: sleeping {args.delay:.1f}s...")
            time.sleep(args.delay)

    # Report total database state
    stored_df = get_stored_articles()
    nlp_df = get_stored_nlp_results()
    print(f"\nDatabase State:")
    print(f"  Total Articles in DB:    {len(stored_df)}")
    print(f"  Total NLP Records in DB: {len(nlp_df)}")
    print("=" * 65)

if __name__ == "__main__":
    main()
