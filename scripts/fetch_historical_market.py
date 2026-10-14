"""
Historical Market Data Extension CLI Script.

Fetches post-2024 daily market data (from 2024-12-31 forward) using yfinance.
Validates OHLC values, deduplicates, and stores into data/live/market/historical_market_extension.csv.

IMPORTANT: Does NOT modify data/raw/multi_company_stock_data.csv.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.realtime_market import fetch_historical_market_extension, SUPPORTED_TICKERS

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Fetch historical daily market extension using yfinance.")
    parser.add_argument("--start", type=str, default="2024-12-31", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD, defaults to today).")
    parser.add_argument("--tickers", nargs="+", default=SUPPORTED_TICKERS, help="List of tickers to fetch.")
    args = parser.parse_args()

    print("=" * 65)
    print("HISTORICAL MARKET DATA EXTENSION (YFINANCE)")
    print("=" * 65)
    print(f"Target Universe ({len(args.tickers)} tickers): {args.tickers}")
    print(f"Date Range: {args.start} to {args.end or 'Current Date'}")

    df_ext = fetch_historical_market_extension(
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end
    )

    if not df_ext.empty:
        print(f"\n[OK] Fetched and validated {len(df_ext)} total extension rows.")
        print(f"Saved to: data/live/market/historical_market_extension.csv")
        print("\nBreakdown by Ticker:")
        print(df_ext["Ticker"].value_counts().to_string())
    else:
        print("\n[INFO] No extension rows returned (or market data already current).")

    print("\n" + "=" * 65)

if __name__ == "__main__":
    main()
