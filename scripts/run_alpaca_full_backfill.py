"""
CLI Runner for Full Alpaca Historical News Backfill into Isolated Staging.

Executes the authenticated download and NLP enrichment of historical financial news
across 10 target companies (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, V, JNJ)
from 2019-01-01 through 2024-12-31.

ISOLATION GUARANTEES:
- Staged strictly in data/live/alpaca_staging/
- Production database (data/live/news_database.db) is untouched.
- Decision Fusion, Dashboard V2, notebooks, and academic files are untouched.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.alpaca_backfill_manager import execute_full_staging_pipeline


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Full Alpaca Historical News Staging Backfill.")
    parser.add_argument("--start", type=str, default="2019-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--rate-delay", type=float, default=0.40, help="Sustained inter-request delay in seconds (default: 0.40)")
    parser.add_argument("--force-refresh", action="store_true", help="Force re-download of existing checkpointed segments")

    args = parser.parse_args()

    result = execute_full_staging_pipeline(
        start_date=args.start,
        end_date=args.end,
        rate_delay=args.rate_delay,
        force_refresh=args.force_refresh
    )

    if result.get("status") == "SUCCESS":
        print("\nBackfill pipeline completed successfully!")
        print(f"Report: {result['output_files']['report_md']}")
        sys.exit(0)
    else:
        print("\nBackfill pipeline finished with errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
