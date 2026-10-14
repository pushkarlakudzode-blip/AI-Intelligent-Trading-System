"""
CLI Runner for Alpaca Real-Time Financial News Stream Listener.

Allows running the persistent background listener from the terminal or as an
independent background daemon process.

Features:
- Subscribes to all 10 supported tickers (or explicit list).
- Provides optional staging mode (--staging) to ingest into an isolated
  staging directory for non-destructive verification.
- Supports optional bounded execution (--duration <sec>) for automated tests.
- Gracefully handles SIGINT (Ctrl-C) and terminates cleanly.
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.alpaca_stream_listener import (
    AlpacaStreamListener,
    SUPPORTED_TICKERS,
    DEFAULT_STATUS_PATH,
    get_stream_status
)
from src.data.data_storage import configure_storage, init_storage

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("run_alpaca_stream")


def parse_args():
    parser = argparse.ArgumentParser(description="Alpaca Real-Time News Stream Runner")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=SUPPORTED_TICKERS,
        help="List of ticker symbols to track (default: all 10 project tickers)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional duration in seconds before auto-stopping (default: run indefinitely)"
    )
    parser.add_argument(
        "--staging",
        action="store_true",
        help="Route storage to isolated staging folder (data/live/alpaca_staging/stream_test/)"
    )
    parser.add_argument(
        "--status-path",
        type=str,
        default=str(DEFAULT_STATUS_PATH),
        help="Custom path for live status JSON export"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Initializing Alpaca Stream Runner for %d tickers...", len(args.tickers))

    if args.staging:
        staging_dir = PROJECT_ROOT / "data" / "live" / "alpaca_staging" / "stream_test"
        staging_dir.mkdir(parents=True, exist_ok=True)
        configure_storage(staging_dir)
        init_storage()
        logger.info("STAGING MODE: Storage routed to %s", staging_dir)
    else:
        logger.info("PRODUCTION MODE: Storage routed to standard live storage.")

    db_path = (staging_dir / "news_database.db") if args.staging else None
    status_path = Path(args.status_path)

    listener = AlpacaStreamListener(
        tickers=args.tickers,
        status_path=status_path,
        db_path=db_path
    )

    def handle_exit(sig, frame):
        logger.info("Termination signal received. Shutting down listener...")
        listener.stop(timeout=5.0)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    listener.start()
    logger.info("Listener started. Streaming active for tickers: %s", ", ".join(args.tickers))

    start_time = time.time()
    try:
        while listener.is_alive():
            time.sleep(1.0)
            status = listener.get_status()
            
            # Print periodic heartbeat if articles arrive
            if status.get("last_article"):
                art = status["last_article"]
                logger.info("LATEST STREAM ARTICLE: [%s] %s (Sent: %s, Score: %+.2f)",
                            art["ticker"], art["headline"][:60], art["sentiment_label"], art["sentiment_score"])

            if args.duration is not None and (time.time() - start_time) >= args.duration:
                logger.info("Specified duration (%.1fs) reached. Stopping listener...", args.duration)
                break
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop(timeout=5.0)
        final_status = listener.get_status()
        logger.info("Final Session Telemetry: Received=%d, Stored=%d, Duplicates=%d",
                    final_status["articles_received_count"],
                    final_status["articles_stored_count"],
                    final_status["articles_duplicate_count"])
        if args.staging:
            configure_storage(None)


if __name__ == "__main__":
    main()
