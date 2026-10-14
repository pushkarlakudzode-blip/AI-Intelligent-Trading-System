"""
Unit & Integration Regression Test Suite for Alpaca Real-Time News Stream Listener.

Verifies:
1. Credential detection and redaction privacy.
2. Normalization of incoming Benzinga/Alpaca news payloads.
3. Article quality and lookahead guard validation.
4. Multi-ticker symbol handling and FB -> META aliasing.
5. Persistence, deduplication, and SQLite schema alignment.
6. Real-time Loughran-McDonald NLP scoring on streamed articles.
7. Mocked WebSocket lifecycle (connect -> auth -> subscribe -> stream -> close).
8. Transient disconnection and backoff reconnection logic.
9. Status export and telemetry consumption for Real-Time Data dashboard.
10. Strict production database preservation (zero test mutations to news_database.db).
"""

import json
import os
import sqlite3
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.alpaca_stream_listener import (
    AlpacaStreamListener,
    get_alpaca_credentials,
    sanitize_text,
    clean_html,
    parse_stream_timestamp,
    normalize_stream_article,
    get_stream_status,
    SUPPORTED_TICKERS,
    TICKER_ALIASES
)
from src.data.data_storage import (
    configure_storage,
    init_storage,
    store_articles,
    store_nlp_results,
    get_existing_news_ids,
    validate_article,
    generate_news_id
)
from src.nlp.news_nlp_pipeline import process_articles_nlp

PROD_DB_PATH = PROJECT_ROOT / "data" / "live" / "news_database.db"


class TestAlpacaStreamListener(unittest.TestCase):

    def setUp(self):
        # Record production database baseline counts
        if PROD_DB_PATH.exists():
            conn = sqlite3.connect(f"file:{PROD_DB_PATH}?mode=ro", uri=True)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM articles")
            self.baseline_articles = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM nlp_results")
            self.baseline_nlp = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM validation_logs")
            self.baseline_val = cur.fetchone()[0]
            conn.close()
        else:
            self.baseline_articles = 0
            self.baseline_nlp = 0
            self.baseline_val = 0

        # Create isolated temporary directory for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.test_status_path = self.temp_path / "test_stream_status.json"

        # Point storage to isolated temporary directory
        configure_storage(self.temp_path)
        init_storage()

    def tearDown(self):
        # Reset storage to default production configuration
        configure_storage(None)
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

        # Strict safety assertion: Production database MUST be 100% untouched
        if PROD_DB_PATH.exists():
            conn = sqlite3.connect(f"file:{PROD_DB_PATH}?mode=ro", uri=True)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM articles")
            curr_articles = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM nlp_results")
            curr_nlp = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM validation_logs")
            curr_val = cur.fetchone()[0]
            conn.close()

            self.assertEqual(curr_articles, self.baseline_articles, "Production articles count changed!")
            self.assertEqual(curr_nlp, self.baseline_nlp, "Production nlp_results count changed!")
            self.assertEqual(curr_val, self.baseline_val, "Production validation_logs count changed!")

    # 1. Credential detection & redaction privacy
    def test_01_credentials_detection_and_privacy(self):
        key, sec = get_alpaca_credentials()
        # Credentials must be detected from environment
        self.assertIsNotNone(key)
        self.assertIsNotNone(sec)
        self.assertTrue(len(key) > 10)
        self.assertTrue(len(sec) > 10)

        # Test sanitize_text
        test_msg = f"Failed to authenticate with key {key} and secret {sec} on server"
        sanitized = sanitize_text(test_msg, key_id=key, secret_key=sec)
        self.assertNotIn(key, sanitized)
        self.assertNotIn(sec, sanitized)
        self.assertIn("[REDACTED_KEY]", sanitized)
        self.assertIn("[REDACTED_SECRET]", sanitized)

    # 2. Normalization of incoming Benzinga payloads
    def test_02_stream_article_normalization(self):
        sample_item = {
            "T": "n",
            "id": 1001234,
            "headline": "Apple Unveils Next-Gen AI Chip Architecture for Enterprise Servers",
            "summary": "Apple Inc announced breakthrough silicon hardware designed for private enterprise clouds.",
            "content": "<p>Apple Inc (NASDAQ: AAPL) today announced its <b>M4 Ultra</b> enterprise server chips...</p>",
            "symbols": ["AAPL"],
            "created_at": "2026-10-13T12:15:00Z",
            "updated_at": "2026-10-13T12:15:30Z",
            "url": "https://www.benzinga.com/news/26/10/1001234/apple-unveils-ai-chip",
            "source": "Benzinga"
        }

        norm = normalize_stream_article(sample_item, target_ticker="AAPL")
        self.assertIsNotNone(norm)
        self.assertEqual(norm["Ticker"], "AAPL")
        self.assertEqual(norm["Company_Name"], "Apple")
        self.assertEqual(norm["Headline"], "Apple Unveils Next-Gen AI Chip Architecture for Enterprise Servers")
        self.assertEqual(norm["Source"], "Benzinga")
        self.assertEqual(norm["Provider"], "Alpaca")
        self.assertEqual(norm["Dataset_Type"], "REAL")
        self.assertEqual(len(norm["News_ID"]), 32)
        self.assertEqual(norm["Published_At_UTC"], "2026-10-13T12:15:00Z")

        # Verify HTML cleaning in content fallback
        sample_no_summary = dict(sample_item)
        sample_no_summary["summary"] = ""
        norm_fallback = normalize_stream_article(sample_no_summary, target_ticker="AAPL")
        self.assertIsNotNone(norm_fallback)
        self.assertNotIn("<p>", norm_fallback["Summary"])
        self.assertNotIn("<b>", norm_fallback["Summary"])
        self.assertIn("Apple Inc (NASDAQ: AAPL)", norm_fallback["Summary"])

    # 3. Ticker Aliasing (FB -> META)
    def test_03_ticker_aliasing(self):
        sample_fb = {
            "T": "n",
            "id": 1005678,
            "headline": "Meta Platforms Announces Expansion of Data Center Fleet",
            "summary": "Meta Platforms is expanding its renewable infrastructure.",
            "symbols": ["FB"],
            "created_at": "2026-10-13T19:30:00Z",
            "url": "https://www.benzinga.com/news/26/10/1005678/meta-data-center",
            "source": "Benzinga"
        }
        norm = normalize_stream_article(sample_fb, target_ticker="META")
        self.assertIsNotNone(norm)
        self.assertEqual(norm["Ticker"], "META")
        self.assertEqual(norm["Company_Name"], "Meta Platforms")

    # 4. Article quality validation & lookahead guard
    def test_04_article_validation_and_lookahead(self):
        # Empty headline
        bad_art = {
            "News_ID": "12345678901234567890123456789012",
            "Ticker": "AAPL",
            "Published_At_UTC": "2026-10-13T20:00:00Z",
            "Headline": ""
        }
        is_valid, reason = validate_article(bad_art)
        self.assertFalse(is_valid)

        # Future publication timestamp (> 15 min into future)
        future_pub = (datetime.now(timezone.utc) + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        future_art = {
            "News_ID": "12345678901234567890123456789012",
            "Ticker": "AAPL",
            "Published_At_UTC": future_pub,
            "Headline": "Valid Headline About Apple Future"
        }
        is_valid, reason = validate_article(future_art)
        self.assertFalse(is_valid)
        self.assertIn("FUTURE_PUBLICATION_TIMESTAMP", reason)

        # Unsupported ticker
        unsupported_art = {
            "News_ID": "12345678901234567890123456789012",
            "Ticker": "UNKNOWN_TICKER",
            "Published_At_UTC": "2026-10-13T20:00:00Z",
            "Headline": "Valid Headline For Unknown Equity"
        }
        is_valid, reason = validate_article(unsupported_art)
        self.assertFalse(is_valid)
        self.assertIn("UNSUPPORTED_TICKER", reason)

    # 5. Storage persistence & deduplication
    def test_05_storage_and_deduplication(self):
        article_1 = {
            "News_ID": generate_news_id("MSFT", "2026-10-13T18:00:00Z", "Microsoft Expands Azure AI Foundry", "https://example.com/1"),
            "Ticker": "MSFT",
            "Company_Name": "Microsoft",
            "Published_At_UTC": "2026-10-13T18:00:00Z",
            "Headline": "Microsoft Expands Azure AI Foundry",
            "Summary": "Microsoft today announced enhanced Azure enterprise tooling.",
            "Source": "Benzinga",
            "URL": "https://example.com/1",
            "Provider": "Alpaca",
            "Retrieved_At_UTC": "2026-10-13T18:01:00Z",
            "Provider_Sentiment": 0.0,
            "Provider_Relevance": 0.50,
            "Dataset_Type": "REAL"
        }

        # First insert -> Added = 1
        added, dups, rejs = store_articles([article_1])
        self.assertEqual(added, 1)
        self.assertEqual(dups, 0)
        self.assertEqual(rejs, 0)

        # Immediate duplicate insert -> Added = 0, Dups = 1
        added2, dups2, rejs2 = store_articles([article_1])
        self.assertEqual(added2, 0)
        self.assertEqual(dups2, 1)
        self.assertEqual(rejs2, 0)

        # Verify single record in test SQLite DB
        test_db = self.temp_path / "news_database.db"
        conn = sqlite3.connect(test_db)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM articles WHERE ticker = 'MSFT'")
        count = cur.fetchone()[0]
        conn.close()
        self.assertEqual(count, 1)

    # 6. Loughran-McDonald NLP Pipeline scoring on streamed article
    def test_06_nlp_pipeline_on_streamed_article(self):
        article_dict = {
            "News_ID": generate_news_id("NVDA", "2026-10-13T17:00:00Z", "NVIDIA Reports Record Server GPU Shipments", "https://example.com/nvda"),
            "Ticker": "NVDA",
            "Company_Name": "NVIDIA",
            "Published_At_UTC": "2026-10-13T17:00:00Z",
            "Headline": "NVIDIA Reports Record Server GPU Shipments and Strong Revenue Growth",
            "Summary": "NVIDIA posted record quarterly shipments with profitable operating margins.",
            "Source": "Benzinga",
            "URL": "https://example.com/nvda",
            "Provider": "Alpaca",
            "Retrieved_At_UTC": "2026-10-13T17:01:00Z",
            "Provider_Sentiment": 0.0,
            "Provider_Relevance": 0.50,
            "Dataset_Type": "REAL"
        }

        df_in = pd.DataFrame([article_dict])
        nlp_df, daily_df = process_articles_nlp(df_in)

        self.assertFalse(nlp_df.empty)
        self.assertEqual(len(nlp_df), 1)
        row = nlp_df.iloc[0]
        self.assertEqual(row["News_ID"], article_dict["News_ID"])
        self.assertEqual(row["Ticker"], "NVDA")
        self.assertIn(row["Sentiment_Label"], ["Positive", "Neutral", "Negative"])
        self.assertTrue(isinstance(row["Sentiment_Score"], (int, float)))
        self.assertTrue(isinstance(row["Triage_Score"], (int, float)))

        # Store NLP results in test DB
        store_nlp_results(nlp_df, daily_df)
        test_db = self.temp_path / "news_database.db"
        conn = sqlite3.connect(test_db)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM nlp_results WHERE ticker = 'NVDA'")
        nlp_count = cur.fetchone()[0]
        conn.close()
        self.assertEqual(nlp_count, 1)

    # 7. Mocked WebSocket Stream Lifecycle (Connect -> Auth -> Subscribe -> Ingest -> Stop)
    @patch("websockets.sync.client.connect")
    def test_07_mocked_websocket_lifecycle(self, mock_ws_connect):
        mock_ws = MagicMock()
        mock_ws_connect.return_value = mock_ws

        # Sequence of messages returned by mock ws.recv()
        greeting_msg = json.dumps([{"T": "success", "msg": "connected"}])
        auth_msg = json.dumps([{"T": "success", "msg": "authenticated"}])
        sub_msg = json.dumps([{"T": "subscription", "news": ["AAPL", "MSFT"]}])
        news_frame = json.dumps([{
            "T": "n",
            "id": 999123,
            "headline": "Tesla Expands Supercharger Network Across High-Traffic Corridors",
            "summary": "Tesla Inc announced 500 new fast-charging sites globally.",
            "symbols": ["TSLA"],
            "created_at": "2026-10-13T16:00:00Z",
            "url": "https://www.benzinga.com/news/tesla-charger",
            "source": "Benzinga"
        }])

        messages = [greeting_msg, auth_msg, sub_msg, news_frame]
        def mock_recv(*args, **kwargs):
            if messages:
                return messages.pop(0)
            time.sleep(0.05)
            raise TimeoutError("Socket idle timeout")
        mock_ws.recv.side_effect = mock_recv

        listener = AlpacaStreamListener(
            tickers=["TSLA", "AAPL"],
            status_path=self.test_status_path,
            max_reconnects=1
        )

        listener.start()
        # Allow background thread to process handshake and article frame
        time.sleep(0.3)

        status = listener.get_status()
        self.assertTrue(status["connected"])
        self.assertTrue(status["authenticated"])
        self.assertTrue(status["subscribed"])
        self.assertEqual(status["articles_received_count"], 1)
        self.assertEqual(status["articles_stored_count"], 1)
        self.assertIsNotNone(status["last_article"])
        self.assertEqual(status["last_article"]["ticker"], "TSLA")

        listener.stop(timeout=2.0)
        self.assertFalse(listener.is_alive())
        self.assertEqual(listener.get_status()["status"], "STOPPED")

    # 8. Reconnection and Backoff Logic
    @patch("websockets.sync.client.connect")
    def test_08_reconnection_backoff_handling(self, mock_ws_connect):
        # Simulate connection error on first attempt, success on second
        mock_ws_first = MagicMock()
        mock_ws_first.recv.side_effect = ConnectionResetError("Connection dropped by peer")

        messages_second = [
            json.dumps([{"T": "success", "msg": "connected"}]),
            json.dumps([{"T": "success", "msg": "authenticated"}]),
            json.dumps([{"T": "subscription", "news": ["AAPL"]}])
        ]
        def mock_recv_second(*args, **kwargs):
            if messages_second:
                return messages_second.pop(0)
            time.sleep(0.05)
            raise TimeoutError()

        mock_ws_second = MagicMock()
        mock_ws_second.recv.side_effect = mock_recv_second

        mock_ws_connect.side_effect = [
            mock_ws_first,
            mock_ws_second
        ]

        listener = AlpacaStreamListener(
            tickers=["AAPL"],
            status_path=self.test_status_path,
            reconnect_delay_base=0.05,
            reconnect_delay_max=0.1,
            max_reconnects=2
        )

        listener.start()
        time.sleep(0.4)
        status = listener.get_status()
        # Listener must have performed at least 1 reconnection attempt
        self.assertTrue(status.get("total_reconnects", 0) >= 1)
        self.assertEqual(status.get("active_retry_attempt", 0), 0)
        self.assertEqual(status.get("reconnect_count", 0), 0)
        listener.stop(timeout=1.0)

    # 9. Status export and telemetry consumption for Real-Time Data Dashboard
    def test_09_status_export_and_dashboard_consumption(self):
        listener = AlpacaStreamListener(
            tickers=["AAPL", "MSFT", "GOOGL"],
            status_path=self.test_status_path
        )
        self.assertTrue(self.test_status_path.exists())

        # Test get_stream_status reads exported JSON accurately
        status = get_stream_status(self.test_status_path)
        self.assertEqual(status["status"], "STOPPED")
        self.assertEqual(status["subscribed_tickers"], ["AAPL", "MSFT", "GOOGL"])
        self.assertEqual(status["articles_received_count"], 0)
        self.assertIsNone(status["last_article"])

    # 10. Stale Heartbeat Detection (Auto-degrade to OFFLINE if heartbeat > threshold)
    def test_10_stale_heartbeat_detection(self):
        # 1. Fresh heartbeat -> STREAMING maintained
        fresh_time = (datetime.now(timezone.utc) - timedelta(seconds=5)).strftime("%Y-%m-%d %H:%M:%S UTC")
        fresh_status = {
            "status": "STREAMING",
            "connected": True,
            "authenticated": True,
            "subscribed": True,
            "endpoint": "wss://stream.data.alpaca.markets/v1beta1/news",
            "process_id": os.getpid(),
            "last_heartbeat_utc": fresh_time,
            "subscribed_tickers": ["AAPL", "MSFT"],
            "articles_received_count": 0,
            "articles_stored_count": 0,
            "active_retry_attempt": 0,
            "total_reconnects": 0
        }
        with open(self.test_status_path, "w", encoding="utf-8") as f:
            json.dump(fresh_status, f)

        status = get_stream_status(self.test_status_path, stale_threshold_seconds=30.0)
        self.assertEqual(status["status"], "STREAMING")
        self.assertTrue(status["connected"])
        self.assertFalse(status["is_stale"])
        self.assertIsNotNone(status["heartbeat_age_seconds"])
        self.assertTrue(status["heartbeat_age_seconds"] < 30.0)

        # 2. Stale heartbeat (e.g. 60 seconds ago) -> Auto-degrades to OFFLINE
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=60)).strftime("%Y-%m-%d %H:%M:%S UTC")
        stale_status = dict(fresh_status)
        stale_status["last_heartbeat_utc"] = stale_time
        with open(self.test_status_path, "w", encoding="utf-8") as f:
            json.dump(stale_status, f)

        status_stale = get_stream_status(self.test_status_path, stale_threshold_seconds=30.0)
        self.assertEqual(status_stale["status"], "OFFLINE")
        self.assertFalse(status_stale["connected"])
        self.assertFalse(status_stale["authenticated"])
        self.assertFalse(status_stale["subscribed"])
        self.assertTrue(status_stale["is_stale"])
        self.assertIsNotNone(status_stale["error_message"])
        self.assertIn("Heartbeat is stale", status_stale["error_message"])
        self.assertTrue(status_stale["heartbeat_age_seconds"] >= 59.0)

    # 11. Active retry counter vs cumulative reconnects distinction
    @patch("websockets.sync.client.connect")
    def test_11_active_vs_total_reconnect_counters(self, mock_ws_connect):
        mock_ws_fail = MagicMock()
        mock_ws_fail.recv.side_effect = ConnectionResetError("Gateway reset")

        mock_ws_ok = MagicMock()
        messages_ok = [
            json.dumps([{"T": "success", "msg": "connected"}]),
            json.dumps([{"T": "success", "msg": "authenticated"}]),
            json.dumps([{"T": "subscription", "news": ["AAPL"]}])
        ]
        def mock_recv_ok(*args, **kwargs):
            if messages_ok:
                return messages_ok.pop(0)
            time.sleep(0.05)
            raise TimeoutError()
        mock_ws_ok.recv.side_effect = mock_recv_ok

        mock_ws_connect.side_effect = [mock_ws_fail, mock_ws_ok]

        listener = AlpacaStreamListener(
            tickers=["AAPL"],
            status_path=self.test_status_path,
            reconnect_delay_base=0.05,
            reconnect_delay_max=0.1,
            max_reconnects=2
        )
        listener.start()
        time.sleep(0.4)
        status = listener.get_status()

        self.assertGreaterEqual(status["total_reconnects"], 1)
        self.assertEqual(status["active_retry_attempt"], 0)
        self.assertEqual(status["reconnect_count"], 0)
        self.assertEqual(status["status"], "STREAMING")
        listener.stop(timeout=1.0)

    # 12. Alpaca Code 406 (Connection Limit Exceeded) handled as transient backoff
    @patch("websockets.sync.client.connect")
    def test_12_code_406_connection_limit_handling(self, mock_ws_connect):
        mock_ws_406 = MagicMock()
        messages_406 = [
            json.dumps([{"T": "success", "msg": "connected"}]),
            json.dumps([{"T": "error", "code": 406, "msg": "connection limit exceeded"}])
        ]
        def mock_recv_406(*args, **kwargs):
            if messages_406:
                return messages_406.pop(0)
            raise TimeoutError()
        mock_ws_406.recv.side_effect = mock_recv_406

        mock_ws_ok = MagicMock()
        messages_ok = [
            json.dumps([{"T": "success", "msg": "connected"}]),
            json.dumps([{"T": "success", "msg": "authenticated"}]),
            json.dumps([{"T": "subscription", "news": ["AAPL"]}])
        ]
        def mock_recv_ok(*args, **kwargs):
            if messages_ok:
                return messages_ok.pop(0)
            time.sleep(0.05)
            raise TimeoutError()
        mock_ws_ok.recv.side_effect = mock_recv_ok

        mock_ws_connect.side_effect = [mock_ws_406, mock_ws_ok]

        listener = AlpacaStreamListener(
            tickers=["AAPL"],
            status_path=self.test_status_path,
            reconnect_delay_base=0.05,
            reconnect_delay_max=0.1,
            max_reconnects=2
        )
        listener.start()
        time.sleep(0.4)
        status = listener.get_status()

        self.assertEqual(status["status"], "STREAMING")
        self.assertTrue(status["connected"])
        self.assertTrue(status["authenticated"])
        self.assertGreaterEqual(status["total_reconnects"], 1)
        listener.stop(timeout=1.0)

    # 13. Fatal authentication rejection handling (e.g. Code 401)
    @patch("websockets.sync.client.connect")
    def test_13_fatal_auth_failure_handling(self, mock_ws_connect):
        mock_ws_401 = MagicMock()
        messages_401 = [
            json.dumps([{"T": "success", "msg": "connected"}]),
            json.dumps([{"T": "error", "code": 401, "msg": "authentication failed: invalid key"}])
        ]
        def mock_recv_401(*args, **kwargs):
            if messages_401:
                return messages_401.pop(0)
            raise TimeoutError()
        mock_ws_401.recv.side_effect = mock_recv_401
        mock_ws_connect.return_value = mock_ws_401

        listener = AlpacaStreamListener(
            tickers=["AAPL"],
            status_path=self.test_status_path,
            max_reconnects=3
        )
        listener.start()
        time.sleep(0.3)
        self.assertFalse(listener.is_alive())
        status = listener.get_status()
        self.assertEqual(status["status"], "ERROR")
        self.assertEqual(status["error_code"], 401)
        self.assertIn("Alpaca auth error 401", status["error_message"])

    # 14. Strict symbol association: Exclude and never relabel unrelated tickers
    def test_14_unrelated_ticker_exclusion(self):
        listener = AlpacaStreamListener(
            tickers=["AAPL", "MSFT"],
            status_path=self.test_status_path
        )

        item_unrelated = {
            "T": "n",
            "id": 888001,
            "headline": "Dutch Semiconductor Firm ASML Delivers Breakthrough Optics",
            "summary": "ASML reported strong bookings across European semiconductor fabs.",
            "symbols": ["ASML", "NVDA"],
            "created_at": "2026-10-13T14:00:00Z",
            "url": "https://www.benzinga.com/news/asml-breakthrough",
            "source": "Benzinga"
        }

        item_no_symbols = {
            "T": "n",
            "id": 888002,
            "headline": "Macro Economy: Federal Reserve Comments on Yield Curve",
            "summary": "Bond yields shifted following today's remarks.",
            "symbols": [],
            "created_at": "2026-10-13T14:05:00Z",
            "url": "https://www.benzinga.com/news/fed-yields",
            "source": "Benzinga"
        }

        listener._process_incoming_news_item(item_unrelated)
        listener._process_incoming_news_item(item_no_symbols)

        status = listener.get_status()
        self.assertEqual(status["articles_received_count"], 0)
        self.assertEqual(status["articles_stored_count"], 0)
        self.assertEqual(status["non_tracked_articles_count"], 2)
        self.assertIsNone(status["last_article"])

        test_db = self.temp_path / "news_database.db"
        conn = sqlite3.connect(test_db)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM articles")
        count = cur.fetchone()[0]
        conn.close()
        self.assertEqual(count, 0)

    # 15. Strict symbol association: Ingest and run NLP for matching tracked ticker
    def test_15_tracked_ticker_association(self):
        listener = AlpacaStreamListener(
            tickers=["AAPL", "MSFT"],
            status_path=self.test_status_path
        )

        item_matched = {
            "T": "n",
            "id": 888003,
            "headline": "Apple Partners With Major Health Network on Watch OS Diagnostics",
            "summary": "Apple Inc expands health monitoring clinical trials across North America.",
            "symbols": ["AAPL", "UNH"],
            "created_at": "2026-10-13T15:00:00Z",
            "url": "https://www.benzinga.com/news/apple-health-watch",
            "source": "Benzinga"
        }

        listener._process_incoming_news_item(item_matched)

        status = listener.get_status()
        self.assertEqual(status["articles_received_count"], 1)
        self.assertEqual(status["articles_stored_count"], 1)
        self.assertEqual(status["non_tracked_articles_count"], 0)
        self.assertIsNotNone(status["last_article"])
        self.assertEqual(status["last_article"]["ticker"], "AAPL")
        self.assertIn("Apple Partners With Major Health", status["last_article"]["headline"])

        test_db = self.temp_path / "news_database.db"
        conn = sqlite3.connect(test_db)
        cur = conn.cursor()
        cur.execute("SELECT ticker, headline FROM articles")
        rows = cur.fetchall()
        cur.execute("SELECT COUNT(*) FROM nlp_results WHERE ticker = 'AAPL'")
        nlp_count = cur.fetchone()[0]
        conn.close()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "AAPL")
        self.assertEqual(nlp_count, 1)


if __name__ == "__main__":
    unittest.main()
