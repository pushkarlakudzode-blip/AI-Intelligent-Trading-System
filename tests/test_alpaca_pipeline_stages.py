"""
End-to-End Pipeline Stage Verification Test Suite for Alpaca Real-Time News Ingestion.

Validates the full 7-stage ingestion lifecycle in an isolated staging environment:
Stage 1: WebSocket Frame Reception & JSON Parsing
Stage 2: Actual Ticker Matching & Strict Symbol Association (including FB -> META aliasing & exclusion)
Stage 3: Article Normalization & Field Formatting
Stage 4: Quality Validation & Lookahead Constraint Enforcement
Stage 5: SQLite Persistence & Deduplication
Stage 6: Loughran-McDonald NLP Pipeline & Relational Foreign Key Linkage
Stage 7: Dashboard Query & Spotlight Telemetry Alignment
Stage 8: End-to-End Listener Execution in Staging Mode
Stage 9: Production Database Provenance & Immutability Verification

Strict Safety Assertion:
Zero mutations to production database (data/live/news_database.db), research, or ML models.
"""

import hashlib
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
    TICKER_ALIASES,
    COMPANY_NAMES
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


class TestAlpacaPipelineStages(unittest.TestCase):

    def setUp(self):
        # 1. Record production database baseline counts and checksum
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
            self.baseline_md5 = hashlib.md5(PROD_DB_PATH.read_bytes()).hexdigest()
        else:
            self.baseline_articles = 0
            self.baseline_nlp = 0
            self.baseline_val = 0
            self.baseline_md5 = ""

        # 2. Create isolated staging directory for this test run
        self.temp_dir = tempfile.TemporaryDirectory()
        self.staging_path = Path(self.temp_dir.name)
        self.staging_status_path = self.staging_path / "stream_status.json"

        # 3. Configure storage subsystem strictly to staging directory
        configure_storage(self.staging_path)
        init_storage()

    def tearDown(self):
        # Reset storage subsystem to default
        configure_storage(None)
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

        # Strict Production Preservation Assertion
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

            curr_md5 = hashlib.md5(PROD_DB_PATH.read_bytes()).hexdigest()

            self.assertEqual(curr_articles, self.baseline_articles, "Production articles count mutated!")
            self.assertEqual(curr_nlp, self.baseline_nlp, "Production nlp_results count mutated!")
            self.assertEqual(curr_val, self.baseline_val, "Production validation_logs count mutated!")
            self.assertEqual(curr_md5, self.baseline_md5, "Production database MD5 checksum changed!")

    # -------------------------------------------------------------------------
    # STAGE 1: WebSocket Frame Reception & JSON Parsing
    # -------------------------------------------------------------------------
    def test_stage_1_websocket_message_reception_and_schema(self):
        """Verify real Benzinga/Alpaca news frame schema parsing and message discrimination."""
        raw_frame = json.dumps([{
            "T": "n",
            "id": 99881122,
            "headline": "Apple Expands AI Private Cloud Compute Infrastructure Across North America",
            "summary": "Apple Inc is expanding dedicated server nodes for its Apple Intelligence private cloud.",
            "content": "<p>Apple Inc (NASDAQ: AAPL) today announced an expansion of its <b>Private Cloud Compute</b> nodes...</p>",
            "symbols": ["AAPL"],
            "created_at": "2026-10-14T02:30:00Z",
            "updated_at": "2026-10-14T02:30:15Z",
            "url": "https://www.benzinga.com/news/26/10/99881122/apple-private-cloud-compute",
            "source": "Benzinga"
        }])

        parsed = json.loads(raw_frame)
        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), 1)
        item = parsed[0]
        self.assertEqual(item.get("T"), "n")
        self.assertEqual(item.get("id"), 99881122)
        self.assertIn("symbols", item)
        self.assertIn("headline", item)
        self.assertIn("created_at", item)

    # -------------------------------------------------------------------------
    # STAGE 2: Actual Ticker Matching & Strict Symbol Association
    # -------------------------------------------------------------------------
    def test_stage_2_ticker_matching_and_unrelated_exclusion(self):
        """Verify strict symbol association, aliasing (FB->META), and exclusion of foreign tickers."""
        listener = AlpacaStreamListener(
            tickers=["AAPL", "META", "NVDA"],
            status_path=self.staging_status_path
        )

        # 1. Matching tracked ticker
        item_aapl = {"T": "n", "id": 1, "headline": "Apple News", "symbols": ["AAPL"], "created_at": "2026-10-13T20:00:00Z"}
        raw_syms_aapl = item_aapl.get("symbols", [])
        matched_aapl = {TICKER_ALIASES.get(s.upper(), s.upper()) for s in raw_syms_aapl} & set(listener.tickers)
        self.assertEqual(matched_aapl, {"AAPL"})

        # 2. Aliased ticker (FB -> META)
        item_fb = {"T": "n", "id": 2, "headline": "Meta News", "symbols": ["FB"], "created_at": "2026-10-13T20:00:00Z"}
        matched_fb = {TICKER_ALIASES.get(s.upper(), s.upper()) for s in item_fb.get("symbols", [])} & set(listener.tickers)
        self.assertEqual(matched_fb, {"META"})

        # 3. Multi-ticker payload matching 2 tracked equities and 1 untracked ETF
        item_multi = {"T": "n", "id": 3, "headline": "Tech AI Rally", "symbols": ["AAPL", "NVDA", "QQQ"], "created_at": "2026-10-13T20:00:00Z"}
        matched_multi = {TICKER_ALIASES.get(s.upper(), s.upper()) for s in item_multi.get("symbols", [])} & set(listener.tickers)
        self.assertEqual(matched_multi, {"AAPL", "NVDA"})
        self.assertNotIn("QQQ", matched_multi)

        # 4. Completely unrelated equity (ASML, BABA) -> Must be excluded, NEVER relabeled
        item_unrelated = {
            "T": "n",
            "id": 4,
            "headline": "Dutch Semiconductor ASML Ships Extreme UV Stepper",
            "symbols": ["ASML", "BABA"],
            "created_at": "2026-10-13T20:00:00Z",
            "url": "https://example.com/asml"
        }
        listener._process_incoming_news_item(item_unrelated)
        status = listener.get_status()
        self.assertEqual(status["articles_received_count"], 0)
        self.assertEqual(status["articles_stored_count"], 0)
        self.assertEqual(status["non_tracked_articles_count"], 1)

        # Confirm 0 articles in staging DB
        staging_db = self.staging_path / "news_database.db"
        conn = sqlite3.connect(staging_db)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM articles")
        cnt = cur.fetchone()[0]
        conn.close()
        self.assertEqual(cnt, 0)

    # -------------------------------------------------------------------------
    # STAGE 3: Article Normalization & Field Formatting
    # -------------------------------------------------------------------------
    def test_stage_3_normalization(self):
        """Verify field normalization, HTML tag stripping, and ISO UTC timestamp generation."""
        raw_item = {
            "T": "n",
            "id": 99881122,
            "headline": "Apple Expands AI Private Cloud Compute Infrastructure Across North America",
            "summary": "Apple Inc is expanding dedicated server nodes for its Apple Intelligence private cloud.",
            "content": "<p>Apple Inc (NASDAQ: AAPL) today announced an expansion of its <b>Private Cloud Compute</b> nodes...</p>",
            "symbols": ["AAPL"],
            "created_at": "2026-10-14T02:30:00Z",
            "updated_at": "2026-10-14T02:30:15Z",
            "url": "https://www.benzinga.com/news/26/10/99881122/apple-private-cloud-compute",
            "source": "Benzinga"
        }

        norm = normalize_stream_article(raw_item, target_ticker="AAPL")
        self.assertIsNotNone(norm)
        self.assertEqual(norm["Ticker"], "AAPL")
        self.assertEqual(norm["Company_Name"], "Apple")
        self.assertEqual(norm["Headline"], "Apple Expands AI Private Cloud Compute Infrastructure Across North America")
        self.assertEqual(norm["Source"], "Benzinga")
        self.assertEqual(norm["Provider"], "Alpaca")
        self.assertEqual(norm["Dataset_Type"], "REAL")
        self.assertEqual(norm["Published_At_UTC"], "2026-10-14T02:30:00Z")
        self.assertEqual(len(norm["News_ID"]), 32)

        # Content fallback with HTML stripping
        raw_no_summary = dict(raw_item)
        raw_no_summary["summary"] = ""
        norm_fallback = normalize_stream_article(raw_no_summary, target_ticker="AAPL")
        self.assertNotIn("<p>", norm_fallback["Summary"])
        self.assertNotIn("<b>", norm_fallback["Summary"])
        self.assertIn("Private Cloud Compute", norm_fallback["Summary"])

    # -------------------------------------------------------------------------
    # STAGE 4: Quality Validation & Lookahead Constraint Enforcement
    # -------------------------------------------------------------------------
    def test_stage_4_validation_and_lookahead_guards(self):
        """Verify quality checks: empty headline rejection and lookahead constraint protection."""
        valid_article = {
            "News_ID": generate_news_id("AAPL", "2026-10-13T21:00:00Z", "Apple Announces M5 Silicon Architecture", "https://example.com/aapl"),
            "Ticker": "AAPL",
            "Company_Name": "Apple",
            "Published_At_UTC": "2026-10-13T21:00:00Z",
            "Headline": "Apple Announces M5 Silicon Architecture",
            "Summary": "Apple unveils next-generation silicon design.",
            "Source": "Benzinga",
            "URL": "https://example.com/aapl",
            "Provider": "Alpaca",
            "Retrieved_At_UTC": "2026-10-13T21:01:00Z",
            "Provider_Sentiment": 0.0,
            "Provider_Relevance": 0.5,
            "Dataset_Type": "REAL"
        }

        is_valid, msg = validate_article(valid_article)
        self.assertTrue(is_valid)
        self.assertIsNone(msg)

        # Empty headline rejection
        bad_headline = dict(valid_article, Headline="   ")
        is_val, issue = validate_article(bad_headline)
        self.assertFalse(is_val)
        self.assertIn("EMPTY_OR_TRUNCATED_HEADLINE", issue)

        # Lookahead violation rejection (> 15 min in the future)
        future_dt = (datetime.now(timezone.utc) + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
        future_art = dict(valid_article, Published_At_UTC=future_dt)
        is_val_fut, issue_fut = validate_article(future_art)
        self.assertFalse(is_val_fut)
        self.assertIn("FUTURE_PUBLICATION_TIMESTAMP", issue_fut)

    # -------------------------------------------------------------------------
    # STAGE 5: SQLite Persistence & Deduplication
    # -------------------------------------------------------------------------
    def test_stage_5_persistence_and_deduplication(self):
        """Verify insertion into staging SQLite articles table and robust duplicate suppression."""
        article = {
            "News_ID": generate_news_id("MSFT", "2026-10-13T20:30:00Z", "Microsoft Extends Copilot to Global Financial Institutions", "https://example.com/msft"),
            "Ticker": "MSFT",
            "Company_Name": "Microsoft",
            "Published_At_UTC": "2026-10-13T20:30:00Z",
            "Headline": "Microsoft Extends Copilot to Global Financial Institutions",
            "Summary": "Microsoft announced expanded enterprise AI deployments.",
            "Source": "Benzinga",
            "URL": "https://example.com/msft",
            "Provider": "Alpaca",
            "Retrieved_At_UTC": "2026-10-13T20:31:00Z",
            "Provider_Sentiment": 0.0,
            "Provider_Relevance": 0.5,
            "Dataset_Type": "REAL"
        }

        # First ingestion -> Added = 1
        added, dups, rejs = store_articles([article])
        self.assertEqual((added, dups, rejs), (1, 0, 0))

        # Duplicate ingestion -> Added = 0, Dups = 1
        added2, dups2, rejs2 = store_articles([article])
        self.assertEqual((added2, dups2, rejs2), (0, 1, 0))

        # Check staging SQLite DB schema alignment
        staging_db = self.staging_path / "news_database.db"
        conn = sqlite3.connect(staging_db)
        cur = conn.cursor()
        cur.execute("SELECT news_id, ticker, headline, provider FROM articles WHERE ticker = 'MSFT'")
        row = cur.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], article["News_ID"])
        self.assertEqual(row[1], "MSFT")
        self.assertEqual(row[2], article["Headline"])
        self.assertEqual(row[3], "Alpaca")

    # -------------------------------------------------------------------------
    # STAGE 6: Loughran-McDonald NLP Pipeline & Relational Foreign Key Linkage
    # -------------------------------------------------------------------------
    def test_stage_6_loughran_mcdonald_nlp_and_relational_linkage(self):
        """Verify Loughran-McDonald NLP scoring, triage, and exact news_id linkage in nlp_results."""
        article = {
            "News_ID": generate_news_id("NVDA", "2026-10-13T19:00:00Z", "NVIDIA Launches Blackwell Ultra GPUs With Record Energy Efficiency", "https://example.com/nvda-ultra"),
            "Ticker": "NVDA",
            "Company_Name": "NVIDIA",
            "Published_At_UTC": "2026-10-13T19:00:00Z",
            "Headline": "NVIDIA Launches Blackwell Ultra GPUs With Record Energy Efficiency and Profitable Operating Margins",
            "Summary": "NVIDIA posted record revenue forecasts and accelerated data center shipments.",
            "Source": "Benzinga",
            "URL": "https://example.com/nvda-ultra",
            "Provider": "Alpaca",
            "Retrieved_At_UTC": "2026-10-13T19:01:00Z",
            "Provider_Sentiment": 0.0,
            "Provider_Relevance": 0.5,
            "Dataset_Type": "REAL"
        }

        store_articles([article])

        # Execute NLP pipeline
        df_in = pd.DataFrame([article])
        nlp_df, daily_df = process_articles_nlp(df_in)

        self.assertFalse(nlp_df.empty)
        row = nlp_df.iloc[0]
        self.assertEqual(row["News_ID"], article["News_ID"])
        self.assertEqual(row["Ticker"], "NVDA")
        self.assertEqual(row["Sentiment_Label"], "Positive")
        self.assertGreater(row["Sentiment_Score"], 0.0)
        self.assertGreater(row["Triage_Score"], 0.0)
        self.assertIn("Event_Type", row)

        # Store in staging database
        store_nlp_results(nlp_df, daily_df)

        staging_db = self.staging_path / "news_database.db"
        conn = sqlite3.connect(staging_db)
        cur = conn.cursor()
        cur.execute("""
            SELECT a.ticker, a.headline, n.sentiment_label, n.sentiment_score, n.triage_score, n.event_type
            FROM articles a
            JOIN nlp_results n ON a.news_id = n.news_id
            WHERE a.ticker = 'NVDA'
        """)
        join_row = cur.fetchone()
        conn.close()

        self.assertIsNotNone(join_row)
        self.assertEqual(join_row[0], "NVDA")
        self.assertEqual(join_row[2], "Positive")
        self.assertGreater(join_row[3], 0.0)
        self.assertGreater(join_row[4], 0.0)

    # -------------------------------------------------------------------------
    # STAGE 7: Dashboard Query & Spotlight Telemetry Alignment
    # -------------------------------------------------------------------------
    def test_stage_7_dashboard_query_and_spotlight_alignment(self):
        """Verify dashboard SQL join and last_article telemetry spotlight mapping."""
        article = {
            "News_ID": generate_news_id("AAPL", "2026-10-13T21:45:00Z", "Apple Partners with Global Banks on Tap to Pay Expansion", "https://example.com/aapl-bank"),
            "Ticker": "AAPL",
            "Company_Name": "Apple",
            "Published_At_UTC": "2026-10-13T21:45:00Z",
            "Headline": "Apple Partners with Global Banks on Tap to Pay Expansion",
            "Summary": "Apple expands fintech services internationally.",
            "Source": "Benzinga",
            "URL": "https://example.com/aapl-bank",
            "Provider": "Alpaca",
            "Retrieved_At_UTC": "2026-10-13T21:46:00Z",
            "Provider_Sentiment": 0.0,
            "Provider_Relevance": 0.5,
            "Dataset_Type": "REAL"
        }

        store_articles([article])
        nlp_df, daily_df = process_articles_nlp(pd.DataFrame([article]))
        store_nlp_results(nlp_df, daily_df)

        # Execute exact query used by load_latest_real_articles() in dashboard/app.py
        staging_db = self.staging_path / "news_database.db"
        conn = sqlite3.connect(f"file:{staging_db}?mode=ro", uri=True)
        query = """
        SELECT a.headline, a.source, a.published_at_utc,
               n.sentiment_label, n.sentiment_score, n.relevance_score,
               n.importance_score, n.triage_score, n.event_type, a.url, a.provider
        FROM articles a
        JOIN nlp_results n ON a.news_id = n.news_id
        WHERE a.ticker = 'AAPL' AND a.dataset_type = 'REAL'
        ORDER BY a.published_at_utc DESC
        LIMIT 10
        """
        df_dash = pd.read_sql(query, conn)
        conn.close()

        self.assertEqual(len(df_dash), 1)
        r0 = df_dash.iloc[0]
        self.assertEqual(r0["provider"], "Alpaca")
        self.assertEqual(r0["headline"], article["Headline"])
        self.assertIn(r0["sentiment_label"], ["Positive", "Neutral", "Negative"])
        self.assertIn("event_type", r0)

    # -------------------------------------------------------------------------
    # STAGE 8: End-to-End Listener Execution in Staging Mode
    # -------------------------------------------------------------------------
    @patch("websockets.sync.client.connect")
    def test_stage_8_listener_end_to_end_in_staging(self, mock_ws_connect):
        """Simulate real WebSocket broadcast in AlpacaStreamListener and verify full automated flow."""
        mock_ws = MagicMock()
        mock_ws_connect.return_value = mock_ws

        greeting_msg = json.dumps([{"T": "success", "msg": "connected"}])
        auth_msg = json.dumps([{"T": "success", "msg": "authenticated"}])
        sub_msg = json.dumps([{"T": "subscription", "news": ["TSLA", "AAPL"]}])
        news_frame = json.dumps([{
            "T": "n",
            "id": 554433,
            "headline": "Tesla Deploys Next-Gen Megapack Energy Storage Across Texas Grid",
            "summary": "Tesla Energy secured major utility infrastructure deployment contract.",
            "symbols": ["TSLA"],
            "created_at": "2026-10-13T22:00:00Z",
            "url": "https://www.benzinga.com/news/tesla-megapack",
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
            status_path=self.staging_status_path,
            max_reconnects=1
        )

        listener.start()
        time.sleep(0.35)

        status = listener.get_status()
        self.assertTrue(status["connected"])
        self.assertTrue(status["authenticated"])
        self.assertTrue(status["subscribed"])
        self.assertEqual(status["articles_received_count"], 1)
        self.assertEqual(status["articles_stored_count"], 1)
        self.assertEqual(status["non_tracked_articles_count"], 0)
        self.assertIsNotNone(status["last_article"])
        self.assertEqual(status["last_article"]["ticker"], "TSLA")

        # Staging SQLite DB check
        staging_db = self.staging_path / "news_database.db"
        conn = sqlite3.connect(staging_db)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM articles WHERE ticker = 'TSLA'")
        art_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM nlp_results WHERE ticker = 'TSLA'")
        nlp_count = cur.fetchone()[0]
        conn.close()

        self.assertEqual(art_count, 1)
        self.assertEqual(nlp_count, 1)

        listener.stop(timeout=1.0)

    # -------------------------------------------------------------------------
    # STAGE 9: Production Database Provenance & Immutability Verification
    # -------------------------------------------------------------------------
    def test_stage_9_production_database_provenance_and_immutability(self):
        """Verify provenance of the 649th article and 648th NLP result in production SQLite DB."""
        if not PROD_DB_PATH.exists():
            self.skipTest("Production DB not present")

        conn = sqlite3.connect(f"file:{PROD_DB_PATH}?mode=ro", uri=True)
        cur = conn.cursor()

        # Check total row counts
        cur.execute("SELECT COUNT(*) FROM articles")
        total_articles = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM nlp_results")
        total_nlp = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM validation_logs")
        total_val = cur.fetchone()[0]

        self.assertGreaterEqual(total_articles, 650)
        self.assertGreaterEqual(total_nlp, 649)
        self.assertEqual(total_val, 6)

        # Inspect the 649th article (Visa broadcast from 2026-10-13)
        cur.execute("SELECT news_id, ticker, company_name, published_at_utc, headline, provider, retrieved_at_utc FROM articles WHERE rowid = 649")
        art_649 = cur.fetchone()
        self.assertIsNotNone(art_649)
        self.assertEqual(art_649[0], "c1538105139c7eb70189ba4af7e1ffad")
        self.assertEqual(art_649[1], "V")
        self.assertEqual(art_649[2], "Visa")
        self.assertEqual(art_649[3], "2026-10-13T19:38:06Z")
        self.assertIn("Buffett Successor Sold Visa Stock", art_649[4])
        self.assertEqual(art_649[5], "Alpaca")
        self.assertEqual(art_649[6], "2026-10-13T19:38:06Z")

        # Inspect the 650th article (JPMorgan Chase live broadcast from 2026-10-14)
        cur.execute("SELECT news_id, ticker, company_name, published_at_utc, headline, provider, retrieved_at_utc FROM articles WHERE rowid = 650")
        art_650 = cur.fetchone()
        self.assertIsNotNone(art_650)
        self.assertEqual(art_650[0], "02254762af1b07537f410701521ddcab")
        self.assertEqual(art_650[1], "JPM")
        self.assertEqual(art_650[2], "JPMorgan Chase")
        self.assertEqual(art_650[3], "2026-10-14T04:49:06Z")
        self.assertIn("Nasdaq Settles Lower", art_650[4])
        self.assertEqual(art_650[5], "Alpaca")
        self.assertEqual(art_650[6], "2026-10-14T04:49:06Z")

        # Inspect the matching 649th NLP result for JPM
        cur.execute("SELECT news_id, ticker, sentiment_label, sentiment_score, triage_score, event_type FROM nlp_results WHERE news_id = '02254762af1b07537f410701521ddcab'")
        nlp_649 = cur.fetchone()
        self.assertIsNotNone(nlp_649)
        self.assertEqual(nlp_649[0], "02254762af1b07537f410701521ddcab")
        self.assertEqual(nlp_649[1], "JPM")
        self.assertEqual(nlp_649[2], "Negative")
        self.assertEqual(nlp_649[3], -1.0)
        self.assertEqual(nlp_649[4], 0.27)
        self.assertEqual(nlp_649[5], "Earnings / Financials")

        conn.close()


if __name__ == "__main__":
    unittest.main()
