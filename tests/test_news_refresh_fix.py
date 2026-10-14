"""
Regression Test Suite for News Refresh Workflow Bug Fixes.

Covers all 10 verification requirements:
1. Default ticker set (tickers=None) no longer causes slicing TypeError.
2. Explicit ticker order is preserved.
3. Default ticker list is deterministic and sorted.
4. Alpha Vantage rate-limit responses do not crash the workflow.
5. Daily quota exhaustion stops further requests.
6. Remaining tickers are accurately marked skipped (SKIPPED_DUE_TO_QUOTA_LIMIT).
7. Duplicate articles do not increase the stored article count.
8. A zero-new-article refresh is reported accurately.
9. Cache invalidation and reload expose newly stored articles.
10. API failures do not erase or corrupt existing data.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.data_storage import (
    configure_storage,
    init_storage,
    store_articles,
    get_stored_articles,
    SUPPORTED_TICKERS
)
from src.analytics.refresh_manager import (
    execute_real_data_refresh,
    get_last_refresh_info,
    log_refresh_event,
    configure_refresh_manager,
    get_deterministic_tickers,
    is_quota_or_rate_limit,
    record_ticker_refresh,
    get_ticker_last_refresh,
    reset_cooldown_tracking,
)
from src.data.historical_news import query_alpha_vantage_news


class TestNewsRefreshFix(unittest.TestCase):

    def setUp(self):
        # Create an isolated temporary directory for test storage & logs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Configure data storage and refresh log in temp directory
        configure_storage(self.temp_path)
        init_storage()

        self.test_log_path = self.temp_path / "test_refresh_log.csv"
        self.test_market_path = self.temp_path / "test_market_extension.csv"
        self.test_analytics_dir = self.temp_path / "analytics"
        self.test_analytics_dir.mkdir(parents=True, exist_ok=True)

        configure_refresh_manager(self.test_log_path)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass
        finally:
            configure_storage(None)
            configure_refresh_manager(None)

    # 1. Default ticker set (tickers=None) no longer causes slicing TypeError
    @patch("src.analytics.refresh_manager.fetch_latest_news_for_ticker")
    def test_01_default_tickers_no_type_error(self, mock_news):
        # Return quota notice on the second ticker in the sorted universe
        mock_news.side_effect = [
            {"ticker": "AAPL", "status": "SUCCESS", "raw_fetched": 2, "added": 2, "duplicates": 0, "rejected": 0},
            {"ticker": "AMZN", "status": "Rate limit reached: Standard call frequency is 5 calls per minute",
             "raw_fetched": 0, "added": 0, "duplicates": 0, "rejected": 0}
        ]

        # Call with tickers=None (must default cleanly to sorted list, not crash on target_tickers[idx+1:])
        res = execute_real_data_refresh(
            scope="News only",
            tickers=None,
            rebuild_analytics=False,
            log_path=self.test_log_path,
            market_save_path=self.test_market_path
        )

        self.assertTrue(res["quota_limited"])
        self.assertEqual(res["articles_added"], 2)
        # Remaining tickers beyond AMZN must be marked SKIPPED_DUE_TO_QUOTA_LIMIT without any TypeError
        self.assertEqual(res["news_details"]["GOOGL"]["status"], "SKIPPED_DUE_TO_QUOTA_LIMIT")
        self.assertEqual(res["news_details"]["MSFT"]["status"], "SKIPPED_DUE_TO_QUOTA_LIMIT")
        self.assertNotIn("News refresh failed: 'set' object is not subscriptable", res.get("error_message", ""))

    # 2. Explicit ticker order is preserved
    def test_02_explicit_ticker_order_preserved(self):
        explicit_list = ["TSLA", "META", "AAPL", "NVDA"]
        result = get_deterministic_tickers(explicit_list)
        self.assertEqual(result, ["TSLA", "META", "AAPL", "NVDA"])

        # Also handles tuple and strips whitespace / uppercase
        explicit_tuple = (" jpm ", "V", "JNJ")
        result_tuple = get_deterministic_tickers(explicit_tuple)
        self.assertEqual(result_tuple, ["JPM", "V", "JNJ"])

    # 3. Default ticker list is deterministic and sorted
    def test_03_default_ticker_list_deterministic(self):
        result_1 = get_deterministic_tickers(None)
        result_2 = get_deterministic_tickers(None)
        expected = sorted(list(SUPPORTED_TICKERS))

        self.assertEqual(result_1, expected)
        self.assertEqual(result_2, expected)
        self.assertIsInstance(result_1, list)
        # Ensure SUPPORTED_TICKERS was not mutated
        self.assertIsInstance(SUPPORTED_TICKERS, set)

    # 4. Alpha Vantage rate-limit responses do not crash the workflow
    @patch("src.analytics.refresh_manager.fetch_latest_news_for_ticker")
    def test_04_rate_limit_response_handling(self, mock_news):
        mock_news.return_value = {
            "ticker": "AAPL",
            "status": "Rate limit reached: Standard call frequency is 5 calls per minute",
            "raw_fetched": 0,
            "added": 0,
            "duplicates": 0,
            "rejected": 0
        }

        res = execute_real_data_refresh(
            scope="News only",
            tickers=["AAPL"],
            rebuild_analytics=False,
            log_path=self.test_log_path,
            market_save_path=self.test_market_path
        )

        self.assertTrue(res["quota_limited"])
        self.assertEqual(res["articles_added"], 0)
        self.assertEqual(res["status"], "FAILED")
        self.assertIn("Alpha Vantage quota limit reached on AAPL", res["error_message"])

    # 5. Daily quota exhaustion stops further requests
    @patch("src.analytics.refresh_manager.fetch_latest_news_for_ticker")
    def test_05_daily_quota_exhaustion_stops_further_requests(self, mock_news):
        mock_news.return_value = {
            "ticker": "AAPL",
            "status": "Alpha Vantage Quota Limit: We have detected your API key as ... standard API rate limit is 25 requests per day",
            "raw_fetched": 0,
            "added": 0,
            "duplicates": 0,
            "rejected": 0
        }

        res = execute_real_data_refresh(
            scope="News only",
            tickers=["AAPL", "MSFT", "GOOGL", "AMZN"],
            rebuild_analytics=False,
            log_path=self.test_log_path,
            market_save_path=self.test_market_path
        )

        # Ensure only 1 network call occurred before breaking loop
        self.assertEqual(mock_news.call_count, 1)
        self.assertTrue(res["quota_limited"])

    # 6. Remaining tickers are accurately marked skipped
    @patch("src.analytics.refresh_manager.fetch_latest_news_for_ticker")
    def test_06_remaining_tickers_marked_skipped(self, mock_news):
        mock_news.side_effect = [
            {"ticker": "AAPL", "status": "SUCCESS", "raw_fetched": 3, "added": 3, "duplicates": 0, "rejected": 0},
            {"ticker": "MSFT", "status": "Rate limit reached: 25 requests per day", "raw_fetched": 0, "added": 0, "duplicates": 0, "rejected": 0}
        ]

        res = execute_real_data_refresh(
            scope="News only",
            tickers=["AAPL", "MSFT", "GOOGL", "AMZN"],
            rebuild_analytics=False,
            log_path=self.test_log_path,
            market_save_path=self.test_market_path
        )

        self.assertEqual(res["news_details"]["AAPL"]["status"], "SUCCESS")
        self.assertIn("Rate limit reached", res["news_details"]["MSFT"]["status"])
        self.assertEqual(res["news_details"]["GOOGL"]["status"], "SKIPPED_DUE_TO_QUOTA_LIMIT")
        self.assertEqual(res["news_details"]["AMZN"]["status"], "SKIPPED_DUE_TO_QUOTA_LIMIT")

    # 7. Duplicate articles do not increase the stored article count
    def test_07_duplicate_articles_do_not_increase_count(self):
        articles = [
            {
                "News_ID": "REAL_AAPL_DEDUP_TEST_01",
                "Headline": "Apple Announces Breakthrough AI Architecture",
                "Summary": "Summary text for test",
                "Ticker": "AAPL",
                "Published_At_UTC": "2025-01-20T10:00:00Z",
                "Source": "Reuters",
                "URL": "https://example.com/test-article-1",
                "Dataset_Type": "REAL"
            }
        ]

        added_1, dup_1, rej_1 = store_articles(articles)
        self.assertEqual(added_1, 1)
        self.assertEqual(dup_1, 0)

        # Store identical article again
        added_2, dup_2, rej_2 = store_articles(articles)
        self.assertEqual(added_2, 0)
        self.assertEqual(dup_2, 1)

        stored = get_stored_articles()
        self.assertEqual(len(stored), 1, "Stored count must remain 1 after duplicate insertion")

    # 8. A zero-new-article refresh is reported accurately
    @patch("src.analytics.refresh_manager.fetch_latest_news_for_ticker")
    def test_08_zero_new_article_refresh_reported_accurately(self, mock_news):
        mock_news.return_value = {
            "ticker": "AAPL",
            "status": "SUCCESS",
            "raw_fetched": 5,
            "added": 0,
            "duplicates": 5,
            "rejected": 0
        }

        res = execute_real_data_refresh(
            scope="News only",
            tickers=["AAPL"],
            rebuild_analytics=False,
            log_path=self.test_log_path,
            market_save_path=self.test_market_path
        )

        self.assertEqual(res["articles_added"], 0)
        self.assertEqual(res["articles_duplicate"], 5)
        self.assertEqual(res["news_refreshed_count"], 1)
        self.assertTrue(res["news_success"])
        self.assertEqual(res["status"], "SUCCESS")

    # 9. Cache invalidation and dashboard reload expose newly stored articles
    def test_09_cache_invalidation_and_reload(self):
        import dashboard.app as app_mod
        test_db = self.temp_path / "news_database.db"

        with patch.object(app_mod, "LIVE_DB_PATH", test_db):
            # Initial query for AAPL in empty test database
            app_mod.load_latest_real_articles.clear()
            initial_df = app_mod.load_latest_real_articles("AAPL", limit=10)
            initial_count = len(initial_df) if initial_df is not None else 0
            self.assertEqual(initial_count, 0)

            # Insert a new article directly into temp storage
            new_article = [{
                "News_ID": "REAL_AAPL_CACHE_TEST_01",
                "Headline": "Apple Cache Invalidation Verified Article",
                "Summary": "Testing cache clearing",
                "Ticker": "AAPL",
                "Published_At_UTC": "2025-01-22T12:00:00Z",
                "Source": "Bloomberg",
                "URL": "https://example.com/cache-test-1",
                "Dataset_Type": "REAL"
            }]
            store_articles(new_article)

            # Store matching nlp result so JOIN in load_latest_real_articles succeeds
            import sqlite3
            conn = sqlite3.connect(test_db)
            c = conn.cursor()
            c.execute("""
                INSERT OR REPLACE INTO nlp_results (
                    news_id, ticker, published_at_utc, sentiment_label, sentiment_score,
                    relevance_score, event_type, importance_score, recency_score, triage_score,
                    processed_at_utc, dataset_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                "REAL_AAPL_CACHE_TEST_01", "AAPL", "2025-01-22T12:00:00Z", "Positive", 0.85,
                0.90, "Product", 0.80, 0.95, 0.68, "2025-01-22T12:05:00Z", "REAL"
            ))
            conn.commit()
            conn.close()

            # Invalidate cache
            app_mod.load_latest_real_articles.clear()

            # Reload after cache clearing
            updated_df = app_mod.load_latest_real_articles("AAPL", limit=10)
            self.assertIsNotNone(updated_df)
            self.assertEqual(len(updated_df), 1)
            self.assertEqual(updated_df.iloc[0]["headline"], "Apple Cache Invalidation Verified Article")

    # 10. API failures do not erase or corrupt existing data
    @patch("src.analytics.refresh_manager.fetch_latest_news_for_ticker")
    def test_10_api_failures_preserve_existing_data(self, mock_news):
        # Insert initial baseline article
        baseline_article = [{
            "News_ID": "REAL_AAPL_PRESERVE_TEST",
            "Headline": "Apple Baseline Preserved Article",
            "Summary": "Must not be deleted or corrupted",
            "Ticker": "AAPL",
            "Published_At_UTC": "2025-01-21T08:00:00Z",
            "Source": "Dow Jones",
            "URL": "https://example.com/preserve-test",
            "Dataset_Type": "REAL"
        }]
        store_articles(baseline_article)
        stored_before = get_stored_articles()
        self.assertEqual(len(stored_before), 1)

        # Mock API failure
        mock_news.side_effect = RuntimeError("Simulated API failure or network timeout")

        res = execute_real_data_refresh(
            scope="News only",
            tickers=["AAPL"],
            rebuild_analytics=False,
            log_path=self.test_log_path,
            market_save_path=self.test_market_path
        )

        self.assertFalse(res["news_success"])
        self.assertEqual(res["status"], "FAILED")

        # Verify existing data is completely intact
        stored_after = get_stored_articles()
        self.assertEqual(len(stored_after), 1)
        self.assertEqual(stored_after.iloc[0]["headline"], "Apple Baseline Preserved Article")


if __name__ == "__main__":
    unittest.main()
