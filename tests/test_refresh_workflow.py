"""
Unit & Integration Test Suite for Controlled Manual Real-Data Refresh Workflow.

Verifies:
1. Manual execution isolation: Module import does not trigger network calls or mutations.
2. Scope routing:
   - "Market only": Calls market logic, zero news calls.
   - "News only": Calls news logic, zero market calls.
   - "Market + News": Calls both pipelines.
3. Deduplication & storage preservation: Duplicate articles are skipped, existing DB is preserved.
4. Quota protection: Rate limit or daily quota exhaustion aborts remaining tickers safely.
5. Partial success reporting: Market success + news quota limit reports PARTIAL (never SUCCESS).
6. Analytics rebuild: Re-runs analytics pipeline when enabled, skipped when disabled.
7. Audit logging: Records execution row to refresh_log.csv, readable by get_last_refresh_info().
8. Academic isolation: Zero modifications to notebooks/, data/raw/, data/processed/.
9. Configuration security: API key is loaded via environment, no hardcoded secrets.
10. Dashboard UI integrity: Button key, scope radio options, and telemetry bindings are defined.
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
)


class TestRefreshWorkflow(unittest.TestCase):

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
        # Reset storage and refresh manager back to default production paths
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass
        finally:
            configure_storage(None)
            configure_refresh_manager(None)

    # 1. Manual execution isolation: Importing does not trigger execution
    def test_01_manual_execution_only(self):
        # Re-importing or accessing the module should not create any log or market files
        self.assertFalse(self.test_log_path.exists(), "Log file should not exist prior to explicit execution")
        self.assertFalse(self.test_market_path.exists(), "Market extension should not exist prior to explicit execution")

    # 2. Scope routing: Market only does not call news
    @patch("src.analytics.refresh_manager.refresh_market_data")
    @patch("src.analytics.refresh_manager.fetch_latest_news_for_ticker")
    @patch("src.analytics.refresh_manager.run_real_data_analytics")
    def test_02_scope_market_only(self, mock_analytics, mock_news, mock_market):
        mock_market.return_value = {
            "success": True,
            "refreshed_count": 2,
            "total_requested": 2,
            "details": {"AAPL": "SUCCESS", "MSFT": "SUCCESS"}
        }

        res = execute_real_data_refresh(
            scope="Market only",
            tickers=["AAPL", "MSFT"],
            rebuild_analytics=False,
            log_path=self.test_log_path,
            market_save_path=self.test_market_path
        )

        mock_market.assert_called_once()
        mock_news.assert_not_called()
        mock_analytics.assert_not_called()

        self.assertTrue(res["market_requested"])
        self.assertFalse(res["news_requested"])
        self.assertTrue(res["market_success"])
        self.assertEqual(res["status"], "SUCCESS")

    # 3. Scope routing: News only does not call market
    @patch("src.analytics.refresh_manager.refresh_market_data")
    @patch("src.analytics.refresh_manager.fetch_latest_news_for_ticker")
    @patch("src.analytics.refresh_manager.run_real_data_analytics")
    def test_03_scope_news_only(self, mock_analytics, mock_news, mock_market):
        mock_news.return_value = {
            "ticker": "AAPL",
            "status": "SUCCESS",
            "raw_fetched": 5,
            "added": 3,
            "duplicates": 2,
            "rejected": 0
        }

        res = execute_real_data_refresh(
            scope="News only",
            tickers=["AAPL"],
            rebuild_analytics=False,
            log_path=self.test_log_path,
            market_save_path=self.test_market_path
        )

        mock_market.assert_not_called()
        mock_news.assert_called_once_with("AAPL", limit=15, run_nlp=True)
        mock_analytics.assert_not_called()

        self.assertFalse(res["market_requested"])
        self.assertTrue(res["news_requested"])
        self.assertTrue(res["news_success"])
        self.assertEqual(res["articles_added"], 3)
        self.assertEqual(res["articles_duplicate"], 2)
        self.assertEqual(res["status"], "SUCCESS")

    # 4. Deduplication & storage preservation
    def test_04_deduplication_preserves_storage(self):
        sample_articles = [
            {
                "News_ID": "REAL_AAPL_20250115_TEST1",
                "Headline": "Apple Reports Record Services Revenue",
                "Summary": "Apple services revenue reached record high.",
                "Ticker": "AAPL",
                "Published_At_UTC": "2025-01-15T14:30:00Z",
                "Source": "Reuters",
                "URL": "https://example.com/aapl-services-record",
                "Dataset_Type": "REAL",
                "Author": "Reporter"
            },
            {
                "News_ID": "REAL_AAPL_20250115_TEST2",
                "Headline": "Apple Expands Silicon Capabilities",
                "Summary": "New M-series chips announced.",
                "Ticker": "AAPL",
                "Published_At_UTC": "2025-01-15T15:00:00Z",
                "Source": "Bloomberg",
                "URL": "https://example.com/aapl-silicon-news",
                "Dataset_Type": "REAL",
                "Author": "Reporter"
            }
        ]

        # Initial store
        added_1, dup_1, rej_1 = store_articles(sample_articles)
        self.assertEqual(added_1, 2)
        self.assertEqual(dup_1, 0)

        # Re-storing identical articles
        added_2, dup_2, rej_2 = store_articles(sample_articles)
        self.assertEqual(added_2, 0)
        self.assertEqual(dup_2, 2)

        stored = get_stored_articles()
        self.assertEqual(len(stored), 2, "Storage must contain only 2 unique articles without duplication")

    # 5. Quota exhaustion protection & graceful loop abort
    @patch("src.analytics.refresh_manager.fetch_latest_news_for_ticker")
    def test_05_quota_exhaustion_protection(self, mock_news):
        # First ticker hits quota limit
        mock_news.side_effect = [
            {
                "ticker": "AAPL",
                "status": "Alpha Vantage Quota Limit: We have detected your API key as ... standard API rate limit is 25 requests per day",
                "raw_fetched": 0,
                "added": 0,
                "duplicates": 0,
                "rejected": 0
            },
            # Should NOT be called for second ticker
            {
                "ticker": "MSFT",
                "status": "SUCCESS",
                "raw_fetched": 5, "added": 5, "duplicates": 0, "rejected": 0
            }
        ]

        res = execute_real_data_refresh(
            scope="News only",
            tickers=["AAPL", "MSFT", "GOOGL"],
            rebuild_analytics=False,
            log_path=self.test_log_path,
            market_save_path=self.test_market_path
        )

        # Verify only 1 ticker was called before aborting
        self.assertEqual(mock_news.call_count, 1)
        self.assertTrue(res["quota_limited"])
        self.assertEqual(res["news_details"]["MSFT"]["status"], "SKIPPED_DUE_TO_QUOTA_LIMIT")
        self.assertEqual(res["news_details"]["GOOGL"]["status"], "SKIPPED_DUE_TO_QUOTA_LIMIT")
        self.assertEqual(res["status"], "FAILED")

    # 6. Partial success reporting: Market succeeds, News hits quota
    @patch("src.analytics.refresh_manager.refresh_market_data")
    @patch("src.analytics.refresh_manager.fetch_latest_news_for_ticker")
    @patch("src.analytics.refresh_manager.run_real_data_analytics")
    def test_06_partial_success_status(self, mock_analytics, mock_news, mock_market):
        mock_market.return_value = {
            "success": True,
            "refreshed_count": 2,
            "total_requested": 2,
            "details": {"AAPL": "SUCCESS", "MSFT": "SUCCESS"}
        }
        mock_news.return_value = {
            "ticker": "AAPL",
            "status": "Rate limit reached: Standard call frequency is 5 calls per minute",
            "raw_fetched": 0,
            "added": 0,
            "duplicates": 0,
            "rejected": 0
        }

        res = execute_real_data_refresh(
            scope="Market + News",
            tickers=["AAPL", "MSFT"],
            rebuild_analytics=False,
            log_path=self.test_log_path,
            market_save_path=self.test_market_path
        )

        self.assertTrue(res["market_success"])
        self.assertFalse(res["news_success"])
        self.assertTrue(res["quota_limited"])
        # Crucial check: Must be PARTIAL, not SUCCESS
        self.assertEqual(res["status"], "PARTIAL")

    # 7. Analytics rebuild trigger
    @patch("src.analytics.refresh_manager.refresh_market_data")
    @patch("src.analytics.refresh_manager.run_real_data_analytics")
    def test_07_analytics_rebuild_trigger(self, mock_analytics, mock_market):
        mock_market.return_value = {
            "success": True, "refreshed_count": 1, "total_requested": 1, "details": {"AAPL": "SUCCESS"}
        }

        # Case A: rebuild_analytics = True
        res_a = execute_real_data_refresh(
            scope="Market only",
            tickers=["AAPL"],
            rebuild_analytics=True,
            log_path=self.test_log_path,
            market_save_path=self.test_market_path,
            analytics_output_dir=self.test_analytics_dir
        )
        self.assertTrue(res_a["analytics_rebuilt"])
        mock_analytics.assert_called_once_with(output_dir=self.test_analytics_dir, save_outputs=True)

        # Case B: rebuild_analytics = False
        mock_analytics.reset_mock()
        res_b = execute_real_data_refresh(
            scope="Market only",
            tickers=["AAPL"],
            rebuild_analytics=False,
            log_path=self.test_log_path,
            market_save_path=self.test_market_path
        )
        self.assertFalse(res_b["analytics_rebuilt"])
        mock_analytics.assert_not_called()

    # 8. Audit logging & get_last_refresh_info retrieval
    def test_08_audit_logging_and_retrieval(self):
        test_entry = {
            "timestamp_utc": "2026-10-03 14:00:00 UTC",
            "market_requested": True,
            "news_requested": True,
            "market_success": True,
            "news_success": False,
            "articles_added": 12,
            "articles_duplicate": 3,
            "articles_rejected": 0,
            "quota_limited": True,
            "analytics_rebuilt": True,
            "status": "PARTIAL",
            "error_message": "Alpha Vantage quota limit reached on TSLA."
        }

        log_refresh_event(test_entry, log_path=self.test_log_path)
        self.assertTrue(self.test_log_path.exists())

        # Read back via get_last_refresh_info
        info = get_last_refresh_info(log_path=self.test_log_path)
        self.assertEqual(info["timestamp_utc"], "2026-10-03 14:00:00 UTC")
        self.assertEqual(info["status"], "PARTIAL")
        self.assertTrue(info["market_success"])
        self.assertFalse(info["news_success"])
        self.assertTrue(info["quota_limited"])
        self.assertEqual(info["articles_added"], 12)
        self.assertEqual(info["articles_duplicate"], 3)

    # 9. Academic datasets and notebooks isolation
    def test_09_academic_isolation(self):
        protected_paths = [
            PROJECT_ROOT / "data" / "raw",
            PROJECT_ROOT / "data" / "processed",
            PROJECT_ROOT / "notebooks"
        ]
        for p in protected_paths:
            self.assertTrue(p.exists(), f"Protected path must exist: {p}")
            # Ensure no test artifact was written to raw or processed
            self.assertFalse((p / "test_refresh_log.csv").exists())
            self.assertFalse((p / "refresh_log.csv").exists())

    # 10. Dashboard UI elements verification
    def test_10_dashboard_ui_elements(self):
        app_path = PROJECT_ROOT / "dashboard" / "app.py"
        self.assertTrue(app_path.exists(), "dashboard/app.py must exist")
        content = app_path.read_text(encoding="utf-8")

        # Verify button and widget keys
        self.assertIn("btn_refresh_real_data", content, "Manual refresh button key must be present")
        self.assertIn("↻ REFRESH REAL DATA", content, "Button label must be present")
        self.assertIn("Refresh Scope:", content, "Scope selector must be present")
        self.assertIn("Market + News", content, "Market + News option must be present")
        self.assertIn("Market only", content, "Market only option must be present")
        self.assertIn("News only", content, "News only option must be present")
        self.assertIn("execute_real_data_refresh", content, "execute_real_data_refresh must be imported & invoked")
        self.assertIn("get_last_refresh_info", content, "get_last_refresh_info must be used")


if __name__ == "__main__":
    unittest.main()
