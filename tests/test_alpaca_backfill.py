"""
Unit Tests for Alpaca Historical News Staging Backfill Module.

Verifies:
1. Staging directory hierarchy initialization
2. Checkpoint manifest serialization, saving, and resumption
3. Meta Platforms (FB / META) canonical normalization
4. Lookahead protection (future-dated rejection)
5. Malformed record rejection
6. Multi-level deduplication (ID, URL, Headline+Date)
7. Data quality metrics computation
8. Cursor pagination and segment checkpoint resumption
9. HTTP 429 rate limit backoff and retry handling
10. In-memory and staging-only NLP scoring and daily signal generation
11. Absolute production database non-interference guarantee
"""

import os
import sys
import json
import unittest
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import urllib.error

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.alpaca_backfill_manager import (
    init_staging_directories,
    load_manifest,
    save_manifest,
    normalize_staging_article,
    deduplicate_articles,
    validate_staging_quality,
    fetch_segment_articles,
    query_alpaca_news_robust,
    STAGING_ROOT,
    RAW_DIR,
    NORMALIZED_DIR,
    CHECKPOINTS_DIR,
    REPORTS_DIR,
    MANIFEST_PATH
)
from src.nlp.news_nlp_pipeline import process_articles_nlp
import pandas as pd

PROD_DB_PATH = PROJECT_ROOT / "data" / "live" / "news_database.db"


class TestAlpacaBackfill(unittest.TestCase):

    def setUp(self):
        # Record baseline DB counts
        if PROD_DB_PATH.exists():
            conn = sqlite3.connect(PROD_DB_PATH)
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

    def tearDown(self):
        # Strict safety check: verify database remained completely untouched
        if PROD_DB_PATH.exists():
            conn = sqlite3.connect(PROD_DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM articles")
            current_articles = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM nlp_results")
            current_nlp = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM validation_logs")
            current_val = cur.fetchone()[0]
            conn.close()

            self.assertEqual(self.baseline_articles, current_articles, "Production articles count modified!")
            self.assertEqual(self.baseline_nlp, current_nlp, "Production nlp_results count modified!")
            self.assertEqual(self.baseline_val, current_val, "Production validation_logs count modified!")

    def test_01_staging_directory_structure(self):
        """Verify staging directory hierarchy is initialized properly."""
        dirs = init_staging_directories()
        self.assertTrue(dirs["raw"].exists())
        self.assertTrue(dirs["normalized"].exists())
        self.assertTrue(dirs["checkpoints"].exists())
        self.assertTrue(dirs["reports"].exists())
        self.assertTrue(dirs["logs"].exists())

    def test_02_manifest_checkpoint_lifecycle(self):
        """Verify checkpoint manifest reading and writing."""
        manifest = load_manifest()
        self.assertIn("completed_segments", manifest)
        self.assertIn("summary", manifest)

        # Test writing a dummy segment
        manifest["completed_segments"]["TEST_2020"] = {
            "ticker": "TEST",
            "year": 2020,
            "articles_count": 50,
            "completed_at_utc": datetime.now(timezone.utc).isoformat()
        }
        save_manifest(manifest)

        reloaded = load_manifest()
        self.assertIn("TEST_2020", reloaded["completed_segments"])
        self.assertEqual(reloaded["completed_segments"]["TEST_2020"]["articles_count"], 50)

        # Cleanup test entry
        del reloaded["completed_segments"]["TEST_2020"]
        save_manifest(reloaded)

    def test_03_normalization_and_meta_fb_handling(self):
        """Verify article normalization and FB to META canonical mapping."""
        # Pre-2022 Facebook article
        raw_fb = {
            "id": 5555,
            "headline": "Facebook Announces AI Research Innovations",
            "created_at": "2020-05-10T12:00:00Z",
            "summary": "Facebook open sources cutting-edge natural language processing models.",
            "url": "https://benzinga.com/fb-ai",
            "source": "Benzinga"
        }
        norm_fb = normalize_staging_article(raw_fb, target_ticker="FB")
        self.assertIsNotNone(norm_fb)
        self.assertEqual(norm_fb["Ticker"], "META")
        self.assertEqual(norm_fb["Company_Name"], "Meta Platforms")
        self.assertEqual(norm_fb["Published_At_UTC"], "2020-05-10T12:00:00Z")
        self.assertEqual(norm_fb["Dataset_Type"], "REAL")
        self.assertEqual(len(norm_fb["News_ID"]), 32)

        # Post-2022 Meta article
        raw_meta = {
            "id": 6666,
            "headline": "Meta Launches Llama 3 Open Weights",
            "created_at": "2024-04-18T14:00:00Z",
            "summary": "Meta Platforms unveils next-generation foundation models.",
            "url": "https://benzinga.com/meta-llama",
            "source": "Benzinga"
        }
        norm_meta = normalize_staging_article(raw_meta, target_ticker="META")
        self.assertIsNotNone(norm_meta)
        self.assertEqual(norm_meta["Ticker"], "META")
        self.assertEqual(norm_meta["Company_Name"], "Meta Platforms")

    def test_04_future_dated_lookahead_rejection(self):
        """Verify future-dated articles are rejected."""
        future_art = {
            "id": 9999,
            "headline": "Future Event in 2030",
            "created_at": "2030-01-01T00:00:00Z",
            "summary": "Future news summary.",
            "url": "https://example.com/future"
        }
        self.assertIsNone(normalize_staging_article(future_art, "AAPL"))

    def test_05_malformed_article_rejection(self):
        """Verify articles missing headlines or timestamps return None."""
        self.assertIsNone(normalize_staging_article({"id": 1, "headline": ""}, "MSFT"))
        self.assertIsNone(normalize_staging_article({"id": 2, "headline": "Valid", "created_at": None}, "MSFT"))

    def test_06_multi_level_deduplication(self):
        """Verify deduplication detects duplicate IDs, URLs, and title+dates."""
        articles = [
            {"Raw_ID": "101", "News_ID": "N1", "URL": "http://a.com/1", "Headline": "Apple Record Q1", "Published_At_UTC": "2020-01-02T10:00:00Z"},
            {"Raw_ID": "101", "News_ID": "N2", "URL": "http://a.com/2", "Headline": "Different Title", "Published_At_UTC": "2020-01-02T10:00:00Z"}, # Dup ID
            {"Raw_ID": "102", "News_ID": "N3", "URL": "http://a.com/1", "Headline": "Another Title", "Published_At_UTC": "2020-01-02T10:00:00Z"}, # Dup URL
            {"Raw_ID": "103", "News_ID": "N4", "URL": "http://a.com/3", "Headline": "Apple Record Q1", "Published_At_UTC": "2020-01-02T15:00:00Z"}, # Dup title+date
            {"Raw_ID": "104", "News_ID": "N5", "URL": "http://a.com/4", "Headline": "Unique Article", "Published_At_UTC": "2020-01-03T10:00:00Z"}  # Unique
        ]
        unique, stats = deduplicate_articles(articles)
        self.assertEqual(len(unique), 2)
        self.assertEqual(stats["total_duplicates_removed"], 3)
        self.assertEqual(stats["duplicate_ids"], 1)
        self.assertEqual(stats["duplicate_urls"], 1)
        self.assertEqual(stats["duplicate_title_dates"], 1)

    def test_07_quality_validation_metrics(self):
        """Verify validate_staging_quality accurately calculates metrics."""
        sample_articles = [
            {"Ticker": "AAPL", "Published_At_UTC": "2019-01-02T10:00:00Z", "Headline": "H1", "Summary": "S1", "Has_Full_Content": True, "News_ID": "1", "Raw_ID": "1", "URL": "u1"},
            {"Ticker": "AAPL", "Published_At_UTC": "2020-06-15T10:00:00Z", "Headline": "H2", "Summary": "S2", "Has_Full_Content": False, "News_ID": "2", "Raw_ID": "2", "URL": "u2"},
            {"Ticker": "META", "Published_At_UTC": "2021-12-01T10:00:00Z", "Headline": "H3", "Summary": "", "Has_Full_Content": True, "News_ID": "3", "Raw_ID": "3", "URL": "u3"}
        ]
        q = validate_staging_quality(sample_articles, raw_articles_count=4)
        self.assertEqual(q["total_unique"], 3)
        self.assertEqual(q["duplicates_removed"], 1)
        self.assertEqual(q["usable_text_pct"], 100.0)
        self.assertEqual(q["records_per_company"]["AAPL"], 2)
        self.assertEqual(q["records_per_company"]["META"], 1)
        self.assertEqual(q["meta_normalized_count"], 1)
        self.assertEqual(q["outside_range"], 0)
        self.assertEqual(q["future_dated"], 0)

    def test_08_pagination_and_checkpoint_resume(self):
        """Verify fetch_segment_articles follows page_token and reloads from cache."""
        page1_data = {
            "news": [{"id": 1, "headline": "Page 1 Item", "created_at": "2019-01-02T10:00:00Z"}],
            "next_page_token": "token_page2"
        }
        page2_data = {
            "news": [{"id": 2, "headline": "Page 2 Item", "created_at": "2019-01-03T10:00:00Z"}],
            "next_page_token": None
        }

        call_count = [0]

        def mock_query(params, **kwargs):
            call_count[0] += 1
            if params.get("page_token") == "token_page2":
                return page2_data, "OK", 200, {}
            return page1_data, "OK", 200, {}

        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("src.data.alpaca_backfill_manager.RAW_DIR", Path(tmp_dir)):
                with patch("src.data.alpaca_backfill_manager.query_alpaca_news_robust", side_effect=mock_query):
                    articles, stats = fetch_segment_articles(
                        ticker="TEST_TICKER",
                        year=2019,
                        query_symbol="TEST_TICKER",
                        rate_delay=0.01,
                        force_refresh=True
                    )
                    self.assertEqual(len(articles), 2)
                    self.assertEqual(stats["pages_count"], 2)
                    self.assertEqual(call_count[0], 2)

                    # Second call should resume from cache without invoking mock_query
                    articles2, stats2 = fetch_segment_articles(
                        ticker="TEST_TICKER",
                        year=2019,
                        query_symbol="TEST_TICKER",
                        rate_delay=0.01,
                        force_refresh=False
                    )
                    self.assertEqual(len(articles2), 2)
                    self.assertTrue(stats2["resumed_from_cache"])
                    # call_count should still be 2 (no extra query)
                    self.assertEqual(call_count[0], 2)

    def test_09_rate_limit_429_backoff(self):
        """Verify query_alpaca_news_robust handles HTTP 429 with backoff."""
        http_err_429 = urllib.error.HTTPError(
            url="https://data.alpaca.markets/v1beta1/news",
            code=429,
            msg="Too Many Requests",
            hdrs={"Retry-After": "0.1"},
            fp=MagicMock(read=lambda: b'{"message": "rate limit exceeded"}')
        )
        success_response = MagicMock()
        success_response.getcode.return_value = 200
        success_response.read.return_value = json.dumps({"news": []}).encode("utf-8")
        success_response.headers = {}
        success_response.__enter__.return_value = success_response

        # Sequence: 429 once, then success 200
        with patch("urllib.request.urlopen", side_effect=[http_err_429, success_response]):
            data, msg, status, _ = query_alpaca_news_robust(
                {"symbols": "AAPL"},
                key_id="KEY",
                secret_key="SECRET"
            )
            self.assertEqual(status, 200)
            self.assertEqual(msg, "OK")

    def test_10_nlp_scoring_staging_execution(self):
        """Verify NLP scoring executes and formats daily news signals without DB writes."""
        test_articles = [
            {
                "News_ID": "N1",
                "Ticker": "NVDA",
                "Company_Name": "NVIDIA",
                "Published_At_UTC": "2023-05-24T18:00:00Z",
                "Headline": "NVIDIA Reports Record Revenue and Strong Profit Growth",
                "Summary": "NVIDIA accelerates AI chip sales across global enterprise datacenter customers."
            },
            {
                "News_ID": "N2",
                "Ticker": "NVDA",
                "Company_Name": "NVIDIA",
                "Published_At_UTC": "2023-05-24T20:00:00Z",
                "Headline": "Analysts Upgrade NVIDIA Price Target Following Milestone Beat",
                "Summary": "Wall Street analysts lift forward estimates on unprecedented hardware demand."
            }
        ]
        df_in = pd.DataFrame(test_articles)
        enriched, daily = process_articles_nlp(df_in)

        self.assertEqual(len(enriched), 2)
        self.assertIn("Sentiment_Score", enriched.columns)
        self.assertIn("Triage_Score", enriched.columns)
        self.assertEqual(enriched["Sentiment_Label"].iloc[0], "Positive")

        self.assertEqual(len(daily), 1)
        self.assertEqual(daily["Ticker"].iloc[0], "NVDA")
        self.assertEqual(daily["News_Count"].iloc[0], 2)
        self.assertGreater(daily["Average_Sentiment"].iloc[0], 0.0)


if __name__ == "__main__":
    unittest.main()
