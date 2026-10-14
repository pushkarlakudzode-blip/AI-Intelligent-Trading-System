"""
Unit & Integration Test Suite for Real-Time and Historical Data Layer.

Verifies:
1. Ticker validation
2. Article normalization
3. Deterministic News_ID generation
4. Duplicate detection and rejection
5. Timestamp normalization (Timezone-aware UTC)
6. Malformed article handling
7. NLP enrichment output schema and Dataset_Type = "REAL"
8. Lookahead protection (no future-dated articles accepted)
9. SQLite & CSV storage and retrieval
10. API failure handling with mock provider responses
"""

import sys
import unittest
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.data.data_storage import (
    generate_news_id,
    validate_article,
    store_articles,
    store_nlp_results,
    get_stored_articles,
    get_stored_nlp_results,
    init_storage,
    configure_storage,
    SUPPORTED_TICKERS
)
from src.data.realtime_market import _validate_ticker, get_current_quote
from src.data.historical_news import normalize_av_article, query_alpha_vantage_news
from src.nlp.news_nlp_pipeline import (
    compute_financial_sentiment,
    compute_company_relevance,
    compute_event_importance,
    compute_recency_score,
    compute_triage_score,
    process_articles_nlp
)


class TestLiveDataLayer(unittest.TestCase):

    def setUp(self):
        # Create an isolated temporary directory for test storage
        self.temp_dir = tempfile.TemporaryDirectory()
        configure_storage(self.temp_dir.name)
        init_storage()

    def tearDown(self):
        # Clean up temporary storage and restore production live storage path
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass
        finally:
            configure_storage(None)

    # 1. Ticker Validation Test
    def test_01_ticker_validation(self):
        for t in ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ"]:
            self.assertEqual(_validate_ticker(t), t)
            self.assertEqual(_validate_ticker(t.lower()), t)
            
        with self.assertRaises(ValueError):
            _validate_ticker("INVALID_TICKER_XYZ")

    # 2. Article Normalization Test
    def test_02_article_normalization(self):
        sample_raw = {
            "title": "Microsoft Launches New Azure Copilot Features for Enterprise",
            "url": "https://example.com/msft-azure-news",
            "time_published": "20240415T143000",
            "summary": "Microsoft Corporation announced updates to Azure and generative AI tools.",
            "source": "TechCrunch",
            "overall_sentiment_score": 0.35,
            "ticker_sentiment": [
                {"ticker": "MSFT", "relevance_score": "0.95", "ticker_sentiment_score": "0.42"}
            ]
        }
        norm = normalize_av_article(sample_raw, "MSFT")
        self.assertIsNotNone(norm)
        self.assertEqual(norm["Ticker"], "MSFT")
        self.assertEqual(norm["Company_Name"], "Microsoft")
        self.assertEqual(norm["Published_At_UTC"], "2024-04-15T14:30:00Z")
        self.assertEqual(norm["Provider"], "Alpha Vantage")
        self.assertEqual(norm["Dataset_Type"], "REAL")
        self.assertAlmostEqual(norm["Provider_Sentiment"], 0.42, places=2)
        self.assertAlmostEqual(norm["Provider_Relevance"], 0.95, places=2)

    # 3. Deterministic News_ID Generation Test
    def test_03_news_id_generation(self):
        id1 = generate_news_id("AAPL", "2024-05-10T12:00:00Z", "Headline A", "https://url.com/a")
        id2 = generate_news_id("AAPL", "2024-05-10T12:00:00Z", "Headline A", "https://url.com/a")
        id3 = generate_news_id("AAPL", "2024-05-10T12:00:00Z", "Headline B", "https://url.com/a")
        
        self.assertEqual(id1, id2, "Identical articles must yield identical News_IDs")
        self.assertNotEqual(id1, id3, "Different headlines must yield distinct News_IDs")
        self.assertEqual(len(id1), 32)

    # 4. Duplicate Detection Test
    def test_04_duplicate_detection(self):
        article = {
            "News_ID": generate_news_id("NVDA", "2024-06-01T10:00:00Z", "Nvidia Blackwell Shipments Begin", "https://example.com/nvda"),
            "Ticker": "NVDA",
            "Company_Name": "NVIDIA",
            "Published_At_UTC": "2024-06-01T10:00:00Z",
            "Headline": "Nvidia Blackwell Shipments Begin Worldwide",
            "Summary": "Semiconductor manufacturer ships AI accelerators.",
            "Source": "Reuters",
            "URL": "https://example.com/nvda",
            "Provider": "Alpha Vantage",
            "Dataset_Type": "REAL"
        }
        # First insertion
        added1, dups1, rejs1 = store_articles([article])
        self.assertEqual(rejs1, 0)
        self.assertEqual(added1, 1)
        self.assertEqual(dups1, 0)
        
        # Second insertion of same article
        added2, dups2, rejs2 = store_articles([article])
        self.assertEqual(added2, 0, "Duplicate article must not be added again")
        self.assertEqual(dups2, 1, "Duplicate must be counted as skipped")
        self.assertEqual(rejs2, 0)

    # 5. Timestamp Normalization Test
    def test_05_timestamp_normalization(self):
        sample = {
            "title": "Alphabet Unveils Gemini Advances at Google I/O",
            "url": "https://example.com/googl-gemini",
            "time_published": "20240514T170000",
            "summary": "Google leadership demonstrated new multimodal AI models."
        }
        norm = normalize_av_article(sample, "GOOGL")
        pub_dt = pd.to_datetime(norm["Published_At_UTC"])
        self.assertIsNotNone(pub_dt.tzinfo, "Timestamp must be timezone-aware")
        self.assertEqual(norm["Published_At_UTC"], "2024-05-14T17:00:00Z")

    # 6. Malformed Article Handling Test
    def test_06_malformed_article_handling(self):
        # Empty headline
        bad_art1 = {"News_ID": "1", "Ticker": "AAPL", "Published_At_UTC": "2024-01-01T00:00:00Z", "Headline": ""}
        valid1, reason1 = validate_article(bad_art1)
        self.assertFalse(valid1)
        self.assertIn("HEADLINE", reason1)

        # Missing timestamp
        bad_art2 = {"News_ID": "2", "Ticker": "AAPL", "Headline": "Apple Announces Product"}
        valid2, reason2 = validate_article(bad_art2)
        self.assertFalse(valid2)
        self.assertIn("TIMESTAMP", reason2)

        # Unsupported ticker
        bad_art3 = {"News_ID": "3", "Ticker": "FAKECO", "Published_At_UTC": "2024-01-01T00:00:00Z", "Headline": "Company News"}
        valid3, reason3 = validate_article(bad_art3)
        self.assertFalse(valid3)
        self.assertIn("UNSUPPORTED_TICKER", reason3)

        # Verify rejection in storage layer and validation report generation
        added, dups, rejs = store_articles([bad_art1, bad_art2, bad_art3])
        self.assertEqual(rejs, 3)
        self.assertEqual(added, 0)
        from src.data.data_storage import CSV_VALIDATION_PATH
        self.assertTrue(CSV_VALIDATION_PATH.exists())

    # 7. NLP Pipeline Output Schema & Dataset_Type = "REAL"
    def test_07_nlp_pipeline_schema(self):
        articles = pd.DataFrame([{
            "News_ID": "test_nlp_01",
            "Ticker": "TSLA",
            "Company_Name": "Tesla",
            "Published_At_UTC": "2024-03-20T11:00:00Z",
            "Headline": "Tesla Surges as Cybertruck Deliveries Exceed Expectations and Profit Rises",
            "Summary": "Tesla Inc beat quarterly delivery consensus with robust demand.",
            "Source": "Bloomberg",
            "URL": "https://example.com/tsla-surge"
        }])
        
        nlp_df, daily_df = process_articles_nlp(articles)
        self.assertFalse(nlp_df.empty)
        self.assertFalse(daily_df.empty)
        
        # Verify schema requirements
        self.assertEqual(nlp_df.iloc[0]["Dataset_Type"], "REAL")
        self.assertEqual(daily_df.iloc[0]["Dataset_Type"], "REAL")
        self.assertIn("Sentiment_Score", nlp_df.columns)
        self.assertIn("Relevance_Score", nlp_df.columns)
        self.assertIn("Importance_Score", nlp_df.columns)
        self.assertIn("Triage_Score", nlp_df.columns)
        self.assertIn("News_Signal", daily_df.columns)
        
        # Verify math
        self.assertEqual(nlp_df.iloc[0]["Sentiment_Label"], "Positive")
        self.assertGreater(nlp_df.iloc[0]["Sentiment_Score"], 0.0)
        self.assertAlmostEqual(nlp_df.iloc[0]["Relevance_Score"], 1.0, places=2)

        # Persist and verify in storage
        store_nlp_results(nlp_df, daily_df)
        stored_nlp = get_stored_nlp_results("TSLA")
        self.assertFalse(stored_nlp.empty)
        self.assertIn("test_nlp_01", stored_nlp["news_id"].values)

    # 8. Strict Lookahead Protection Test (No Future-Dated Articles)
    def test_08_no_future_dated_articles(self):
        future_pub = (datetime.now(timezone.utc) + timedelta(days=5)).isoformat()
        future_article = {
            "News_ID": "future_art_01",
            "Ticker": "AMZN",
            "Company_Name": "Amazon",
            "Published_At_UTC": future_pub,
            "Headline": "Amazon AWS Releases Cloud Architecture in Future Year",
            "Summary": "Future report.",
            "URL": "https://example.com/future"
        }
        is_valid, reason = validate_article(future_article)
        self.assertFalse(is_valid, "Future-dated articles must be rejected")
        self.assertIn("FUTURE_PUBLICATION_TIMESTAMP", reason)

    # 9. Storage & Retrieval Round-Trip Test
    def test_09_storage_retrieval(self):
        art_id = generate_news_id("JPM", "2024-02-15T09:30:00Z", "JPMorgan Chase Reports Robust Net Interest Income", "https://jpm.com/news")
        article = {
            "News_ID": art_id,
            "Ticker": "JPM",
            "Company_Name": "JPMorgan Chase",
            "Published_At_UTC": "2024-02-15T09:30:00Z",
            "Headline": "JPMorgan Chase Reports Robust Net Interest Income",
            "Summary": "CEO Jamie Dimon highlighted commercial banking strength.",
            "Source": "Wall Street Journal",
            "URL": "https://jpm.com/news",
            "Provider": "Alpha Vantage",
            "Dataset_Type": "REAL"
        }
        store_articles([article])
        df_retrieved = get_stored_articles("JPM")
        self.assertFalse(df_retrieved.empty)
        self.assertIn(art_id, df_retrieved["news_id"].values)

    # 10. API Failure Handling & Mocking Test
    @patch("urllib.request.urlopen")
    def test_10_api_failure_and_mock(self, mock_urlopen):
        # Mock rate limit response from Alpha Vantage
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"Note": "Thank you for using Alpha Vantage! Standard call frequency is 5 calls per minute."}'
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        # Temporarily mock api key
        with patch("src.data.historical_news.get_api_key", return_value="TEST_MOCK_KEY"):
            feed, status = query_alpha_vantage_news("AAPL", limit=10, max_retries=1)
            self.assertEqual(feed, [])
            self.assertIn("Rate limit", status)


if __name__ == "__main__":
    unittest.main()
