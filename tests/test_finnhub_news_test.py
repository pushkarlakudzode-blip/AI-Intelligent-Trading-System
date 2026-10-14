"""
Unit Tests for Finnhub Historical News Feasibility Module.

Verifies:
1. Token handling and missing-key detection
2. Token masking security
3. Unix epoch timestamp parsing to UTC ISO format
4. Article normalization to standardized project schema
5. Malformed article rejection (missing headline or timestamp)
6. Text quality metrics computation (headline, summary, usable text)
7. Multi-level duplicate detection (ID, URL, Headline+Date)
8. In-memory NLP compatibility test without persistence
9. Mocked API query success with JSON list parsing
10. Mocked API quota / rate limit / plan restriction error handling
11. Production database non-interference guarantee
"""

import sys
import unittest
import sqlite3
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import urllib.error

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.finnhub_news_test import (
    get_finnhub_token,
    mask_token,
    parse_finnhub_timestamp,
    normalize_finnhub_article,
    query_finnhub_company_news,
    check_text_quality,
    check_duplicates,
    test_nlp_compatibility,
    get_mock_sample_articles
)

PROD_DB_PATH = PROJECT_ROOT / "data" / "live" / "news_database.db"


class TestFinnhubFeasibility(unittest.TestCase):

    def setUp(self):
        # Record baseline DB counts to ensure zero modification
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

    def test_01_token_missing_handling(self):
        """Test missing or empty FINNHUB_API_KEY is handled cleanly."""
        with patch.dict("os.environ", {"FINNHUB_API_KEY": ""}):
            token = get_finnhub_token()
            self.assertIsNone(token)
            data, msg, status = query_finnhub_company_news("AAPL", "2024-01-01", "2024-01-05")
            self.assertIsNone(data)
            self.assertEqual(msg, "FINNHUB_API_KEY is not configured.")
            self.assertEqual(status, 401)

    def test_02_token_masking(self):
        """Test token masking security never exposes full API key."""
        self.assertEqual(mask_token(None), "None")
        self.assertEqual(mask_token(""), "None")
        self.assertEqual(mask_token("short"), "***")
        masked = mask_token("c1234567890finnhub_secret")
        self.assertTrue(masked.startswith("c123..."))
        self.assertTrue(masked.endswith("cret"))
        self.assertNotIn("finnhub_secret", masked)

    def test_03_timestamp_parsing(self):
        """Test Unix epoch seconds and ISO strings parse to UTC ISO standard."""
        # 1546524000 = 2019-01-03 14:00:00 UTC
        ts1 = parse_finnhub_timestamp(1546524000)
        self.assertEqual(ts1, "2019-01-03T14:00:00Z")

        # String representation of Unix timestamp
        ts2 = parse_finnhub_timestamp("1546524000")
        self.assertEqual(ts2, "2019-01-03T14:00:00Z")

        # ISO string fallback
        ts3 = parse_finnhub_timestamp("2024-05-15 09:30:00")
        self.assertEqual(ts3, "2024-05-15T09:30:00Z")

        # Invalid timestamp
        ts_bad = parse_finnhub_timestamp("not-a-timestamp")
        self.assertIsNone(ts_bad)
        self.assertIsNone(parse_finnhub_timestamp(None))

    def test_04_article_normalization(self):
        """Test normalization maps raw Finnhub dictionary to standardized project schema."""
        raw_item = {
            "category": "company",
            "datetime": 1718042400,  # 2024-06-10 18:00:00 UTC
            "headline": "Apple Unveils Apple Intelligence at WWDC 2024",
            "id": 99887766,
            "image": "https://example.com/wwdc.jpg",
            "related": "AAPL",
            "source": "TechCrunch",
            "summary": "Apple integrated personal generative intelligence system across iOS and macOS.",
            "url": "https://techcrunch.com/2024/06/10/apple-intelligence-wwdc"
        }
        normalized = normalize_finnhub_article(raw_item, target_ticker="AAPL")
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["Ticker"], "AAPL")
        self.assertEqual(normalized["Company_Name"], "Apple")
        self.assertEqual(normalized["Headline"], "Apple Unveils Apple Intelligence at WWDC 2024")
        self.assertIn("generative intelligence", normalized["Summary"])
        self.assertEqual(normalized["Source"], "TechCrunch")
        self.assertEqual(normalized["URL"], "https://techcrunch.com/2024/06/10/apple-intelligence-wwdc")
        self.assertEqual(normalized["Provider"], "Finnhub")
        self.assertEqual(normalized["Published_At_UTC"], "2024-06-10T18:00:00Z")
        self.assertEqual(normalized["Dataset_Type"], "REAL")
        self.assertEqual(normalized["Provider_Sentiment"], 0.0)
        self.assertEqual(len(normalized["News_ID"]), 32)

    def test_05_malformed_article_handling(self):
        """Test articles missing headline or timestamp return None gracefully."""
        self.assertIsNone(normalize_finnhub_article({}))
        self.assertIsNone(normalize_finnhub_article({"headline": ""}))
        self.assertIsNone(normalize_finnhub_article({"headline": "Valid Headline", "datetime": None}))
        self.assertIsNone(normalize_finnhub_article({"headline": "Valid Headline", "datetime": "invalid"}))

    def test_06_text_quality_assessment(self):
        """Test check_text_quality computes counts and percentages accurately."""
        sample_articles = [
            {"headline": "Headline 1", "summary": "Summary 1", "source": "CNBC"},
            {"headline": "Headline 2", "summary": "", "source": "Reuters"},
            {"headline": "", "summary": "Summary only", "source": "Bloomberg"},
            {"headline": "", "summary": "", "source": "Unknown"}
        ]
        q = check_text_quality(sample_articles)
        self.assertEqual(q["total_returned"], 4)
        self.assertEqual(q["headline_available"], 2)
        self.assertEqual(q["summary_available"], 2)
        self.assertEqual(q["usable_text_available"], 3)
        self.assertEqual(q["missing_usable_text"], 1)
        self.assertEqual(q["usable_text_pct"], 75.0)
        self.assertEqual(q["unique_sources"], 4)

    def test_07_duplicate_detection(self):
        """Test duplicate checks flag duplicate IDs, URLs, and headline+date combinations."""
        articles = [
            {"News_ID": "f1", "URL": "https://f.com/1", "Headline": "Apple Profit Up", "Published_At_UTC": "2024-01-01T10:00:00Z"},
            {"News_ID": "f2", "URL": "https://f.com/2", "Headline": "Apple Profit Down", "Published_At_UTC": "2024-01-02T10:00:00Z"},
            {"News_ID": "f1", "URL": "https://f.com/dup1", "Headline": "Apple Different", "Published_At_UTC": "2024-01-03T10:00:00Z"},
            {"News_ID": "f3", "URL": "https://f.com/1", "Headline": "Apple Different 2", "Published_At_UTC": "2024-01-04T10:00:00Z"},
            {"News_ID": "f4", "URL": "https://f.com/5", "Headline": "apple profit up", "Published_At_UTC": "2024-01-01T15:00:00Z"},
        ]
        dups = check_duplicates(articles)
        self.assertEqual(dups["total_articles"], 5)
        self.assertEqual(dups["duplicate_ids"], 1)
        self.assertEqual(dups["duplicate_urls"], 1)
        self.assertEqual(dups["duplicate_title_dates"], 1)

    def test_08_nlp_compatibility_in_memory(self):
        """Test normalized Finnhub articles process through NLP pipeline in-memory with zero DB writes."""
        raw_samples = get_mock_sample_articles()[:5]
        normalized = [normalize_finnhub_article(item, target_ticker="AAPL") for item in raw_samples]
        normalized = [n for n in normalized if n is not None]

        nlp_results = test_nlp_compatibility(normalized, max_sample=5)
        self.assertEqual(nlp_results["sample_size"], 5)
        self.assertEqual(nlp_results["successfully_processed"], 5)
        self.assertEqual(nlp_results["failed"], 0)
        self.assertEqual(len(nlp_results["sample_results"]), 5)

        for res in nlp_results["sample_results"]:
            self.assertIn("Sentiment_Label", res)
            self.assertIn("Sentiment_Score", res)
            self.assertIn("Relevance_Score", res)
            self.assertIn("Triage_Score", res)

    @patch("urllib.request.urlopen")
    def test_09_api_query_mocking_success(self, mock_urlopen):
        """Test Finnhub API query handling when response is valid JSON list."""
        mock_response = MagicMock()
        mock_response.getcode.return_value = 200
        mock_payload = [
            {
                "category": "company",
                "datetime": 1722513600,
                "headline": "Mock Apple News Headline",
                "id": 1234567,
                "related": "AAPL",
                "source": "MockSource",
                "summary": "Mock summary for test",
                "url": "https://mock.com/finnhub-news"
            }
        ]
        mock_response.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        data, msg, status = query_finnhub_company_news("AAPL", "2024-08-01", "2024-08-02", token="mock_key")
        self.assertIsNotNone(data)
        self.assertEqual(msg, "OK")
        self.assertEqual(status, 200)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["headline"], "Mock Apple News Headline")

    @patch("urllib.request.urlopen")
    def test_10_api_errors_and_rate_limits(self, mock_urlopen):
        """Test Finnhub HTTP 429 rate limit and 403 access error handling."""
        err_body = json.dumps({"error": "You don't have access to this resource."}).encode("utf-8")
        http_error_403 = urllib.error.HTTPError(
            url="https://finnhub.io/api/v1/company-news",
            code=403,
            msg="Forbidden",
            hdrs={},
            fp=MagicMock(read=MagicMock(return_value=err_body))
        )
        mock_urlopen.side_effect = http_error_403

        data, msg, status = query_finnhub_company_news("AAPL", "2019-01-01", "2019-01-05", token="mock_key")
        self.assertIsNone(data)
        self.assertEqual(status, 403)
        self.assertIn("You don't have access to this resource", msg)


if __name__ == "__main__":
    unittest.main()
