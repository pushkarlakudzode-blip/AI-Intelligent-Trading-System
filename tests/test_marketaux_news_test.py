"""
Unit Tests for MarketAux Historical News Feasibility Module.

Verifies:
1. Token handling and missing-token detection
2. Token masking security
3. Article normalization to standardized project schema
4. Timestamp parsing across variable ISO formats
5. Malformed article rejection (missing title or published_at)
6. Text quality metrics computation
7. Multi-level duplicate detection (ID, URL, Title+Date)
8. In-memory NLP compatibility test without persistence
9. Mocked API query success with meta.found parsing
10. Mocked API quota / rate limit / auth error handling
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

from src.data.marketaux_news_test import (
    get_marketaux_token,
    mask_token,
    parse_marketaux_timestamp,
    normalize_marketaux_article,
    query_marketaux,
    check_text_quality,
    check_duplicates,
    test_nlp_compatibility,
    get_mock_sample_articles
)

PROD_DB_PATH = PROJECT_ROOT / "data" / "live" / "news_database.db"


class TestMarketAuxFeasibility(unittest.TestCase):

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
        """Test missing or empty MARKETAUX_API_TOKEN is handled cleanly."""
        with patch.dict("os.environ", {"MARKETAUX_API_TOKEN": ""}):
            token = get_marketaux_token()
            self.assertIsNone(token)
            data, msg, status = query_marketaux({"symbols": "AAPL"})
            self.assertIsNone(data)
            self.assertEqual(msg, "MARKETAUX_API_TOKEN is not configured.")
            self.assertEqual(status, 401)

    def test_02_token_masking(self):
        """Test token masking security never exposes full API key."""
        self.assertEqual(mask_token(None), "None")
        self.assertEqual(mask_token(""), "None")
        self.assertEqual(mask_token("short"), "***")
        masked = mask_token("secret_marketaux_token_987654321")
        self.assertTrue(masked.startswith("secr..."))
        self.assertTrue(masked.endswith("4321"))
        self.assertNotIn("marketaux_token", masked)

    def test_03_article_normalization(self):
        """Test normalization maps raw MarketAux dictionary to standardized schema."""
        raw_item = {
            "uuid": "test-uuid-12345",
            "title": "Apple Introduces Next Generation M4 Silicon",
            "description": "Apple announced the M4 chip delivering industry-leading performance for AI.",
            "snippet": "New M4 architecture boosts neural engine capabilities.",
            "url": "https://example.com/apple-m4-announcement",
            "published_at": "2024-05-07T14:00:00.000000Z",
            "source": {"domain": "techcrunch.com", "name": "TechCrunch"},
            "entities": [
                {
                    "symbol": "AAPL",
                    "name": "Apple Inc.",
                    "match_score": 95.5,
                    "sentiment_score": 0.65
                }
            ]
        }
        normalized = normalize_marketaux_article(raw_item, target_ticker="AAPL")
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["Ticker"], "AAPL")
        self.assertEqual(normalized["Company_Name"], "Apple")
        self.assertEqual(normalized["Headline"], "Apple Introduces Next Generation M4 Silicon")
        self.assertIn("M4 chip", normalized["Summary"])
        self.assertEqual(normalized["Source"], "techcrunch.com")
        self.assertEqual(normalized["URL"], "https://example.com/apple-m4-announcement")
        self.assertEqual(normalized["Provider"], "MarketAux")
        self.assertEqual(normalized["Published_At_UTC"], "2024-05-07T14:00:00Z")
        self.assertEqual(normalized["Dataset_Type"], "REAL")
        self.assertEqual(normalized["Provider_Sentiment"], 0.65)
        self.assertAlmostEqual(normalized["Provider_Relevance"], 0.955, places=3)
        self.assertEqual(len(normalized["News_ID"]), 32)

    def test_04_timestamp_parsing(self):
        """Test timestamp parser normalizes varying ISO formats to standard UTC."""
        ts1 = parse_marketaux_timestamp("2024-10-31T20:30:00.000000Z")
        self.assertEqual(ts1, "2024-10-31T20:30:00Z")

        ts2 = parse_marketaux_timestamp("2024-01-15T09:15:30Z")
        self.assertEqual(ts2, "2024-01-15T09:15:30Z")

        ts3 = parse_marketaux_timestamp("2023-06-01 12:00:00")
        self.assertEqual(ts3, "2023-06-01T12:00:00Z")

        ts_bad = parse_marketaux_timestamp("not-a-valid-date")
        self.assertIsNone(ts_bad)

    def test_05_malformed_article_handling(self):
        """Test articles missing title or publication date return None gracefully."""
        self.assertIsNone(normalize_marketaux_article({}))
        self.assertIsNone(normalize_marketaux_article({"title": ""}))
        self.assertIsNone(normalize_marketaux_article({"title": "Test Title", "published_at": ""}))
        self.assertIsNone(normalize_marketaux_article({"title": "Test Title", "published_at": "invalid-date"}))

    def test_06_text_quality_assessment(self):
        """Test check_text_quality computes counts and percentages accurately."""
        sample_articles = [
            {"title": "Valid Title 1", "description": "Valid desc 1", "source": "reuters.com"},
            {"title": "Valid Title 2", "description": "", "snippet": "Snippet text", "source": "bloomberg.com"},
            {"title": "Title Only", "description": "", "snippet": "", "source": "cnbc.com"},
            {"title": "", "description": "", "snippet": "", "source": "unknown"}
        ]
        q = check_text_quality(sample_articles)
        self.assertEqual(q["total_returned"], 4)
        self.assertEqual(q["title_available"], 3)
        self.assertEqual(q["description_available"], 2)
        self.assertEqual(q["usable_text_available"], 3)
        self.assertEqual(q["missing_usable_text"], 1)
        self.assertEqual(q["usable_text_pct"], 75.0)
        self.assertEqual(q["unique_sources"], 4)

    def test_07_duplicate_detection(self):
        """Test duplicate checks flag duplicate IDs, URLs, and title+date combinations."""
        articles = [
            {"News_ID": "id1", "URL": "https://a.com/1", "Headline": "Apple Profit Up", "Published_At_UTC": "2024-01-01T10:00:00Z"},
            {"News_ID": "id2", "URL": "https://a.com/2", "Headline": "Apple Profit Down", "Published_At_UTC": "2024-01-02T10:00:00Z"},
            {"News_ID": "id1", "URL": "https://a.com/dup1", "Headline": "Apple Different", "Published_At_UTC": "2024-01-03T10:00:00Z"},  # Dup ID
            {"News_ID": "id3", "URL": "https://a.com/1", "Headline": "Apple Different 2", "Published_At_UTC": "2024-01-04T10:00:00Z"}, # Dup URL
            {"News_ID": "id4", "URL": "https://a.com/5", "Headline": "apple profit up", "Published_At_UTC": "2024-01-01T15:00:00Z"},   # Dup title+date
        ]
        dups = check_duplicates(articles)
        self.assertEqual(dups["total_articles"], 5)
        self.assertEqual(dups["duplicate_ids"], 1)
        self.assertEqual(dups["duplicate_urls"], 1)
        self.assertEqual(dups["duplicate_title_dates"], 1)

    def test_08_nlp_compatibility_in_memory(self):
        """Test normalized MarketAux articles process through NLP pipeline in-memory with zero DB writes."""
        raw_samples = get_mock_sample_articles()[:5]
        normalized = [normalize_marketaux_article(item, target_ticker="AAPL") for item in raw_samples]
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
        """Test MarketAux API query handling when response is valid JSON."""
        mock_response = MagicMock()
        mock_response.getcode.return_value = 200
        mock_payload = {
            "meta": {"found": 1250, "returned": 3, "limit": 3, "page": 1},
            "data": [
                {
                    "uuid": "test-uuid-mock",
                    "title": "Mock Apple News Headline",
                    "description": "Mock description for test",
                    "url": "https://mock.com/news",
                    "published_at": "2024-08-01T12:00:00.000000Z",
                    "source": "mocknews.com"
                }
            ]
        }
        mock_response.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        data, msg, status = query_marketaux({"symbols": "AAPL"}, token="mock_valid_token")
        self.assertIsNotNone(data)
        self.assertEqual(msg, "OK")
        self.assertEqual(status, 200)
        self.assertEqual(data["meta"]["found"], 1250)
        self.assertEqual(len(data["data"]), 1)

    @patch("urllib.request.urlopen")
    def test_10_api_quota_limit_handling(self, mock_urlopen):
        """Test MarketAux HTTP 429 quota/rate limit error handling."""
        err_body = json.dumps({"error": {"code": "usage_limit_reached", "message": "Daily quota exceeded"}}).encode("utf-8")
        http_error = urllib.error.HTTPError(
            url="https://api.marketaux.com",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=MagicMock(read=MagicMock(return_value=err_body))
        )
        mock_urlopen.side_effect = http_error

        data, msg, status = query_marketaux({"symbols": "AAPL"}, token="mock_token")
        self.assertIsNone(data)
        self.assertEqual(status, 429)
        self.assertIn("Daily quota exceeded", msg)


if __name__ == "__main__":
    unittest.main()
