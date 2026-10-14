"""
Unit Tests for Alpaca Historical News Feasibility Module.

Verifies:
1. Credential handling and missing-key detection
2. Credential masking security
3. RFC-3339 timestamp parsing to UTC ISO format
4. HTML content stripping and summary fallback
5. Article normalization to standardized project schema
6. Lookahead protection (future-dated article rejection)
7. Malformed article rejection (missing headline or timestamp)
8. Text quality metrics computation (headline, summary, content)
9. Multi-level duplicate detection (ID, URL, Headline+Date)
10. In-memory NLP compatibility test without persistence
11. Mocked API query success with rate limit header extraction
12. Mocked API error handling (401 auth, 429 rate limit)
13. Production database non-interference guarantee
"""

import sys
import unittest
import sqlite3
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import urllib.error

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.alpaca_news_test import (
    get_alpaca_credentials,
    mask_credential,
    parse_alpaca_timestamp,
    clean_html_content,
    normalize_alpaca_article,
    query_alpaca_news,
    check_text_quality,
    check_duplicates,
    test_nlp_compatibility,
    get_mock_sample_articles
)

PROD_DB_PATH = PROJECT_ROOT / "data" / "live" / "news_database.db"


class TestAlpacaFeasibility(unittest.TestCase):

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

    def test_01_credential_missing_handling(self):
        """Test missing credentials return error and status 401 without network calls."""
        with patch.dict("os.environ", {"ALPACA_API_KEY_ID": "", "ALPACA_API_SECRET_KEY": ""}):
            key, secret = get_alpaca_credentials()
            self.assertIsNone(key)
            self.assertIsNone(secret)
            data, msg, status, headers = query_alpaca_news({"symbols": "AAPL"})
            self.assertIsNone(data)
            self.assertEqual(status, 401)
            self.assertIn("ALPACA_API_KEY_ID", msg)

    def test_02_credential_masking(self):
        """Test credential masking security never exposes full API key or secret."""
        self.assertEqual(mask_credential(None), "None")
        self.assertEqual(mask_credential(""), "None")
        self.assertEqual(mask_credential("short"), "***")
        masked = mask_credential("PKTESTKEY1234567890")
        self.assertTrue(masked.startswith("PKTE..."))
        self.assertTrue(masked.endswith("7890"))
        self.assertNotIn("KEY123456", masked)

    def test_03_timestamp_parsing(self):
        """Test RFC-3339 timestamps parse into standardized UTC ISO format."""
        ts1 = parse_alpaca_timestamp("2019-01-02T21:25:29Z")
        self.assertEqual(ts1, "2019-01-02T21:25:29Z")

        ts2 = parse_alpaca_timestamp("2024-10-31T20:30:00.123456Z")
        self.assertEqual(ts2, "2024-10-31T20:30:00Z")

        ts_bad = parse_alpaca_timestamp("invalid-date")
        self.assertIsNone(ts_bad)
        self.assertIsNone(parse_alpaca_timestamp(None))

    def test_04_html_content_cleaning(self):
        """Test clean_html_content strips tags and unescapes entities."""
        raw_html = "<p>Apple Inc. &amp; partners announce <strong>strong</strong> revenue.</p>"
        cleaned = clean_html_content(raw_html)
        self.assertEqual(cleaned, "Apple Inc. & partners announce strong revenue.")

    def test_05_article_normalization(self):
        """Test normalization maps raw Alpaca dictionary to standardized project schema."""
        raw_item = {
            "id": 13079882,
            "headline": "Apple Revises Q1 Guidance Citing China Slowdown",
            "author": "Benzinga Newsdesk",
            "created_at": "2019-01-02T21:32:06Z",
            "summary": "Apple lowered its first-quarter revenue projection to 84 billion.",
            "content": "<p>Full article body here.</p>",
            "url": "https://www.benzinga.com/node/13079882",
            "symbols": ["AAPL"],
            "source": "benzinga"
        }
        normalized = normalize_alpaca_article(raw_item, target_ticker="AAPL")
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["Ticker"], "AAPL")
        self.assertEqual(normalized["Company_Name"], "Apple")
        self.assertEqual(normalized["Headline"], "Apple Revises Q1 Guidance Citing China Slowdown")
        self.assertIn("84 billion", normalized["Summary"])
        self.assertEqual(normalized["Source"], "benzinga")
        self.assertEqual(normalized["URL"], "https://www.benzinga.com/node/13079882")
        self.assertEqual(normalized["Provider"], "Alpaca")
        self.assertEqual(normalized["Published_At_UTC"], "2019-01-02T21:32:06Z")
        self.assertEqual(normalized["Dataset_Type"], "REAL")
        self.assertEqual(normalized["Provider_Sentiment"], 0.0)
        self.assertEqual(len(normalized["News_ID"]), 32)

    def test_06_summary_fallback_from_content(self):
        """Test normalization extracts summary from content when summary field is empty."""
        raw_item = {
            "id": 13079885,
            "headline": "Apple Releases Software Update for iPhone",
            "created_at": "2020-05-20T14:00:00Z",
            "summary": "",
            "content": "<p>Apple has deployed iOS 13.5 with contact tracing API improvements and Face ID tweaks.</p>",
            "url": "https://www.benzinga.com/node/13079885"
        }
        normalized = normalize_alpaca_article(raw_item, target_ticker="AAPL")
        self.assertIsNotNone(normalized)
        self.assertIn("iOS 13.5", normalized["Summary"])

    def test_07_lookahead_future_dated_protection(self):
        """Test articles dated in the future beyond 15 minutes are strictly rejected."""
        future_dt = (datetime.now(timezone.utc) + timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
        raw_item = {
            "id": 99999999,
            "headline": "Future Dated Article Should Be Rejected",
            "created_at": future_dt,
            "url": "https://example.com/future"
        }
        normalized = normalize_alpaca_article(raw_item, target_ticker="AAPL")
        self.assertIsNone(normalized)

    def test_08_malformed_article_handling(self):
        """Test articles missing headline or timestamp return None gracefully."""
        self.assertIsNone(normalize_alpaca_article({}))
        self.assertIsNone(normalize_alpaca_article({"headline": ""}))
        self.assertIsNone(normalize_alpaca_article({"headline": "Valid Headline", "created_at": None}))
        self.assertIsNone(normalize_alpaca_article({"headline": "Valid Headline", "created_at": "invalid"}))

    def test_09_text_quality_assessment(self):
        """Test check_text_quality computes counts and percentages accurately."""
        sample_articles = [
            {"headline": "Headline 1", "summary": "Summary 1", "content": "Content 1", "source": "benzinga"},
            {"headline": "Headline 2", "summary": "", "content": "Content 2", "source": "benzinga"},
            {"headline": "Headline 3", "summary": "", "content": "", "source": "reuters"},
            {"headline": "", "summary": "", "content": "", "source": "unknown"}
        ]
        q = check_text_quality(sample_articles)
        self.assertEqual(q["total_returned"], 4)
        self.assertEqual(q["headline_available"], 3)
        self.assertEqual(q["summary_available"], 1)
        self.assertEqual(q["content_available"], 2)
        self.assertEqual(q["usable_text_available"], 3)
        self.assertEqual(q["missing_usable_text"], 1)
        self.assertEqual(q["usable_text_pct"], 75.0)

    def test_10_duplicate_detection(self):
        """Test duplicate checks flag duplicate IDs, URLs, and headline+date combinations."""
        articles = [
            {"News_ID": "a1", "URL": "https://b.com/1", "Headline": "Apple Profit Up", "Published_At_UTC": "2024-01-01T10:00:00Z"},
            {"News_ID": "a2", "URL": "https://b.com/2", "Headline": "Apple Profit Down", "Published_At_UTC": "2024-01-02T10:00:00Z"},
            {"News_ID": "a1", "URL": "https://b.com/dup1", "Headline": "Apple Different", "Published_At_UTC": "2024-01-03T10:00:00Z"},
            {"News_ID": "a3", "URL": "https://b.com/1", "Headline": "Apple Different 2", "Published_At_UTC": "2024-01-04T10:00:00Z"},
            {"News_ID": "a4", "URL": "https://b.com/5", "Headline": "apple profit up", "Published_At_UTC": "2024-01-01T15:00:00Z"},
        ]
        dups = check_duplicates(articles)
        self.assertEqual(dups["total_articles"], 5)
        self.assertEqual(dups["duplicate_ids"], 1)
        self.assertEqual(dups["duplicate_urls"], 1)
        self.assertEqual(dups["duplicate_title_dates"], 1)

    def test_11_nlp_compatibility_in_memory(self):
        """Test normalized Alpaca articles process through NLP pipeline in-memory with zero DB writes."""
        raw_samples = get_mock_sample_articles()[:5]
        normalized = [normalize_alpaca_article(item, target_ticker="AAPL") for item in raw_samples]
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
    def test_12_mocked_api_query_success_and_headers(self, mock_urlopen):
        """Test Alpaca API query handling with rate limit headers extraction."""
        mock_response = MagicMock()
        mock_response.getcode.return_value = 200
        mock_response.headers = {
            "X-RateLimit-Limit": "200",
            "X-RateLimit-Remaining": "195",
            "X-RateLimit-Reset": "1720000000"
        }
        mock_payload = {
            "news": [
                {
                    "id": 12345,
                    "headline": "Mock Headline",
                    "created_at": "2024-01-01T12:00:00Z",
                    "url": "https://mock.com/1"
                }
            ],
            "next_page_token": "token123"
        }
        mock_response.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        data, msg, status, headers = query_alpaca_news(
            {"symbols": "AAPL"},
            key_id="mock_key",
            secret_key="mock_secret"
        )
        self.assertIsNotNone(data)
        self.assertEqual(msg, "OK")
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("X-RateLimit-Limit"), "200")
        self.assertEqual(data.get("next_page_token"), "token123")

    @patch("urllib.request.urlopen")
    def test_13_mocked_api_rate_limit_error(self, mock_urlopen):
        """Test Alpaca HTTP 429 rate limit error handling."""
        err_body = json.dumps({"message": "rate limit exceeded"}).encode("utf-8")
        http_error_429 = urllib.error.HTTPError(
            url="https://data.alpaca.markets/v1beta1/news",
            code=429,
            msg="Too Many Requests",
            hdrs={"X-RateLimit-Limit": "200", "X-RateLimit-Remaining": "0"},
            fp=MagicMock(read=MagicMock(return_value=err_body))
        )
        mock_urlopen.side_effect = http_error_429

        data, msg, status, headers = query_alpaca_news(
            {"symbols": "AAPL"},
            key_id="mock_key",
            secret_key="mock_secret"
        )
        self.assertIsNone(data)
        self.assertEqual(status, 429)
        self.assertIn("rate limit exceeded", msg)


if __name__ == "__main__":
    unittest.main()
