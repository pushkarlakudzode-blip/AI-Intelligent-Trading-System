"""
Unit Tests for 10-Company Alpaca Historical News Feasibility Module.

Verifies:
1. Target company universe registry (all 10 tickers present with valid names and query symbols)
2. Meta Platforms ticker aliasing (META / FB historical transition)
3. Credential retrieval, missing-key detection, and masking security
4. RFC-3339 timestamp parsing and normalization to UTC ISO format
5. HTML content stripping, unescaping, and summary fallback
6. Article normalization to standardized project schema with SHA-256 News_ID
7. Lookahead protection (future-dated article rejection)
8. Malformed article rejection (missing headline or timestamp)
9. Text quality metrics computation (headline, summary, content, sources)
10. Multi-level duplicate detection (ID, URL, Headline+Date)
11. In-memory NLP pipeline compatibility test without database persistence
12. Scale estimation computation for 10-company multi-year backfill
13. Mock sample article generator covering all 10 tickers
14. Mocked API query success with rate limit header extraction
15. Mocked API error handling (401 auth, 429 rate limit)
16. Production database non-interference guarantee
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

from src.data.alpaca_10company_test import (
    get_alpaca_credentials,
    mask_credential,
    parse_alpaca_timestamp,
    clean_html_content,
    normalize_alpaca_article,
    query_alpaca_news,
    check_text_quality,
    check_duplicates,
    test_nlp_compatibility,
    estimate_10company_scale,
    get_mock_10company_samples,
    TARGET_COMPANIES
)

PROD_DB_PATH = PROJECT_ROOT / "data" / "live" / "news_database.db"


class TestAlpaca10CompanyFeasibility(unittest.TestCase):

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

    def test_01_target_companies_registry(self):
        """Verify all 10 project target companies are correctly registered."""
        expected_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ"]
        self.assertEqual(len(TARGET_COMPANIES), 10)
        for ticker in expected_tickers:
            self.assertIn(ticker, TARGET_COMPANIES)
            info = TARGET_COMPANIES[ticker]
            self.assertIn("name", info)
            self.assertIn("query_symbol", info)
            self.assertTrue(len(info["name"]) > 0)

        # Verify META has the dual FB/META query alias for pre-2022 coverage
        self.assertEqual(TARGET_COMPANIES["META"]["query_symbol"], "META,FB")

    def test_02_credential_handling_and_masking(self):
        """Test missing credentials return error and masking never displays plain text."""
        with patch.dict("os.environ", {"ALPACA_API_KEY_ID": "", "ALPACA_API_SECRET_KEY": ""}):
            key, secret = get_alpaca_credentials()
            self.assertIsNone(key)
            self.assertIsNone(secret)
            data, msg, status, headers = query_alpaca_news({"symbols": "AAPL"})
            self.assertIsNone(data)
            self.assertEqual(status, 401)
            self.assertIn("ALPACA_API_KEY_ID", msg)

        # Test masking
        self.assertEqual(mask_credential(None), "None")
        self.assertEqual(mask_credential(""), "None")
        self.assertEqual(mask_credential("short"), "***")
        masked = mask_credential("PKTESTKEY123456789")
        self.assertTrue(masked.startswith("PKTE"))
        self.assertTrue(masked.endswith("6789"))
        self.assertNotIn("12345", masked)

    def test_03_timestamp_parsing(self):
        """Test RFC-3339 timestamps and Unix epochs parse to standardized ISO UTC format."""
        # RFC-3339 standard
        ts1 = parse_alpaca_timestamp("2019-01-02T14:30:00Z")
        self.assertEqual(ts1, "2019-01-02T14:30:00Z")

        # Offset timestamp
        ts2 = parse_alpaca_timestamp("2021-06-15T09:30:00-04:00")
        self.assertEqual(ts2, "2021-06-15T13:30:00Z")

        # Unix epoch
        ts3 = parse_alpaca_timestamp(1546439400)
        self.assertTrue(ts3.startswith("2019-01-02"))

        # Invalid timestamp
        self.assertIsNone(parse_alpaca_timestamp("not-a-date"))
        self.assertIsNone(parse_alpaca_timestamp(None))

    def test_04_html_content_cleaning(self):
        """Test HTML stripping, entity unescaping, and summary fallback."""
        raw_html = "<p>Apple announced <b>strong</b> revenue &amp; earnings growth.</p>"
        cleaned = clean_html_content(raw_html)
        self.assertEqual(cleaned, "Apple announced strong revenue & earnings growth.")

        raw_item = {
            "id": 1001,
            "headline": "Microsoft Cloud Accelerates",
            "created_at": "2020-03-15T10:00:00Z",
            "summary": "",  # Empty summary
            "content": "<p>Microsoft enterprise cloud customer base grew significantly during Q1.</p>",
            "url": "https://benzinga.com/msft-cloud",
            "source": "Benzinga"
        }
        norm = normalize_alpaca_article(raw_item, target_ticker="MSFT")
        self.assertIsNotNone(norm)
        self.assertEqual(norm["Summary"], "Microsoft enterprise cloud customer base grew significantly during Q1.")

    def test_05_article_normalization_and_id_generation(self):
        """Test normalization into project schema with deterministic News_ID."""
        raw_item = {
            "id": 8888,
            "headline": "NVIDIA Unveils Next-Gen AI Accelerators",
            "created_at": "2023-05-24T18:00:00Z",
            "summary": "NVIDIA introduces advanced computing architecture for large language models.",
            "url": "https://benzinga.com/nvda-ai",
            "source": "Benzinga Newsdesk"
        }
        norm = normalize_alpaca_article(raw_item, target_ticker="NVDA")
        self.assertIsNotNone(norm)
        self.assertEqual(norm["Ticker"], "NVDA")
        self.assertEqual(norm["Company_Name"], "NVIDIA")
        self.assertEqual(norm["Published_At_UTC"], "2023-05-24T18:00:00Z")
        self.assertEqual(norm["Headline"], "NVIDIA Unveils Next-Gen AI Accelerators")
        self.assertEqual(norm["Provider"], "Alpaca")
        self.assertEqual(norm["Dataset_Type"], "REAL")
        self.assertEqual(len(norm["News_ID"]), 32)

        # Verify idempotency
        norm2 = normalize_alpaca_article(raw_item, target_ticker="NVDA")
        self.assertEqual(norm["News_ID"], norm2["News_ID"])

    def test_06_meta_ticker_aliasing(self):
        """Test FB historical ticker alias normalizes to META."""
        raw_fb = {
            "id": 7777,
            "headline": "Facebook Announces Annual Developer Conference",
            "created_at": "2019-04-30T16:00:00Z",
            "summary": "Facebook reveals latest software platform updates.",
            "url": "https://benzinga.com/fb-f8",
            "source": "Benzinga"
        }
        # Passed with FB target ticker
        norm_fb = normalize_alpaca_article(raw_fb, target_ticker="FB")
        self.assertIsNotNone(norm_fb)
        self.assertEqual(norm_fb["Ticker"], "META")
        self.assertEqual(norm_fb["Company_Name"], "Meta Platforms")

        # Passed with META target ticker
        norm_meta = normalize_alpaca_article(raw_fb, target_ticker="META")
        self.assertEqual(norm_meta["Ticker"], "META")
        self.assertEqual(norm_meta["Company_Name"], "Meta Platforms")

    def test_07_future_dated_lookahead_rejection(self):
        """Test future-dated articles are rejected to prevent lookahead bias."""
        future_item = {
            "id": 9999,
            "headline": "Future Event in 2030",
            "created_at": "2030-01-01T00:00:00Z",
            "summary": "A future prediction.",
            "url": "https://example.com/future"
        }
        norm = normalize_alpaca_article(future_item, target_ticker="AAPL")
        self.assertIsNone(norm)

    def test_08_malformed_article_rejection(self):
        """Test articles lacking headline or timestamp return None."""
        # Missing headline
        bad_item1 = {
            "id": 1,
            "headline": "",
            "created_at": "2021-01-01T00:00:00Z",
            "url": "https://example.com"
        }
        self.assertIsNone(normalize_alpaca_article(bad_item1, "AAPL"))

        # Missing timestamp
        bad_item2 = {
            "id": 2,
            "headline": "Valid Headline",
            "created_at": None,
            "url": "https://example.com"
        }
        self.assertIsNone(normalize_alpaca_article(bad_item2, "AAPL"))

    def test_09_text_quality_metrics(self):
        """Test check_text_quality correctly computes quality percentages."""
        articles = [
            {"headline": "H1", "summary": "S1", "content": "<p>C1</p>", "source": "Benzinga"},
            {"headline": "H2", "summary": "S2", "content": "", "source": "Benzinga"},
            {"headline": "H3", "summary": "", "content": "", "source": "Partner"},
            {"headline": "", "summary": "", "content": "", "source": "Unknown"}
        ]
        q = check_text_quality(articles)
        self.assertEqual(q["total_returned"], 4)
        self.assertEqual(q["headline_available"], 3)
        self.assertEqual(q["summary_available"], 2)
        self.assertEqual(q["content_available"], 1)
        self.assertEqual(q["usable_text_available"], 3)
        self.assertEqual(q["unique_sources"], 3)
        self.assertEqual(q["usable_text_pct"], 75.0)

    def test_10_duplicate_detection(self):
        """Test check_duplicates flags duplicate IDs, URLs, and title+date."""
        articles = [
            {"News_ID": "ID1", "URL": "http://a.com/1", "Headline": "Title A", "Published_At_UTC": "2020-01-01T00:00:00Z"},
            {"News_ID": "ID1", "URL": "http://a.com/2", "Headline": "Title B", "Published_At_UTC": "2020-01-01T00:00:00Z"},  # Dup ID
            {"News_ID": "ID2", "URL": "http://a.com/1", "Headline": "Title C", "Published_At_UTC": "2020-01-01T00:00:00Z"},  # Dup URL
            {"News_ID": "ID3", "URL": "http://a.com/3", "Headline": "Title A", "Published_At_UTC": "2020-01-01T12:00:00Z"},  # Dup title+date
            {"News_ID": "ID4", "URL": "http://a.com/4", "Headline": "Unique", "Published_At_UTC": "2020-01-02T00:00:00Z"}
        ]
        dups = check_duplicates(articles)
        self.assertEqual(dups["duplicate_ids"], 1)
        self.assertEqual(dups["duplicate_urls"], 1)
        self.assertEqual(dups["duplicate_title_dates"], 1)

    def test_11_nlp_compatibility_in_memory(self):
        """Test normalized articles process through NLP pipeline in-memory with zero DB writes."""
        mock_raw = get_mock_10company_samples()[:5]
        normalized = []
        for item in mock_raw:
            norm = normalize_alpaca_article(item, target_ticker=item["symbols"][0])
            if norm:
                normalized.append(norm)

        self.assertEqual(len(normalized), 5)
        eval_res = test_nlp_compatibility(normalized, max_sample=5)
        self.assertEqual(eval_res["sample_size"], 5)
        self.assertEqual(eval_res["successfully_processed"], 5)
        self.assertEqual(eval_res["failed"], 0)
        self.assertEqual(len(eval_res["sample_results"]), 5)

        first_res = eval_res["sample_results"][0]
        self.assertIn("Sentiment_Score", first_res)
        self.assertIn("Relevance_Score", first_res)
        self.assertIn("Triage_Score", first_res)

    def test_12_scale_estimation_computation(self):
        """Test estimate_10company_scale computes realistic numbers covering all 10 tickers."""
        est = estimate_10company_scale()
        self.assertIn("company_breakdown", est)
        self.assertEqual(len(est["company_breakdown"]), 10)
        for ticker in TARGET_COMPANIES.keys():
            self.assertIn(ticker, est["company_breakdown"])

        self.assertGreater(est["total_estimated_articles"], 50000)
        self.assertLess(est["total_estimated_articles"], 120000)
        self.assertGreater(est["total_estimated_requests"], 1000)
        self.assertLess(est["estimated_duration_minutes"], 20.0)

    def test_13_mock_data_generator(self):
        """Test mock data generator produces articles for all 10 companies across multiple years."""
        mocks = get_mock_10company_samples()
        self.assertGreater(len(mocks), 25)
        tickers_present = set()
        for m in mocks:
            for s in m.get("symbols", []):
                tickers_present.add(s)
        self.assertEqual(len(tickers_present), 10)

    def test_14_mocked_api_query_success_and_headers(self):
        """Test Alpaca API query handling with rate limit headers extraction."""
        mock_resp_data = {
            "news": [
                {
                    "id": 12345,
                    "headline": "Tesla Sets New Quarterly Delivery Record",
                    "created_at": "2021-10-02T12:00:00Z",
                    "symbols": ["TSLA"]
                }
            ],
            "next_page_token": "token_abc123"
        }
        mock_resp_bytes = json.dumps(mock_resp_data).encode("utf-8")

        mock_http_response = MagicMock()
        mock_http_response.getcode.return_value = 200
        mock_http_response.read.return_value = mock_resp_bytes
        mock_http_response.headers = {
            "X-RateLimit-Limit": "200",
            "X-RateLimit-Remaining": "199",
            "Content-Type": "application/json"
        }
        mock_http_response.__enter__.return_value = mock_http_response

        with patch("urllib.request.urlopen", return_value=mock_http_response):
            data, msg, status, rate_headers = query_alpaca_news(
                {"symbols": "TSLA"},
                key_id="TEST_KEY",
                secret_key="TEST_SECRET"
            )
            self.assertEqual(status, 200)
            self.assertEqual(msg, "OK")
            self.assertIsNotNone(data)
            self.assertEqual(len(data["news"]), 1)
            self.assertEqual(rate_headers.get("X-RateLimit-Limit"), "200")
            self.assertEqual(rate_headers.get("X-RateLimit-Remaining"), "199")

    def test_15_mocked_api_error_handling(self):
        """Test Alpaca HTTP 429 rate limit and HTTP 401 error handling."""
        # 429 Rate Limit
        http_err_429 = urllib.error.HTTPError(
            url="https://data.alpaca.markets/v1beta1/news",
            code=429,
            msg="Too Many Requests",
            hdrs={"X-RateLimit-Limit": "200", "X-RateLimit-Remaining": "0"},
            fp=MagicMock(read=lambda: b'{"message": "rate limit exceeded"}')
        )
        with patch("urllib.request.urlopen", side_effect=http_err_429):
            data, msg, status, headers = query_alpaca_news(
                {"symbols": "AAPL"},
                key_id="KEY",
                secret_key="SECRET"
            )
            self.assertIsNone(data)
            self.assertEqual(status, 429)
            self.assertIn("Rate Limit", msg)

        # 401 Unauthorized
        http_err_401 = urllib.error.HTTPError(
            url="https://data.alpaca.markets/v1beta1/news",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=MagicMock(read=lambda: b'{"message": "invalid api key"}')
        )
        with patch("urllib.request.urlopen", side_effect=http_err_401):
            data, msg, status, headers = query_alpaca_news(
                {"symbols": "AAPL"},
                key_id="KEY",
                secret_key="SECRET"
            )
            self.assertIsNone(data)
            self.assertEqual(status, 401)
            self.assertIn("Auth Error", msg)


if __name__ == "__main__":
    unittest.main()
