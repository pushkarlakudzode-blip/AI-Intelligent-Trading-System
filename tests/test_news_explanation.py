"""
Unit & Integration Test Suite for Phase 3: Intelligent Financial News Explanation.

Verifies:
1. Grounded plain-English explanation structure (what, why, implications, evidence, limitations).
2. Positive wording and constructive implication synthesis.
3. Negative wording and risk headwind synthesis.
4. Mixed sentiment identification and contrasting factor synthesis.
5. Insufficient article text handling.
6. Unrelated / untracked ticker handling (zero hallucinated relationships).
7. Insufficient company-specific evidence detection (macro / peer-only news).
8. Strict separation from trading decisions (sentiment != price prediction / sell signal).
9. Provider abstraction and graceful offline fallback on LLM failure.
10. Staging database persistence, caching, and retrieval.
11. Real stored article explanation quality (JPM and Visa articles).
12. Strict safety assertion: Zero mutations to production database or Phase 1-6 research.
"""

import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.nlp.news_explanation import (
    NewsExplanation,
    BaseNewsExplanationProvider,
    GroundedAnalyticalExplanationProvider,
    LLMNewsExplanationProvider,
    NewsExplanationEngine,
    SUPPORTED_TICKERS,
    COMPANY_NAMES
)
from src.data.data_storage import (
    configure_storage,
    init_storage,
    store_articles,
    store_nlp_results,
    store_explanation,
    get_explanation,
    get_stored_explanations
)

PROD_DB_PATH = PROJECT_ROOT / "data" / "live" / "news_database.db"


class TestNewsExplanation(unittest.TestCase):

    def setUp(self):
        # 1. Record production database baseline counts and MD5
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

        # 2. Setup isolated staging environment
        self.temp_dir = tempfile.TemporaryDirectory()
        self.staging_path = Path(self.temp_dir.name)
        configure_storage(self.staging_path)
        init_storage()

        self.provider = GroundedAnalyticalExplanationProvider()
        self.engine = NewsExplanationEngine(
            primary_provider=self.provider,
            db_path=self.staging_path / "news_database.db"
        )

    def tearDown(self):
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

            curr_md5 = hashlib.md5(PROD_DB_PATH.read_bytes()).hexdigest()

            self.assertEqual(curr_articles, self.baseline_articles, "Production articles count mutated!")
            self.assertEqual(curr_nlp, self.baseline_nlp, "Production nlp_results count mutated!")
            self.assertEqual(curr_val, self.baseline_val, "Production validation_logs count mutated!")
            self.assertEqual(curr_md5, self.baseline_md5, "Production database MD5 mutated!")

    # 1. Positive Wording Explanation
    def test_01_positive_wording_explanation(self):
        art = {
            "news_id": "test_pos_001",
            "ticker": "AAPL",
            "headline": "Apple Reports Record Q4 Revenue and Services Margin Expansion",
            "summary": "Apple Inc posted profitable operating margins and surging enterprise device shipments across all major regions."
        }
        expl = self.provider.generate_explanation(art)

        self.assertEqual(expl.ticker, "AAPL")
        self.assertEqual(expl.company_name, "Apple")
        self.assertEqual(expl.implication_direction, "POSITIVE")
        self.assertEqual(expl.evidence_strength, "DIRECT")
        self.assertIn("POTENTIAL POSITIVE IMPLICATIONS", expl.implications)
        self.assertIn("Apple", expl.what_it_reports)
        self.assertIn("consumer technology hardware", expl.why_it_matters)
        self.assertIn("Record Q4 Revenue", expl.evidence)
        self.assertTrue(len(expl.limitations) > 20)
        self.assertIn("DISCLAIMER", expl.directional_disclaimer)

    # 2. Negative Wording Explanation
    def test_02_negative_wording_explanation(self):
        art = {
            "news_id": "test_neg_001",
            "ticker": "TSLA",
            "headline": "Tesla Faces Regulatory Probe and Vehicle Delivery Decline in Europe",
            "summary": "Tesla reported dropping quarterly sales and new lawsuit challenges regarding driver assistance safety systems."
        }
        expl = self.provider.generate_explanation(art)

        self.assertEqual(expl.ticker, "TSLA")
        self.assertEqual(expl.company_name, "Tesla")
        self.assertEqual(expl.implication_direction, "NEGATIVE")
        self.assertEqual(expl.evidence_strength, "DIRECT")
        self.assertIn("POTENTIAL NEGATIVE IMPLICATIONS", expl.implications)
        self.assertIn("electric vehicle", expl.why_it_matters)
        self.assertIn("probe", expl.evidence.lower())
        # Negative headline MUST NOT imply a SELL trading signal
        self.assertIn("do NOT predict stock price direction", expl.directional_disclaimer)
        self.assertIn("does not constitute a SELL signal", expl.directional_disclaimer)

    # 3. Mixed Sentiment Explanation
    def test_03_mixed_sentiment_explanation(self):
        art = {
            "news_id": "test_mix_001",
            "ticker": "MSFT",
            "headline": "Microsoft Cloud Revenue Surges but Hardware Sales Slump and Growth Slows",
            "summary": "Azure posted record growth while commercial PC demand faces weakness, headcount cuts, and supply bottlenecks."
        }
        expl = self.provider.generate_explanation(art)

        self.assertEqual(expl.ticker, "MSFT")
        self.assertEqual(expl.implication_direction, "MIXED")
        self.assertIn("MIXED IMPLICATIONS", expl.implications)
        self.assertIn("constructive elements", expl.implications)
        self.assertIn("headwinds", expl.implications)
        self.assertIn("surges", expl.evidence.lower())
        self.assertIn("slump", expl.evidence.lower())

    # 4. Insufficient Article Text Handling
    def test_04_insufficient_article_text(self):
        # Empty text
        art_empty = {
            "news_id": "test_empty_001",
            "ticker": "NVDA",
            "headline": "",
            "summary": ""
        }
        expl_empty = self.provider.generate_explanation(art_empty)
        self.assertEqual(expl_empty.evidence_strength, "INSUFFICIENT")
        self.assertEqual(expl_empty.implication_direction, "UNCERTAIN")
        self.assertIn("empty, missing, or severely truncated", expl_empty.what_it_reports)
        self.assertIn("Severe data deficit", expl_empty.limitations)

        # Snippet too short (< 15 chars)
        art_short = {
            "news_id": "test_short_001",
            "ticker": "NVDA",
            "headline": "Nvidia alert",
            "summary": ""
        }
        expl_short = self.provider.generate_explanation(art_short)
        self.assertEqual(expl_short.evidence_strength, "INSUFFICIENT")
        self.assertEqual(expl_short.implication_direction, "UNCERTAIN")

    # 5. Unrelated / Untracked Ticker Handling
    def test_05_unrelated_ticker_handling(self):
        art_untracked = {
            "news_id": "test_untracked_001",
            "ticker": "UNKNOWN_TICKER",
            "headline": "Unknown Firm Announces Merger",
            "summary": "A small microcap entity announced acquisition."
        }
        expl = self.provider.generate_explanation(art_untracked)
        self.assertEqual(expl.implication_direction, "UNCERTAIN")
        self.assertEqual(expl.evidence_strength, "INSUFFICIENT")
        self.assertIn("untracked ticker symbol", expl.what_it_reports)
        self.assertIn("Untracked asset outside system monitoring scope", expl.limitations)

    # 6. Insufficient Company-Specific Evidence (Macro / Peer News)
    def test_06_insufficient_company_specific_evidence(self):
        art_macro = {
            "news_id": "test_macro_001",
            "ticker": "AAPL",
            "headline": "Crude Oil Prices Tumble Following OPEC Production Announcement",
            "summary": "Energy sector shares dropped as commodity traders evaluated global refinery output and shipping logistics."
        }
        expl = self.provider.generate_explanation(art_macro)

        self.assertEqual(expl.ticker, "AAPL")
        self.assertEqual(expl.evidence_strength, "INSUFFICIENT")
        self.assertEqual(expl.implication_direction, "UNCERTAIN")
        self.assertIn("INSUFFICIENT COMPANY-SPECIFIC EVIDENCE", expl.why_it_matters)
        self.assertIn("does not disclose firm-specific operational metrics", expl.why_it_matters)
        # Guarantees zero hallucinations (does not claim Apple produces crude oil)
        self.assertNotIn("Apple sells crude oil", expl.what_it_reports)
        self.assertNotIn("Apple sells crude oil", expl.why_it_matters)

    # 7. Strict Separation from Trading Decisions
    def test_07_separation_from_trading_decisions(self):
        art_bearish = {
            "news_id": "test_bear_001",
            "ticker": "JPM",
            "headline": "JPMorgan Faces Major Lawsuit Over Commercial Lending Practices",
            "summary": "Federal prosecutors filed penalty claims alleging regulatory compliance breaches."
        }
        expl = self.provider.generate_explanation(art_bearish)

        # 1. Explanation dictionary MUST NOT contain trading action signals
        expl_dict = expl.to_dict()
        self.assertNotIn("action", expl_dict)
        self.assertNotIn("trading_signal", expl_dict)
        self.assertNotIn("target_weight", expl_dict)
        self.assertNotIn("trade_execution", expl_dict)

        # 2. Disclaimer must be present and explicit
        self.assertIn("DISCLAIMER", expl.directional_disclaimer)
        self.assertIn("A negative headline does not constitute a SELL signal", expl.directional_disclaimer)
        self.assertIn("do NOT predict stock price direction", expl.directional_disclaimer)

    # 8. Provider Abstraction and Fallback on Failure
    def test_08_provider_fallback_on_failure(self):
        # Create LLM provider with missing API key
        with patch.dict(os.environ, {}, clear=True):
            llm_provider = LLMNewsExplanationProvider()
            self.assertFalse(llm_provider.is_available())

            art = {
                "news_id": "test_fallback_001",
                "ticker": "NVDA",
                "headline": "NVIDIA Launches Blackwell Ultra Architecture for Enterprise AI",
                "summary": "NVIDIA unveiled its next generation GPU computing platform."
            }
            # Must fall back gracefully without exception
            expl = llm_provider.generate_explanation(art)
            self.assertIsNotNone(expl)
            self.assertEqual(expl.ticker, "NVDA")
            self.assertEqual(expl.provider_used, "GroundedAnalyticalEngine")
            self.assertEqual(expl.evidence_strength, "DIRECT")

    # 9. Database Caching, Retrieval & Persistence
    def test_09_database_caching_and_persistence(self):
        art = {
            "news_id": "test_cache_001",
            "ticker": "META",
            "headline": "Meta Platforms Deploys Llama Enterprise Open-Source Infrastructure",
            "summary": "Meta announced enterprise developer tools and infrastructure agreements."
        }

        # First call -> generates and writes to staging DB
        expl_1 = self.engine.get_or_create_explanation(art)
        self.assertIsNotNone(expl_1)
        self.assertEqual(expl_1.ticker, "META")

        # Query staging DB directly to verify persistence
        staging_db = self.staging_path / "news_database.db"
        conn = sqlite3.connect(staging_db)
        cur = conn.cursor()
        cur.execute("SELECT news_id, ticker, implication_direction FROM news_explanations WHERE news_id = 'test_cache_001'")
        row = cur.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], "test_cache_001")
        self.assertEqual(row[1], "META")

        # Second call -> retrieves cached from DB
        expl_2 = self.engine.get_or_create_explanation(art, force_refresh=False)
        self.assertEqual(expl_2.news_id, expl_1.news_id)
        self.assertEqual(expl_2.what_it_reports, expl_1.what_it_reports)
        self.assertEqual(expl_2.implications, expl_1.implications)

    # 10. Real Stored Article Explanation Verification (JPM & Visa articles)
    def test_10_real_stored_article_explanation(self):
        # 1. Real JPMorgan Chase broadcast (Article 650)
        art_jpm = {
            "news_id": "02254762af1b07537f410701521ddcab",
            "ticker": "JPM",
            "headline": "Nasdaq Settles Lower As Chip Stocks Slide; Investor Sentiment Weakens, Fear & Greed Index Remains In 'Neutral' Zone",
            "summary": "Market sentiment slipped but stayed neutral at 45 as U.S. stocks fell Tuesday, led by chip losses; investors awaited major bank earnings."
        }
        expl_jpm = self.provider.generate_explanation(art_jpm)
        self.assertEqual(expl_jpm.ticker, "JPM")
        self.assertEqual(expl_jpm.company_name, "JPMorgan Chase")
        # Direct entity is not in headline (macro market news), so evidence strength must be INSUFFICIENT or PARTIAL
        self.assertIn(expl_jpm.evidence_strength, ["INSUFFICIENT", "PARTIAL"])
        self.assertIn("INSUFFICIENT COMPANY-SPECIFIC EVIDENCE", expl_jpm.why_it_matters)
        self.assertEqual(expl_jpm.implication_direction, "UNCERTAIN")

        # 2. Real Visa broadcast (Article 649)
        art_visa = {
            "news_id": "c1538105139c7eb70189ba4af7e1ffad",
            "ticker": "V",
            "headline": "Buffett Successor Sold Visa Stock in Q1: Why That May Have Been an $823M Mistake",
            "summary": "Greg Abel sold off the entire position in Visa stock in Q1, once the 13th largest position in Berkshire's investment portfolio."
        }
        expl_visa = self.provider.generate_explanation(art_visa)
        self.assertEqual(expl_visa.ticker, "V")
        self.assertEqual(expl_visa.company_name, "Visa")
        self.assertEqual(expl_visa.evidence_strength, "DIRECT")
        self.assertIn("$823M", expl_visa.evidence)
        self.assertIn("payment volume", expl_visa.why_it_matters)
        self.assertIn("DISCLAIMER", expl_visa.directional_disclaimer)


if __name__ == "__main__":
    unittest.main()
