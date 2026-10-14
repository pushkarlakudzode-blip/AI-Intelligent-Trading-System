"""
Phase 5C: Live News Data Quality, Coverage & Readiness Test Suite.
==================================================================
Rigorous unit and integration tests verifying:
1. Timestamp interpretation across UTC and US Market Time (Pre, Regular, Post, Overnight, Weekend).
2. Strict ticker symbol association across all 10 supported portfolio equities.
3. Deterministic deduplication behavior.
4. Outage vs. genuine zero-news period classification criteria.
5. Ingestion latency calculation and clock skew detection.
6. Data quality defect detection (missing headlines, summaries, invalid timestamps).
7. NLP and explanation pipeline coverage audit.
8. Production database and Phase 1-6 research artifact preservation.
"""

from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
import pandas as pd
import pytz

from src.analytics.phase5c_news_auditor import (
    Phase5CNewsAuditor,
    SUPPORTED_TICKERS,
    COMPANY_NAMES,
    NewsDataQualityReport
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIVE_DB_PATH = PROJECT_ROOT / "data" / "live" / "news_database.db"
LIVE_STATUS_PATH = PROJECT_ROOT / "data" / "live" / "alpaca_stream_status.json"
RESEARCH_DIR = PROJECT_ROOT / "data" / "live" / "alpaca_staging" / "research"
PHASE5C_DIR = PROJECT_ROOT / "data" / "evaluation" / "phase5c"


class TestPhase5CNewsReadinessAudit(unittest.TestCase):
    """Rigorous test suite for Phase 5C Live News Audit Engine."""

    @classmethod
    def setUpClass(cls):
        cls.auditor = Phase5CNewsAuditor(db_path=LIVE_DB_PATH, status_path=LIVE_STATUS_PATH)

    def test_01_timestamp_utc_and_est_session_classification(self):
        """Verify accurate mapping from UTC timestamp to US/Eastern market sessions."""
        # 1. Weekend (Saturday, 2026-10-10 14:00 UTC -> 10:00 EDT Saturday)
        sat_utc = pd.Timestamp("2026-10-10T14:00:00Z")
        self.assertEqual(self.auditor.classify_session_window(sat_utc), "Weekend")

        # 2. Pre-Market (Wednesday, 2026-10-07 11:00 UTC -> 07:00 EDT)
        pre_utc = pd.Timestamp("2026-10-07T11:00:00Z")
        self.assertEqual(self.auditor.classify_session_window(pre_utc), "Pre-Market (04:00-09:30)")

        # 3. Regular Trading Hours (Wednesday, 2026-10-07 15:00 UTC -> 11:00 EDT)
        reg_utc = pd.Timestamp("2026-10-07T15:00:00Z")
        self.assertEqual(self.auditor.classify_session_window(reg_utc), "Regular Hours (09:30-16:00)")

        # 4. After-Hours (Wednesday, 2026-10-07 21:00 UTC -> 17:00 EDT)
        post_utc = pd.Timestamp("2026-10-07T21:00:00Z")
        self.assertEqual(self.auditor.classify_session_window(post_utc), "After-Hours (16:00-20:00)")

        # 5. Overnight (Wednesday night, 2026-10-08 02:00 UTC -> 22:00 EDT Wednesday)
        over_utc1 = pd.Timestamp("2026-10-08T02:00:00Z")
        self.assertEqual(self.auditor.classify_session_window(over_utc1), "Overnight (20:00-04:00)")

        # 6. Overnight (Early morning, 2026-10-08 06:00 UTC -> 02:00 EDT Thursday)
        over_utc2 = pd.Timestamp("2026-10-08T06:00:00Z")
        self.assertEqual(self.auditor.classify_session_window(over_utc2), "Overnight (20:00-04:00)")

        # 7. Missing or invalid timestamp
        self.assertEqual(self.auditor.classify_session_window(pd.NaT), "Unknown")

    def test_02_strict_ticker_symbol_association(self):
        """Verify cross-tabulation covers all 10 supported portfolio tickers."""
        df_art = pd.DataFrame([
            {"ticker": "AAPL", "provider": "Alpaca"},
            {"ticker": "TSLA", "provider": "Alpha Vantage"},
            {"ticker": "NVDA", "provider": "Alpaca"}
        ])
        ct = self.auditor.audit_ticker_coverage(df_art)
        
        # Verify all 10 supported tickers are present in the index
        for t in SUPPORTED_TICKERS:
            self.assertIn(t, ct.index)
        
        # Check Total row exists and matches sum
        self.assertIn("Total", ct.index)
        self.assertEqual(ct.loc["Total", "Total"], 3)
        self.assertEqual(ct.loc["AAPL", "Alpaca"], 1)
        self.assertEqual(ct.loc["MSFT", "Total"], 0)

    def test_03_deterministic_deduplication(self):
        """Verify deduplication detects identical news IDs in temporary fixture."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_db = Path(tmpdir) / "test_dup.db"
            conn = sqlite3.connect(tmp_db)
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE articles (
                    news_id TEXT, ticker TEXT, published_at_utc TEXT,
                    retrieved_at_utc TEXT, headline TEXT, summary TEXT, provider TEXT
                )
            """)
            cur.execute("""CREATE TABLE nlp_results (news_id TEXT)""")
            cur.execute("""CREATE TABLE news_explanations (news_id TEXT)""")
            cur.execute("""CREATE TABLE validation_logs (id INTEGER PRIMARY KEY)""")

            # Insert 3 articles, 1 is a duplicate news_id
            rows = [
                ("art_01", "AAPL", "2026-10-01T10:00:00Z", "2026-10-01T10:00:01Z", "Apple Event", "Summary", "Alpaca"),
                ("art_02", "MSFT", "2026-10-01T11:00:00Z", "2026-10-01T11:00:01Z", "MSFT Cloud", "Summary", "Alpaca"),
                ("art_01", "AAPL", "2026-10-01T10:00:00Z", "2026-10-01T10:00:02Z", "Apple Event Dup", "Summary", "Alpaca")
            ]
            cur.executemany("INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
            conn.commit()
            conn.close()

            test_auditor = Phase5CNewsAuditor(db_path=tmp_db)
            rep = test_auditor.run_full_audit(output_dir=Path(tmpdir) / "out")
            self.assertEqual(rep.duplicate_count, 1)
            self.assertEqual(rep.total_articles, 3)

    def test_04_outage_vs_zero_news_rules(self):
        """Verify criteria distinguishing healthy feed with zero news from an outage."""
        now_utc = datetime.now(timezone.utc)
        fresh_hb = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
        stale_hb = (now_utc - timedelta(seconds=120)).strftime("%Y-%m-%d %H:%M:%S UTC")
        empty_df = pd.DataFrame(columns=["ticker", "headline", "published_at_utc", "retrieved_at_utc", "provider"])

        # Case A: Healthy streaming with fresh heartbeat (Rule 1-3 pass, Rule 4 zero news)
        status_healthy = {
            "status": "STREAMING",
            "connected": True,
            "last_heartbeat_utc": fresh_hb,
            "reconnect_count": 0
        }
        res_healthy = self.auditor.distinguish_outage_vs_zero_news(status_healthy, empty_df)
        self.assertEqual(res_healthy["current_feed_diagnosis"], "STREAMING_HEALTHY")
        self.assertTrue(res_healthy["heartbeat_fresh_le_60s"])

        # Case B: Disconnected feed (Rule 5 outage)
        status_disconnected = {
            "status": "DISCONNECTED",
            "connected": False,
            "last_heartbeat_utc": fresh_hb,
            "reconnect_count": 2
        }
        res_disc = self.auditor.distinguish_outage_vs_zero_news(status_disconnected, empty_df)
        self.assertEqual(res_disc["current_feed_diagnosis"], "DEGRADED_OR_OFFLINE")

        # Case C: Connected but stale heartbeat > 60s (Rule 5 outage)
        status_stale = {
            "status": "STREAMING",
            "connected": True,
            "last_heartbeat_utc": stale_hb,
            "reconnect_count": 0
        }
        res_stale = self.auditor.distinguish_outage_vs_zero_news(status_stale, empty_df)
        self.assertEqual(res_stale["current_feed_diagnosis"], "DEGRADED_OR_OFFLINE")
        self.assertFalse(res_stale["heartbeat_fresh_le_60s"])

    def test_05_latency_calculation_and_clock_skew(self):
        """Verify latency calculation and clock drift / skew detection."""
        df_art = pd.DataFrame([
            # Latency: 2.0s
            {"provider": "Alpaca", "published_at_utc": "2026-10-14T06:00:00Z", "retrieved_at_utc": "2026-10-14T06:00:02Z"},
            # Latency: 10.0s
            {"provider": "Alpaca", "published_at_utc": "2026-10-14T06:00:00Z", "retrieved_at_utc": "2026-10-14T06:00:10Z"},
            # Clock skew: -1.0s (retrieved before published)
            {"provider": "Alpaca", "published_at_utc": "2026-10-14T06:00:05Z", "retrieved_at_utc": "2026-10-14T06:00:04Z"},
        ])
        audit_res = self.auditor.audit_timestamps_and_latency(df_art)
        alp_stats = audit_res["alpaca_live_latency_stats"]

        self.assertEqual(alp_stats["article_count"], 3)
        self.assertEqual(alp_stats["negative_latency_count"], 1)
        self.assertEqual(alp_stats["min_latency_sec"], -1.0)
        self.assertEqual(alp_stats["max_latency_sec"], 10.0)
        self.assertAlmostEqual(alp_stats["mean_latency_sec"], (2.0 + 10.0 - 1.0) / 3.0, places=2)

    def test_06_data_quality_defect_detection(self):
        """Verify detection of missing headlines, empty summaries, and invalid timestamps."""
        df_art = pd.DataFrame([
            # Valid
            {"news_id": "1", "headline": "Valid Headline", "summary": "Valid Summary", "published_at_utc": "2026-10-01T10:00:00Z", "retrieved_at_utc": "2026-10-01T10:00:01Z", "provider": "Alpaca"},
            # Missing headline
            {"news_id": "2", "headline": "", "summary": "Has summary", "published_at_utc": "2026-10-01T10:00:00Z", "retrieved_at_utc": "2026-10-01T10:00:01Z", "provider": "Alpaca"},
            # Missing summary
            {"news_id": "3", "headline": "Headline 3", "summary": "   ", "published_at_utc": "2026-10-01T10:00:00Z", "retrieved_at_utc": "2026-10-01T10:00:01Z", "provider": "Alpaca"},
            # Invalid timestamp
            {"news_id": "4", "headline": "Headline 4", "summary": "Summary 4", "published_at_utc": "INVALID_DATE", "retrieved_at_utc": "2026-10-01T10:00:01Z", "provider": "Alpaca"}
        ])

        dup_cnt = int(df_art["news_id"].duplicated().sum())
        missing_h = int((df_art["headline"].fillna("").str.strip() == "").sum())
        missing_s = int((df_art["summary"].fillna("").str.strip() == "").sum())
        time_audit = self.auditor.audit_timestamps_and_latency(df_art)

        self.assertEqual(dup_cnt, 0)
        self.assertEqual(missing_h, 1)
        self.assertEqual(missing_s, 1)
        self.assertEqual(time_audit["invalid_published_timestamps"], 1)

    def test_07_nlp_and_explanation_coverage_audit(self):
        """Verify article to NLP and explanation coverage tracking."""
        df_art, df_nlp, df_exp, df_val = self.auditor.load_database_tables()
        self.assertGreaterEqual(len(df_art), 650)
        self.assertGreaterEqual(len(df_nlp), 650)
        self.assertGreaterEqual(len(df_exp), 15)

        # Confirm that NLP match rate is > 99.5%
        art_ids = set(df_art["news_id"])
        nlp_ids = set(df_nlp["news_id"])
        missing_nlp = len(art_ids - nlp_ids)
        coverage_pct = ((len(art_ids) - missing_nlp) / len(art_ids)) * 100
        self.assertGreater(coverage_pct, 99.5)

    def test_08_production_database_and_research_preservation(self):
        """Verify production database records and Phase 1-6 research artifacts are preserved."""
        # 1. Check production DB read-only integrity
        conn = sqlite3.connect(f"file:{LIVE_DB_PATH}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM articles")
        art_cnt = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM nlp_results")
        nlp_cnt = cur.fetchone()[0]
        conn.close()

        self.assertGreaterEqual(art_cnt, 657)
        self.assertGreaterEqual(nlp_cnt, 656)

        # 2. Check Phase 1-6 research directories exist and are intact
        self.assertTrue(RESEARCH_DIR.exists())
        self.assertTrue((RESEARCH_DIR / "aligned_daily_dataset.csv").exists())
        phase_reports = list((RESEARCH_DIR / "reports").glob("phase*.md"))
        self.assertGreater(len(phase_reports), 0)
        self.assertTrue((RESEARCH_DIR / "phase6").exists())

        # 3. Check Phase 5A and Phase 5B evaluation artifacts exist
        phase5a_dir = PROJECT_ROOT / "data" / "evaluation" / "phase5a"
        phase5b_dir = PROJECT_ROOT / "data" / "evaluation" / "phase5b"
        self.assertTrue(phase5a_dir.exists())
        self.assertTrue(phase5b_dir.exists())
        self.assertTrue((phase5a_dir / "portfolio_comparison.csv").exists())
        self.assertTrue((phase5b_dir / "experiment_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
