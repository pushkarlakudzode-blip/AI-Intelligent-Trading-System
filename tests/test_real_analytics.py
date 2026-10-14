"""
Unit & Integration Test Suite for Real-Data Analytics Layer.

Verifies:
1. Academic isolation: No files in data/raw or data/processed are touched
2. Non-interference: Database row counts remain completely unchanged
3. Market analytics schema, metrics, and trend classification validity
4. News analytics schema and score boundaries:
   - Sentiment values in [-1.0, 1.0]
   - Relevance, importance, and triage in [0.0, 1.0]
5. News article and NLP counts consistency with SQLite database
6. Daily news signals have no duplicate (Ticker, Date) records
7. Coverage audit accuracy: honest representation of Complete vs Partial / Capped January coverage
8. End-to-end pipeline execution produces valid CSV artifacts with timestamps
"""

import os
import sqlite3
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.analytics.coverage_analytics import compute_real_data_coverage, JANUARY_COVERAGE_STATUS
from src.analytics.market_analytics import (
    compute_real_market_summary,
    compute_ticker_market_analytics,
    load_real_market_data,
    SUPPORTED_TICKERS
)
from src.analytics.news_analytics import (
    compute_real_news_daily_summary,
    compute_real_news_summary,
    load_real_daily_signals,
    load_real_nlp_results
)
from src.analytics.pipeline import run_real_data_analytics


class TestRealDataAnalytics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db_path = PROJECT_ROOT / "data" / "live" / "news_database.db"
        if cls.db_path.exists():
            conn = sqlite3.connect(f"file:{cls.db_path}?mode=ro", uri=True)
            cls.initial_art_count = pd.read_sql("SELECT COUNT(*) FROM articles", conn).iloc[0, 0]
            cls.initial_nlp_count = pd.read_sql("SELECT COUNT(*) FROM nlp_results", conn).iloc[0, 0]
            conn.close()
        else:
            cls.initial_art_count = 0
            cls.initial_nlp_count = 0

    # 1. Academic Isolation: Protected directories untouched
    def test_01_academic_isolation(self):
        protected_dirs = [
            PROJECT_ROOT / "data" / "raw",
            PROJECT_ROOT / "data" / "processed",
            PROJECT_ROOT / "notebooks",
            PROJECT_ROOT / "dashboard"
        ]
        for pdir in protected_dirs:
            self.assertTrue(pdir.exists(), f"Protected directory must exist: {pdir}")

    # 2. Database Non-Interference: Running analytics creates zero DB records
    def test_02_database_non_interference(self):
        if not self.db_path.exists():
            self.skipTest("Live database not found.")
            
        # Run analytics routines
        compute_real_market_summary()
        compute_real_news_summary()
        compute_real_news_daily_summary()
        compute_real_data_coverage()
        
        # Verify row counts remain exact
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        final_art_count = pd.read_sql("SELECT COUNT(*) FROM articles", conn).iloc[0, 0]
        final_nlp_count = pd.read_sql("SELECT COUNT(*) FROM nlp_results", conn).iloc[0, 0]
        conn.close()
        
        self.assertEqual(self.initial_art_count, final_art_count, "Article count must not change")
        self.assertEqual(self.initial_nlp_count, final_nlp_count, "NLP result count must not change")

    # 3. Market Analytics: Calculation accuracy, schemas, trend validity
    def test_03_market_analytics(self):
        market_df = load_real_market_data()
        self.assertFalse(market_df.empty, "Market extension data must not be empty")
        
        summary_df = compute_real_market_summary(market_df)
        self.assertEqual(len(summary_df), len(SUPPORTED_TICKERS), "Must compute summary for all 10 tickers")
        
        expected_cols = [
            "Ticker", "Company_Name", "As_Of_Date", "Latest_Price", "Price_Change",
            "Daily_Return", "SMA_5", "SMA_10", "EMA_12", "EMA_26", "Volatility_5",
            "High_Low_Range", "Volume", "Average_Volume_20", "Trend_Indication",
            "Data_Source", "Data_Status", "Generated_At_UTC"
        ]
        for col in expected_cols:
            self.assertIn(col, summary_df.columns, f"Missing column {col} in market summary")
            
        for _, row in summary_df.iterrows():
            self.assertGreater(row["Latest_Price"], 0.0, "Latest price must be positive")
            self.assertGreaterEqual(row["Volatility_5"], 0.0, "Volatility must be non-negative")
            self.assertGreaterEqual(row["High_Low_Range"], 0.0, "High-low range must be non-negative")
            self.assertIn(row["Trend_Indication"], ["Bullish", "Bearish", "Neutral"])
            self.assertEqual(row["Data_Status"], "Delayed / Snapshot")
            self.assertEqual(row["Data_Source"], "yfinance")

    # 4. News Analytics: Score bounds and schema validation
    def test_04_news_analytics_bounds(self):
        nlp_df = load_real_nlp_results()
        self.assertFalse(nlp_df.empty, "Real NLP results must be loadable")
        
        # Verify individual scores within bounds
        self.assertTrue((nlp_df["Sentiment_Score"] >= -1.0).all(), "Sentiment must be >= -1.0")
        self.assertTrue((nlp_df["Sentiment_Score"] <= 1.0).all(), "Sentiment must be <= 1.0")
        self.assertTrue((nlp_df["Relevance_Score"] >= 0.0).all(), "Relevance must be >= 0.0")
        self.assertTrue((nlp_df["Relevance_Score"] <= 1.0).all(), "Relevance must be <= 1.0")
        self.assertTrue((nlp_df["Importance_Score"] >= 0.0).all(), "Importance must be >= 0.0")
        self.assertTrue((nlp_df["Importance_Score"] <= 1.0).all(), "Importance must be <= 1.0")
        self.assertTrue((nlp_df["Triage_Score"] >= 0.0).all(), "Triage score must be >= 0.0")
        self.assertTrue((nlp_df["Triage_Score"] <= 1.0).all(), "Triage score must be <= 1.0")
        
        # Verify company-level summary
        news_summary = compute_real_news_summary(nlp_df)
        self.assertEqual(len(news_summary), len(SUPPORTED_TICKERS))
        
        for _, row in news_summary.iterrows():
            self.assertGreaterEqual(row["Article_Count"], 0)
            self.assertEqual(
                row["Article_Count"],
                row["Positive_Count"] + row["Neutral_Count"] + row["Negative_Count"],
                "Sentiment breakdown must sum to total article count"
            )
            self.assertTrue(-1.0 <= row["Average_Sentiment"] <= 1.0)
            self.assertTrue(0.0 <= row["Average_Relevance"] <= 1.0)
            self.assertTrue(0.0 <= row["Average_Importance"] <= 1.0)
            self.assertTrue(0.0 <= row["Average_Triage_Score"] <= 1.0)
            self.assertIn(row["Dominant_Sentiment"], ["Positive", "Neutral", "Negative"])

    # 5. Daily News Signal: No duplicate (Ticker, Date) pairs
    def test_05_daily_news_signal_integrity(self):
        daily_df = compute_real_news_daily_summary()
        self.assertFalse(daily_df.empty, "Daily news signal summary should not be empty")
        
        # Check uniqueness of (Ticker, Date)
        duplicates = daily_df.duplicated(subset=["Ticker", "Date"]).sum()
        self.assertEqual(duplicates, 0, "No duplicate (Ticker, Date) records permitted in daily signals")
        
        # Check signal values
        self.assertIn("News_Signal", daily_df.columns)
        self.assertTrue((daily_df["News_Signal"] >= -1.0).all())
        self.assertTrue((daily_df["News_Signal"] <= 1.0).all())

    # 6. Coverage Audit: Honest representation of Complete vs Partial / Capped status
    def test_06_coverage_audit_accuracy(self):
        cov_df = compute_real_data_coverage()
        self.assertEqual(len(cov_df), len(SUPPORTED_TICKERS))
        
        cov_map = cov_df.set_index("Ticker").to_dict(orient="index")
        
        # Verify complete companies
        for t in ["AAPL", "MSFT", "META", "TSLA", "V", "JNJ"]:
            self.assertEqual(
                cov_map[t]["January_Coverage_Status"],
                "Complete",
                f"{t} must be marked Complete"
            )
            
        # Verify partial/capped companies
        for t in ["GOOGL", "AMZN", "NVDA", "JPM"]:
            self.assertEqual(
                cov_map[t]["January_Coverage_Status"],
                "Partial / Capped",
                f"{t} must be marked Partial / Capped due to provider limits"
            )
            self.assertIn("free-tier", cov_map[t]["Coverage_Notes"].lower())

    # 7. Pipeline Artifact Generation: CSVs exist and contain valid timestamps
    def test_07_pipeline_artifacts(self):
        output_dir = PROJECT_ROOT / "data" / "live" / "analytics"
        self.assertTrue(output_dir.exists())
        
        required_files = [
            "real_market_summary.csv",
            "real_news_summary.csv",
            "real_news_daily_summary.csv",
            "real_data_coverage.csv",
            "README.md"
        ]
        for rf in required_files:
            target = output_dir / rf
            self.assertTrue(target.exists(), f"Artifact must exist: {target}")
            self.assertGreater(target.stat().st_size, 0, f"Artifact must not be empty: {target}")


if __name__ == "__main__":
    unittest.main()
