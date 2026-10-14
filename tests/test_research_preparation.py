"""
Unit Tests for Explanation Discordance Research Data Preparation.

Verifies:
1. Primary key uniqueness and ticker/date alignment
2. Forward target construction consistency (Next_Day_Return and Next_Day_Direction)
3. Zero lookahead leakage in market features
4. Zero lookahead leakage in news publication timestamps
5. Strict temporal ordering of train / val / test splits
6. Safe default handling for dates with missing news
7. Meta Platforms (FB / META) canonical ticker normalization
8. Deterministic creation and schema validity of all research datasets
9. Experiment configuration and attribution group specifications
10. Strict production database non-interference guarantee
"""

import sys
import unittest
import sqlite3
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.prepare_discordance_research import (
    init_research_directory,
    load_raw_inputs,
    construct_forward_targets,
    align_market_and_news,
    perform_temporal_leakage_audit,
    generate_coverage_report,
    prepare_and_save_research_datasets,
    generate_experiment_config,
    RESEARCH_DIR,
    TARGET_COMPANIES,
    TRAIN_END_DATE,
    VAL_END_DATE,
    TEST_START_DATE
)

PROD_DB_PATH = PROJECT_ROOT / "data" / "live" / "news_database.db"


class TestResearchPreparation(unittest.TestCase):

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

    def test_01_target_construction(self):
        """Verify construct_forward_targets correctly shifts next-day return and direction."""
        sample_data = pd.DataFrame({
            "Ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT"],
            "Date": ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-02", "2020-01-03"],
            "Close": [100.0, 105.0, 102.0, 200.0, 190.0]
        })
        res = construct_forward_targets(sample_data)
        
        # AAPL day 1: return = (105 - 100)/100 = 0.05, direction = 1
        self.assertAlmostEqual(res.loc[0, "Next_Day_Return"], 0.05, places=4)
        self.assertEqual(res.loc[0, "Next_Day_Direction"], 1.0)

        # AAPL day 2: return = (102 - 105)/105 = -0.028571, direction = 0
        self.assertAlmostEqual(res.loc[1, "Next_Day_Return"], -0.028571, places=4)
        self.assertEqual(res.loc[1, "Next_Day_Direction"], 0.0)

        # AAPL day 3 (last day): return = NaN, direction = NaN
        self.assertTrue(np.isnan(res.loc[2, "Next_Day_Return"]))
        self.assertTrue(np.isnan(res.loc[2, "Next_Day_Direction"]))

        # MSFT day 1: return = (190 - 200)/200 = -0.05, direction = 0
        self.assertAlmostEqual(res.loc[3, "Next_Day_Return"], -0.05, places=4)
        self.assertEqual(res.loc[3, "Next_Day_Direction"], 0.0)

    def test_02_ticker_date_uniqueness(self):
        """Verify aligned dataset has unique (Ticker, Date) keys."""
        df_feat, df_clust, df_news, df_fusion = load_raw_inputs()
        df_aligned, meta = align_market_and_news(df_feat, df_clust, df_news, df_fusion)
        self.assertEqual(df_aligned.duplicated(subset=["Ticker", "Date"]).sum(), 0)

    def test_03_no_future_market_features(self):
        """Verify market features derive strictly from date t and prior dates."""
        df_feat, df_clust, df_news, df_fusion = load_raw_inputs()
        df_aligned, meta = align_market_and_news(df_feat, df_clust, df_news, df_fusion)
        df_audit, audit_stats = perform_temporal_leakage_audit(df_aligned)
        
        mkt_check = df_audit[df_audit["Check_Name"] == "Market Features Current-Day Alignment"].iloc[0]
        self.assertEqual(mkt_check["Status"], "PASSED")
        self.assertEqual(mkt_check["Violations_Count"], 0)

    def test_04_no_future_news_leakage(self):
        """Verify news features derive strictly from articles published <= date t."""
        df_feat, df_clust, df_news, df_fusion = load_raw_inputs()
        df_aligned, meta = align_market_and_news(df_feat, df_clust, df_news, df_fusion)
        df_audit, audit_stats = perform_temporal_leakage_audit(df_aligned)
        
        news_check = df_audit[df_audit["Check_Name"] == "News Publication Timestamp Safeguard"].iloc[0]
        self.assertEqual(news_check["Status"], "PASSED")
        self.assertEqual(news_check["Violations_Count"], 0)

    def test_05_train_val_test_temporal_separation(self):
        """Verify strict chronological ordering of train, validation, and test splits."""
        df_feat, df_clust, df_news, df_fusion = load_raw_inputs()
        df_aligned, meta = align_market_and_news(df_feat, df_clust, df_news, df_fusion)
        
        train_dates = df_aligned[df_aligned["Split"] == "train"]["Date"]
        val_dates = df_aligned[df_aligned["Split"] == "val"]["Date"]
        test_dates = df_aligned[df_aligned["Split"] == "test"]["Date"]

        self.assertLess(train_dates.max(), val_dates.min())
        self.assertLess(val_dates.max(), test_dates.min())
        self.assertLessEqual(train_dates.max(), TRAIN_END_DATE)
        self.assertLessEqual(val_dates.max(), VAL_END_DATE)
        self.assertGreaterEqual(test_dates.min(), TEST_START_DATE)

    def test_06_missing_news_handling(self):
        """Verify trading days without news have well-defined zero-signal values."""
        df_feat, df_clust, df_news, df_fusion = load_raw_inputs()
        df_aligned, meta = align_market_and_news(df_feat, df_clust, df_news, df_fusion)
        
        no_news = df_aligned[df_aligned["Has_Same_Day_News"] == 0]
        self.assertGreater(len(no_news), 0)
        self.assertTrue((no_news["News_Count"] == 0).all())
        self.assertTrue((no_news["High_Impact_News_Count"] == 0).all())
        self.assertTrue((no_news["News_Signal"] == 0.0).all())
        self.assertTrue((no_news["Average_Sentiment"] == 0.0).all())
        self.assertTrue((no_news["Dominant_Sentiment"] == "None").all())

    def test_07_meta_fb_normalization(self):
        """Verify META encompasses all Facebook articles and no FB ticker rows exist."""
        df_feat, df_clust, df_news, df_fusion = load_raw_inputs()
        df_aligned, meta = align_market_and_news(df_feat, df_clust, df_news, df_fusion)
        
        self.assertIn("META", df_aligned["Ticker"].unique())
        self.assertNotIn("FB", df_aligned["Ticker"].unique())
        meta_rows = df_aligned[df_aligned["Ticker"] == "META"]
        self.assertEqual(len(meta_rows), 1501)

    def test_08_experiment_config_structure(self):
        """Verify experiment_config.json contains all required baseline groups and attribution groups."""
        df_feat, df_clust, df_news, df_fusion = load_raw_inputs()
        df_aligned, meta = align_market_and_news(df_feat, df_clust, df_news, df_fusion)
        config = generate_experiment_config(df_aligned)

        self.assertIn("baseline_groups", config)
        self.assertIn("A_market_only", config["baseline_groups"])
        self.assertIn("B_news_only", config["baseline_groups"])
        self.assertIn("C_multimodal", config["baseline_groups"])
        self.assertIn("D_signal_disagreement", config["baseline_groups"])
        self.assertIn("E_prediction_confidence", config["baseline_groups"])
        self.assertIn("F_proposed_explanation_discordance", config["baseline_groups"])

        self.assertIn("attribution_feature_groups", config)
        self.assertIn("technical_market_group", config["attribution_feature_groups"])
        self.assertIn("news_group", config["attribution_feature_groups"])

    def test_09_research_datasets_persisted(self):
        """Verify research datasets are properly written with correct rows and non-null identifiers."""
        df_feat, df_clust, df_news, df_fusion = load_raw_inputs()
        df_aligned, meta = align_market_and_news(df_feat, df_clust, df_news, df_fusion)
        paths = prepare_and_save_research_datasets(df_aligned)

        for name, p in paths.items():
            self.assertTrue(p.exists(), f"File does not exist: {p}")
            df_check = pd.read_csv(p)
            self.assertEqual(len(df_check), 15010)
            self.assertIn("Ticker", df_check.columns)
            self.assertIn("Date", df_check.columns)
            self.assertIn("Next_Day_Return", df_check.columns)
            self.assertIn("Next_Day_Direction", df_check.columns)


if __name__ == "__main__":
    unittest.main()
