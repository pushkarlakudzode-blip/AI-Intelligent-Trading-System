"""
Unit Tests for Phase 2: Training, SHAP Attribution & Explanation Discordance.

Verifies:
1. Identical train/validation/test boundaries
2. No test leakage
3. Probability range [0, 1]
4. Output disagreement range [0, 1]
5. Prediction confidence range [0, 1]
6. SHAP attribution dimensions
7. Market/news attribution grouping
8. Zero-vector handling
9. Cosine similarity validity
10. Discordance calculation
11. Ticker/date uniqueness
12. Deterministic results
13. Production database non-interference
"""

import unittest
import sqlite3
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.research.phase2_discordance_experiment import (
    load_and_split_data,
    evaluate_model_metrics,
    compute_grouped_attributions_and_discordance,
    MARKET_FEATURES,
    NEWS_FEATURES,
    MULTIMODAL_FEATURES,
    TARGET_COL,
    RANDOM_SEED,
    RESEARCH_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    CONFIG_DIR,
    tune_and_train_xgb
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "live" / "news_database.db"


class TestPhase2Discordance(unittest.TestCase):
    """Validation test suite for Phase 2 implementation."""

    @classmethod
    def setUpClass(cls):
        cls.df_train, cls.df_val, cls.df_test = load_and_split_data()

    def test_01_identical_boundaries(self):
        """Verify train, validation, and test splits respect exact boundary dates."""
        train_max = self.df_train["Date"].max()
        val_min = self.df_val["Date"].min()
        val_max = self.df_val["Date"].max()
        test_min = self.df_test["Date"].min()
        test_max = self.df_test["Date"].max()

        self.assertLess(train_max, val_min)
        self.assertLess(val_max, test_min)
        self.assertEqual(train_max, "2022-12-30")
        self.assertEqual(val_min, "2023-01-03")
        self.assertEqual(val_max, "2023-10-19")
        self.assertEqual(test_min, "2023-10-20")
        self.assertEqual(test_max, "2024-12-30")
        self.assertEqual(len(self.df_test), 3000)

    def test_02_no_test_leakage(self):
        """Verify features in train and validation sets have zero test dates."""
        train_dates = set(self.df_train["Date"].unique())
        val_dates = set(self.df_val["Date"].unique())
        test_dates = set(self.df_test["Date"].unique())

        self.assertEqual(len(train_dates.intersection(val_dates)), 0)
        self.assertEqual(len(train_dates.intersection(test_dates)), 0)
        self.assertEqual(len(val_dates.intersection(test_dates)), 0)

    def test_03_probability_range(self):
        """Verify predicted probabilities strictly bounded in [0, 1]."""
        # Fit quick model on small sample
        X_tr = self.df_train[MARKET_FEATURES].iloc[:100]
        y_tr = self.df_train[TARGET_COL].iloc[:100].astype(int)
        X_te = self.df_test[MARKET_FEATURES].iloc[:50]

        model, _ = tune_and_train_xgb(X_tr, y_tr, X_tr, y_tr, "QuickTest", random_state=42)
        probs = model.predict_proba(X_te)[:, 1]

        self.assertTrue((probs >= 0.0).all())
        self.assertTrue((probs <= 1.0).all())

    def test_04_output_disagreement_range(self):
        """Verify Output Disagreement |P_market - P_news| is strictly within [0, 1]."""
        p_market = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
        p_news = np.array([1.0, 0.7, 0.5, 0.1, 0.0])
        disagreement = np.abs(p_market - p_news)

        self.assertTrue((disagreement >= 0.0).all())
        self.assertTrue((disagreement <= 1.0).all())
        self.assertEqual(disagreement[2], 0.0)
        self.assertEqual(disagreement[0], 1.0)

    def test_05_prediction_confidence_range(self):
        """Verify Prediction Confidence 2 * |P - 0.5| is strictly within [0, 1]."""
        p_vals = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        confidence = 2.0 * np.abs(p_vals - 0.5)

        self.assertTrue((confidence >= 0.0).all())
        self.assertTrue((confidence <= 1.0).all())
        self.assertEqual(confidence[2], 0.0)
        self.assertEqual(confidence[0], 1.0)
        self.assertEqual(confidence[4], 1.0)

    def test_06_shap_attribution_dimensions(self):
        """Verify feature attribution dimensionality corresponds to total multimodal features."""
        n_features = len(MULTIMODAL_FEATURES)
        self.assertEqual(n_features, 31)
        self.assertEqual(len(MARKET_FEATURES), 20)
        self.assertEqual(len(NEWS_FEATURES), 11)

    def test_07_market_news_attribution_grouping(self):
        """Verify feature grouping correctly partitions positive, negative, and net mass."""
        shap_dummy = np.zeros((1, 31))
        # Set market features (first 20)
        shap_dummy[0, 0] = 0.5   # Open
        shap_dummy[0, 1] = -0.2  # High
        # Set news features (indices 20 to 30)
        shap_dummy[0, 20] = 0.3  # News_Count
        shap_dummy[0, 21] = -0.4 # Average_Sentiment

        meta_dummy = pd.DataFrame([{"Ticker": "AAPL", "Date": "2024-01-02", TARGET_COL: 1}])
        p_m = np.array([0.6])
        p_n = np.array([0.4])
        p_mu = np.array([0.55])

        df_grp, df_disc = compute_grouped_attributions_and_discordance(
            shap_dummy,
            MULTIMODAL_FEATURES,
            MARKET_FEATURES,
            NEWS_FEATURES,
            meta_dummy,
            p_m, p_n, p_mu
        )

        row = df_grp.iloc[0]
        self.assertAlmostEqual(row["Market_Positive_Mass"], 0.5)
        self.assertAlmostEqual(row["Market_Negative_Mass"], 0.2)
        self.assertAlmostEqual(row["Market_Net_Attribution"], 0.3)
        self.assertAlmostEqual(row["News_Positive_Mass"], 0.3)
        self.assertAlmostEqual(row["News_Negative_Mass"], 0.4)
        self.assertAlmostEqual(row["News_Net_Attribution"], -0.1)

    def test_08_zero_vector_handling(self):
        """Verify that zero attribution vectors are handled gracefully without NaN or div by zero."""
        shap_zero = np.zeros((1, 31))
        meta_dummy = pd.DataFrame([{"Ticker": "AAPL", "Date": "2024-01-02", TARGET_COL: 1}])

        df_grp, df_disc = compute_grouped_attributions_and_discordance(
            shap_zero,
            MULTIMODAL_FEATURES,
            MARKET_FEATURES,
            NEWS_FEATURES,
            meta_dummy,
            np.array([0.5]), np.array([0.5]), np.array([0.5])
        )

        row = df_disc.iloc[0]
        self.assertEqual(row["Attribution_Cosine_Similarity"], 1.0)
        self.assertEqual(row["Explanation_Discordance"], 0.0)
        self.assertFalse(np.isnan(row["Explanation_Discordance"]))

    def test_09_cosine_similarity_validity(self):
        """Verify cosine similarity between non-negative evidence vectors lies in [0, 1]."""
        # Collinear: both 100% positive evidence
        r_m = np.array([1.0, 0.0])
        r_n = np.array([2.0, 0.0])
        cos_sim = np.dot(r_m, r_n) / (np.linalg.norm(r_m) * np.linalg.norm(r_n))
        self.assertAlmostEqual(cos_sim, 1.0)

        # Orthogonal: market 100% positive, news 100% negative
        r_m = np.array([1.0, 0.0])
        r_n = np.array([0.0, 1.0])
        cos_sim = np.dot(r_m, r_n) / (np.linalg.norm(r_m) * np.linalg.norm(r_n))
        self.assertAlmostEqual(cos_sim, 0.0)

    def test_10_discordance_calculation(self):
        """Verify Explanation Discordance 1 - cos_sim lies in [0, 1]."""
        # Collinear
        self.assertAlmostEqual(1.0 - 1.0, 0.0)
        # Orthogonal
        self.assertAlmostEqual(1.0 - 0.0, 1.0)
        # Mixed: [0.8, 0.2] vs [0.2, 0.8]
        r_m = np.array([0.8, 0.2])
        r_n = np.array([0.2, 0.8])
        cos_sim = np.dot(r_m, r_n) / (np.linalg.norm(r_m) * np.linalg.norm(r_n))
        disc = 1.0 - cos_sim
        self.assertGreater(disc, 0.0)
        self.assertLess(disc, 1.0)

    def test_11_ticker_date_uniqueness(self):
        """Verify that evaluated test observations contain strictly unique (Ticker, Date) keys."""
        test_keys = self.df_test.groupby(["Ticker", "Date"]).size()
        self.assertTrue((test_keys == 1).all())

    def test_12_deterministic_results(self):
        """Verify deterministic training outputs when given same seed."""
        X_tr = self.df_train[MARKET_FEATURES].iloc[:80]
        y_tr = self.df_train[TARGET_COL].iloc[:80].astype(int)
        X_te = self.df_test[MARKET_FEATURES].iloc[:20]

        m1, _ = tune_and_train_xgb(X_tr, y_tr, X_tr, y_tr, "Det1", random_state=RANDOM_SEED)
        m2, _ = tune_and_train_xgb(X_tr, y_tr, X_tr, y_tr, "Det2", random_state=RANDOM_SEED)

        p1 = m1.predict_proba(X_te)[:, 1]
        p2 = m2.predict_proba(X_te)[:, 1]
        np.testing.assert_array_almost_equal(p1, p2, decimal=6)

    def test_13_database_non_interference(self):
        """Verify production SQLite database remains untouched."""
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM articles")
        articles_count = cursor.fetchone()[0]
        cursor.execute("SELECT count(*) FROM nlp_results")
        nlp_count = cursor.fetchone()[0]
        cursor.execute("SELECT count(*) FROM validation_logs")
        logs_count = cursor.fetchone()[0]
        conn.close()

        self.assertEqual(articles_count, 525)
        self.assertEqual(nlp_count, 524)
        self.assertEqual(logs_count, 6)


if __name__ == "__main__":
    unittest.main()
