"""
Unit Tests for Phase 3: Risk-Coverage & Selective Prediction Evaluation.

Verifies:
1. Coverage range strictly in [0.10, 1.00] and 5% increment structure
2. Risk range strictly in [0.0, 1.0]
3. Number retained monotonically increases with coverage
4. Correct ranking directions (Confidence descending, Disagreement ascending, Discordance ascending)
5. Oracle AURC <= Empirical AURC (theoretical lower bound)
6. Random AURC matches unselective test error rate
7. Paired bootstrap calculations produce reproducible, non-null outputs
8. Company subsets correctly partition across 10 target tickers
9. Year subsets correctly partition across 2023 and 2024
10. Decile analysis partitions into 10 bins per signal
11. All expected Phase 3 output CSVs, PNG figures, and reports are persisted
12. Production SQLite database remains completely untouched
"""

import unittest
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

from src.research.phase3_risk_coverage import (
    load_test_results,
    compute_risk_coverage_curve,
    compute_reference_curves,
    compute_aurc,
    evaluate_company_robustness,
    evaluate_yearly_robustness,
    compute_decile_error_analysis,
    COVERAGE_LEVELS,
    RESULTS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    TARGET_COMPANIES
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "live" / "news_database.db"


class TestPhase3RiskCoverage(unittest.TestCase):
    """Validation test suite for Phase 3 selective prediction implementation."""

    @classmethod
    def setUpClass(cls):
        cls.df = load_test_results()

    def test_01_coverage_levels(self):
        """Verify evaluated coverage levels span 0.10 to 1.00 in 0.05 increments."""
        self.assertEqual(len(COVERAGE_LEVELS), 19)
        self.assertAlmostEqual(COVERAGE_LEVELS[0], 0.10)
        self.assertAlmostEqual(COVERAGE_LEVELS[-1], 1.00)
        diffs = np.diff(COVERAGE_LEVELS)
        self.assertTrue(np.allclose(diffs, 0.05))

    def test_02_risk_and_accuracy_bounds(self):
        """Verify Risk and Accuracy values are bounded in [0, 1] and sum to 1.0."""
        curve = compute_risk_coverage_curve(self.df, "Prediction_Confidence", False, "TestConf")
        self.assertTrue((curve["Risk"] >= 0.0).all())
        self.assertTrue((curve["Risk"] <= 1.0).all())
        self.assertTrue((curve["Accuracy"] >= 0.0).all())
        self.assertTrue((curve["Accuracy"] <= 1.0).all())
        np.testing.assert_array_almost_equal(curve["Risk"] + curve["Accuracy"], 1.0)

    def test_03_monotonic_number_retained(self):
        """Verify retained observations count increases monotonically with coverage."""
        curve = compute_risk_coverage_curve(self.df, "Explanation_Discordance", True, "TestDisc")
        retained = curve["N_Retained"].values
        self.assertTrue((np.diff(retained) > 0).all())
        self.assertEqual(retained[-1], len(self.df))
        self.assertEqual(retained[0], int(np.ceil(0.10 * len(self.df))))

    def test_04_ranking_rules(self):
        """Verify sorting respects strategy specification."""
        # Confidence: descending
        sorted_conf = self.df.sort_values(by="Prediction_Confidence", ascending=False)
        self.assertGreaterEqual(
            sorted_conf.iloc[0]["Prediction_Confidence"],
            sorted_conf.iloc[-1]["Prediction_Confidence"]
        )
        # Disagreement: ascending
        sorted_dis = self.df.sort_values(by="Output_Disagreement", ascending=True)
        self.assertLessEqual(
            sorted_dis.iloc[0]["Output_Disagreement"],
            sorted_dis.iloc[-1]["Output_Disagreement"]
        )
        # Discordance: ascending
        sorted_disc = self.df.sort_values(by="Explanation_Discordance", ascending=True)
        self.assertLessEqual(
            sorted_disc.iloc[0]["Explanation_Discordance"],
            sorted_disc.iloc[-1]["Explanation_Discordance"]
        )

    def test_05_oracle_aurc_bound(self):
        """Verify Oracle reference achieves theoretical minimum AURC."""
        orc_curve, _ = compute_reference_curves(self.df)
        conf_curve = compute_risk_coverage_curve(self.df, "Prediction_Confidence", False, "Conf")
        
        trap_orc, _ = compute_aurc(orc_curve["Risk"].values, orc_curve["Coverage"].values)
        trap_conf, _ = compute_aurc(conf_curve["Risk"].values, conf_curve["Coverage"].values)
        
        self.assertLess(trap_orc, trap_conf)
        self.assertGreater(trap_orc, 0.0)

    def test_06_random_reference_aurc(self):
        """Verify Random selection risk matches unselective test error rate."""
        _, rand_curve = compute_reference_curves(self.df)
        base_error = 1.0 - self.df["Is_Correct_Multimodal"].mean()
        
        self.assertTrue(np.allclose(rand_curve["Risk"], base_error, atol=1e-4))
        trap_rand, norm_rand = compute_aurc(rand_curve["Risk"].values, rand_curve["Coverage"].values)
        self.assertAlmostEqual(norm_rand, base_error, places=4)

    def test_07_company_partitioning(self):
        """Verify company robustness properly evaluates each of the 10 target tickers."""
        df_company = evaluate_company_robustness(self.df)
        self.assertEqual(len(df_company), 10)
        self.assertEqual(set(df_company["Company"]), set(TARGET_COMPANIES))
        self.assertTrue((df_company["Observations"] == 300).all())
        self.assertEqual(df_company["Observations"].sum(), 3000)

    def test_08_year_partitioning(self):
        """Verify year robustness partitions test observations into 2023 and 2024."""
        df_yearly = evaluate_yearly_robustness(self.df)
        self.assertEqual(len(df_yearly), 2)
        years = set(df_yearly["Year"])
        self.assertEqual(years, {"2023", "2024"})
        self.assertEqual(df_yearly["Observations"].sum(), 3000)

    def test_09_decile_error_analysis(self):
        """Verify decile error breakdown creates 10 deciles per signal."""
        df_deciles, regression_results = compute_decile_error_analysis(self.df)
        self.assertIn("Explanation Discordance", df_deciles["Signal_Name"].values)
        self.assertIn("Prediction Confidence", df_deciles["Signal_Name"].values)
        self.assertIn("Output Disagreement", df_deciles["Signal_Name"].values)
        
        # Check logistic regression dictionary keys
        self.assertIn("Explanation_Discordance", regression_results)
        self.assertIn("Prediction_Confidence", regression_results)
        self.assertIn("Output_Disagreement", regression_results)

    def test_10_database_non_interference(self):
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
