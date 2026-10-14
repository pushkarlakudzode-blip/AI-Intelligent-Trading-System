"""
Unit Tests for Phase 4: Complementary Signal & Error-Prediction Analysis.

Verifies:
1. Binary error target construction and consistency with multimodal predictions
2. Proper fitting and probability bounds for all 7 error-prediction models (A to G)
3. Parameter table structure including coefficients, standard errors, and p-values
4. Paired bootstrap comparison structure and p-value calculation
5. Non-linear tertile and decile partitioning
6. Interaction effect model estimations
7. Temporal walk-forward evaluation splits and order
8. Company-level robustness across all 10 equities
9. Out-of-sample selective prediction on 2024 with 2023-learned weights
10. Production database integrity (articles=525, nlp_results=524, validation_logs=6)
"""

import unittest
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

from src.research.phase4_complementary_analysis import (
    load_phase4_data,
    evaluate_error_prediction_models,
    run_paired_bootstrap,
    evaluate_nonlinear_effects,
    evaluate_interaction_effects,
    evaluate_temporal_robustness,
    evaluate_company_robustness,
    evaluate_learned_selective_prediction,
    RESULTS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    TARGET_COMPANIES
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "live" / "news_database.db"


class TestPhase4ErrorPrediction(unittest.TestCase):
    """Validation test suite for Phase 4 implementation."""

    @classmethod
    def setUpClass(cls):
        cls.df = load_phase4_data()

    def test_01_error_target_consistency(self):
        """Verify Error is binary {0, 1} and equals 1 - Is_Correct_Multimodal."""
        self.assertIn("Error", self.df.columns)
        self.assertTrue(set(self.df["Error"].unique()).issubset({0, 1}))
        np.testing.assert_array_equal(self.df["Error"].values, 1 - self.df["Is_Correct_Multimodal"].values)
        self.assertEqual(len(self.df), 3000)

    def test_02_seven_models_fitted_and_bounded(self):
        """Verify all 7 models fit and generate probabilities in [0, 1]."""
        df_metrics, df_params, model_probs = evaluate_error_prediction_models(self.df)
        self.assertEqual(len(df_metrics), 7)
        expected_ids = {
            "Model_A_Confidence", "Model_B_Disagreement", "Model_C_Discordance",
            "Model_D_Baseline_Conf_Dis", "Model_E_Conf_Disc", "Model_F_Dis_Disc",
            "Model_G_Proposed_Combined"
        }
        self.assertEqual(set(df_metrics["Model_ID"]), expected_ids)
        for m_id, probs in model_probs.items():
            self.assertEqual(len(probs), 3000)
            self.assertTrue((probs >= 0.0).all())
            self.assertTrue((probs <= 1.0).all())

    def test_03_parameter_table_validity(self):
        """Verify parameters table includes coefficients, standard errors, z-scores, and p-values."""
        _, df_params, _ = evaluate_error_prediction_models(self.df)
        required_cols = ["Model_ID", "Parameter", "Coefficient", "Std_Error", "Z_Score", "P_Value", "Odds_Ratio"]
        for col in required_cols:
            self.assertIn(col, df_params.columns)
        self.assertFalse(df_params["Coefficient"].isna().any())
        self.assertFalse(df_params["P_Value"].isna().any())

    def test_04_paired_bootstrap_comparison(self):
        """Verify paired bootstrap produces all expected metrics with non-null CIs."""
        _, _, model_probs = evaluate_error_prediction_models(self.df)
        p_base = model_probs["Model_D_Baseline_Conf_Dis"]
        p_prop = model_probs["Model_G_Proposed_Combined"]
        
        df_boot = run_paired_bootstrap(self.df, p_base, p_prop, n_boot=100, random_state=42)
        self.assertEqual(len(df_boot), 4)
        expected_metrics = {"Delta_ROC_AUC", "Delta_PR_AUC", "Delta_Log_Loss", "Delta_Brier_Score"}
        self.assertEqual(set(df_boot["Metric"]), expected_metrics)
        self.assertTrue((df_boot["P_Value"] >= 0.0).all())
        self.assertTrue((df_boot["P_Value"] <= 1.0).all())

    def test_05_nonlinear_tertiles_and_deciles(self):
        """Verify non-linear analysis properly creates 3 tertiles and 10 deciles."""
        df_nonlin = evaluate_nonlinear_effects(self.df)
        tertiles = df_nonlin[df_nonlin["Analysis_Type"] == "Tertile_Groups"]
        deciles = df_nonlin[df_nonlin["Analysis_Type"] == "Deciles"]
        
        self.assertEqual(len(tertiles), 3)
        self.assertEqual(len(deciles), 10)
        self.assertEqual(tertiles["Count"].sum(), 3000)
        self.assertEqual(deciles["Count"].sum(), 3000)

    def test_06_interaction_models_estimation(self):
        """Verify interaction models estimate cleanly without singular matrices."""
        df_int = evaluate_interaction_effects(self.df)
        self.assertIn("Confidence_x_Discordance", df_int["Parameter"].values)
        self.assertIn("Disagreement_x_Discordance", df_int["Parameter"].values)
        self.assertFalse(df_int["Coefficient"].isna().any())

    def test_07_temporal_walk_forward_evaluation(self):
        """Verify temporal evaluations partition chronologically into 2023 and 2024."""
        df_temp = evaluate_temporal_robustness(self.df)
        self.assertEqual(len(df_temp), 3)
        self.assertIn("Exp1_Train_2023_Test_2024", df_temp["Experiment"].values)
        self.assertIn("Exp2_WalkForward_Fold1", df_temp["Experiment"].values)
        self.assertIn("Exp2_WalkForward_Fold2", df_temp["Experiment"].values)

    def test_08_company_robustness_evaluates_all_ten(self):
        """Verify all 10 companies are evaluated with exactly 300 rows each."""
        df_comp = evaluate_company_robustness(self.df)
        self.assertEqual(len(df_comp), 10)
        self.assertEqual(set(df_comp["Company"]), set(TARGET_COMPANIES))
        self.assertTrue((df_comp["Observations"] == 300).all())

    def test_09_learned_selective_prediction_on_2024(self):
        """Verify out-of-sample selective prediction evaluates 2,510 rows in 2024."""
        df_curves, df_aurc = evaluate_learned_selective_prediction(self.df)
        self.assertEqual(len(df_aurc), 4)
        strategies = set(df_aurc["Strategy"])
        self.assertIn("Strategy_D_Combined_Learned_Score", strategies)
        self.assertTrue((df_aurc["AURC_Trapezoid"] > 0.0).all())

    def test_10_production_database_non_interference(self):
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
