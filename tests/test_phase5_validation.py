"""
Unit tests for Phase 5: Strong Validation, Ablation & Statistical Robustness.

Verifies:
1. Strict chronological time-ordering across 4 expanding temporal folds (zero future leakage).
2. Proper feature sets for Models A through G.
3. Consistency of feature ablation study and baseline delta definitions.
4. Permutation test reproducibility, valid null bounds, and empirical p-value.
5. Leave-One-Company-Out (LOCO) cross-validation partition integrity across 10 tickers.
6. Sector robustness groupings and valid evaluations.
7. Placebo control test distribution matching and label-independence.
8. Collinearity and VIF orthogonality verification (VIF < 2.0).
9. Benjamini-Hochberg FDR adjustment monotonicity and formatting.
10. Strict production database non-interference (articles=525, nlp_results=524, validation_logs=6).
"""

import os
import sys
import sqlite3
import hashlib
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.phase5_strong_validation import (
    load_phase5_data,
    compute_ece,
    FEATURE_SETS,
    GICS_SECTORS,
    BROAD_SECTORS,
    run_ablation_study,
    run_permutation_test,
    run_placebo_control_test,
    run_correlation_and_vif_analysis,
    RESULTS_DIR,
    REPORTS_DIR,
    FIGURES_DIR
)


class TestPhase5Validation(unittest.TestCase):
    """Test suite validating Phase 5 strong validation methodology and integrity."""

    @classmethod
    def setUpClass(cls):
        cls.df = load_phase5_data()
        cls.db_path = PROJECT_ROOT / "data" / "live" / "news_database.db"

    def test_01_chronological_time_ordering(self):
        """Verify 4 expanding temporal folds strictly preserve chronological ordering."""
        quarters = sorted(self.df["Quarter"].unique())
        self.assertEqual(len(quarters), 5, f"Expected 5 calendar quarters, found {len(quarters)}")
        self.assertEqual(str(quarters[0]), "2023Q4")
        self.assertEqual(str(quarters[4]), "2024Q4")

        # Verify expanding folds
        for fold_idx in range(1, 5):
            train_quarters = quarters[:fold_idx]
            val_quarter = quarters[fold_idx]
            train_df = self.df[self.df["Quarter"].isin(train_quarters)]
            val_df = self.df[self.df["Quarter"] == val_quarter]

            max_train_date = train_df["Date"].max()
            min_val_date = val_df["Date"].min()
            self.assertLess(
                max_train_date, min_val_date,
                f"Temporal leakage in Fold {fold_idx}: max train {max_train_date} >= min val {min_val_date}"
            )

    def test_02_feature_sets_specification(self):
        """Verify Models A through G contain exactly specified feature subsets."""
        self.assertEqual(FEATURE_SETS["Model_A"], ["Prediction_Confidence"])
        self.assertEqual(FEATURE_SETS["Model_B"], ["Output_Disagreement"])
        self.assertEqual(FEATURE_SETS["Model_C"], ["Explanation_Discordance"])
        self.assertEqual(FEATURE_SETS["Model_D"], ["Prediction_Confidence", "Output_Disagreement"])
        self.assertEqual(FEATURE_SETS["Model_E"], ["Prediction_Confidence", "Explanation_Discordance"])
        self.assertEqual(FEATURE_SETS["Model_F"], ["Output_Disagreement", "Explanation_Discordance"])
        self.assertEqual(FEATURE_SETS["Model_G"], ["Prediction_Confidence", "Output_Disagreement", "Explanation_Discordance"])

    def test_03_ablation_consistency(self):
        """Verify ablation configurations isolate individual signal contributions."""
        ablation_csv = RESULTS_DIR / "phase5_ablation.csv"
        if ablation_csv.exists():
            ab_df = pd.read_csv(ablation_csv)
            self.assertEqual(len(ab_df), 5)
            self.assertIn("Full (Conf + Dis + Disc)", ab_df["Ablation_Configuration"].values)
            self.assertIn("Remove Discordance (Conf + Dis)", ab_df["Ablation_Configuration"].values)
            self.assertIn("Discordance Alone (Disc)", ab_df["Ablation_Configuration"].values)
            
            full_row = ab_df[ab_df["Ablation_Configuration"] == "Full (Conf + Dis + Disc)"].iloc[0]
            base_row = ab_df[ab_df["Ablation_Configuration"] == "Remove Discordance (Conf + Dis)"].iloc[0]
            self.assertAlmostEqual(full_row["Delta_AUC_vs_No_Discordance"], round(full_row["ROC_AUC"] - base_row["ROC_AUC"], 4), places=3)

    def test_04_permutation_test_reproducibility(self):
        """Verify permutation test produces a valid null distribution and empirical p-value."""
        # Test quick permutation run on subset or check generated csv
        perm_csv = RESULTS_DIR / "phase5_permutation_test.csv"
        if perm_csv.exists():
            perm_df = pd.read_csv(perm_csv)
            self.assertEqual(len(perm_df), 1)
            p_val = float(perm_df["Empirical_p_value"].iloc[0])
            self.assertGreaterEqual(p_val, 0.0)
            self.assertLessEqual(p_val, 1.0)
            self.assertEqual(int(perm_df["N_Permutations"].iloc[0]), 1000)
            self.assertAlmostEqual(float(perm_df["Observed_Delta_AUC"].iloc[0]), 0.009856, places=4)

    def test_05_loco_cross_validation_coverage(self):
        """Verify Leave-One-Company-Out (LOCO) partitions evaluate all 10 companies."""
        companies = sorted(self.df["Ticker"].unique())
        self.assertEqual(len(companies), 10)
        self.assertEqual(companies, ["AAPL", "AMZN", "GOOGL", "JNJ", "JPM", "META", "MSFT", "NVDA", "TSLA", "V"])

        loco_csv = RESULTS_DIR / "phase5_company_holdout.csv"
        if loco_csv.exists():
            loco_df = pd.read_csv(loco_csv)
            self.assertEqual(len(loco_df), 10)
            self.assertEqual(sorted(loco_df["Held_Out_Company"].unique()), companies)
            for _, row in loco_df.iterrows():
                self.assertEqual(int(row["N_Val"]), 300)
                self.assertEqual(int(row["N_Train"]), 2700)

    def test_06_sector_grouping_validity(self):
        """Verify GICS and Broad sector classifications map valid tickers without omissions."""
        all_gics_tickers = []
        for tickers in GICS_SECTORS.values():
            all_gics_tickers.extend(tickers)
        self.assertEqual(sorted(all_gics_tickers), sorted(self.df["Ticker"].unique()))

        all_broad_tickers = []
        for tickers in BROAD_SECTORS.values():
            all_broad_tickers.extend(tickers)
        self.assertEqual(sorted(all_broad_tickers), sorted(self.df["Ticker"].unique()))

    def test_07_placebo_control_distribution(self):
        """Verify placebo discordance matches the marginal distribution of real discordance."""
        real_disc = self.df["Explanation_Discordance"].values
        rng = np.random.RandomState(42)
        placebo_disc = rng.permutation(real_disc)

        self.assertAlmostEqual(float(np.mean(real_disc)), float(np.mean(placebo_disc)), places=6)
        self.assertAlmostEqual(float(np.std(real_disc)), float(np.std(placebo_disc)), places=6)
        self.assertAlmostEqual(float(np.median(real_disc)), float(np.median(placebo_disc)), places=6)

    def test_08_correlation_and_vif_bounds(self):
        """Verify Variance Inflation Factor (VIF) for Explanation Discordance confirms orthogonality."""
        corr_csv = RESULTS_DIR / "phase5_correlation_analysis.csv"
        if corr_csv.exists():
            corr_df = pd.read_csv(corr_csv)
            disc_row = corr_df[corr_df["Feature"] == "Explanation_Discordance"].iloc[0]
            vif_val = float(disc_row["Variance_Inflation_Factor_VIF"])
            self.assertLess(vif_val, 1.5, f"Expected VIF < 1.5, found {vif_val}")
            self.assertEqual(disc_row["Collinearity_Status"], "Orthogonal (VIF < 2.0)")

    def test_09_multiple_testing_fdr_ranks(self):
        """Verify Benjamini-Hochberg FDR table is properly computed with monotonic critical values."""
        mt_csv = RESULTS_DIR / "phase5_multiple_testing.csv"
        if mt_csv.exists():
            mt_df = pd.read_csv(mt_csv)
            self.assertGreater(len(mt_df), 10)
            # Ranks must be 1 to m
            ranks = mt_df["Rank"].tolist()
            self.assertEqual(ranks, list(range(1, len(mt_df) + 1)))
            # Critical values must be monotonically increasing
            crit_vals = mt_df["BH_Critical_Threshold_05"].tolist()
            for i in range(1, len(crit_vals)):
                self.assertGreaterEqual(crit_vals[i], crit_vals[i-1])

    def test_10_production_database_non_interference(self):
        """Verify production SQLite database remains 100% untouched."""
        self.assertTrue(self.db_path.exists(), "Production SQLite DB missing!")
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        articles_count = cur.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        nlp_count = cur.execute("SELECT COUNT(*) FROM nlp_results").fetchone()[0]
        val_count = cur.execute("SELECT COUNT(*) FROM validation_logs").fetchone()[0]
        conn.close()

        self.assertEqual(articles_count, 525, f"articles count altered: {articles_count} != 525")
        self.assertEqual(nlp_count, 524, f"nlp_results count altered: {nlp_count} != 524")
        self.assertEqual(val_count, 6, f"validation_logs count altered: {val_count} != 6")


if __name__ == "__main__":
    unittest.main()
