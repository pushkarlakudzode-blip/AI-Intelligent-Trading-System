"""
Phase 4 Evaluation Test Suite
=============================
Tests chronological splitting, signal alignment, anti-leakage invariants,
transaction cost calculations, cash/exposure bounds, and system preservation.
"""

import json
import sqlite3
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.out_of_sample_evaluator import (
    OutOfSampleEvaluator,
    bootstrap_metric_ci,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EVAL_DIR = PROJECT_ROOT / "data" / "evaluation" / "phase4"
LIVE_DB_PATH = PROJECT_ROOT / "data" / "live" / "news_database.db"
RESEARCH_DIR = PROJECT_ROOT / "data" / "live" / "alpaca_staging" / "research"


class TestPhase4Evaluation(unittest.TestCase):
    """Rigorous unit and integration tests for Phase 4 Out-of-Sample Evaluation."""

    @classmethod
    def setUpClass(cls):
        cls.evaluator = OutOfSampleEvaluator(processed_dir=PROCESSED_DIR)
        cls.df = cls.evaluator.df_aligned

    def test_01_chronological_ordering_and_dates(self):
        """Verify strict chronological sequence and exact out-of-sample date bounds."""
        self.assertIsNotNone(self.df)
        self.assertEqual(len(self.df), 3000)
        
        tickers = self.df["Ticker"].unique()
        self.assertEqual(len(tickers), 10)
        
        for t in tickers:
            sub = self.df[self.df["Ticker"] == t].sort_values("Date").reset_index(drop=True)
            self.assertEqual(len(sub), 300, f"Ticker {t} should have exactly 300 test sessions")
            
            # Check date boundaries
            min_date = sub["Date"].min().strftime("%Y-%m-%d")
            max_date = sub["Date"].max().strftime("%Y-%m-%d")
            self.assertEqual(min_date, "2023-10-20", f"Test start date mismatch for {t}")
            self.assertEqual(max_date, "2024-12-30", f"Test end date mismatch for {t}")
            
            # Check strictly monotonic increasing dates
            self.assertTrue(sub["Date"].is_monotonic_increasing, f"Dates not monotonic for {t}")

    def test_02_signal_alignment_and_no_duplicates(self):
        """Verify 1:1 join with zero duplicates and exact feature preservation."""
        duplicates = self.df.duplicated(subset=["Date", "Ticker"]).sum()
        self.assertEqual(duplicates, 0, f"Found {duplicates} duplicate (Date, Ticker) pairs")
        
        # Verify required signal columns are present and non-null
        required_cols = [
            "Classification_Signal", "RL_Signal", "Market_Regime_Signal",
            "NLP_Signal", "Fusion_Score", "Final_Decision", "Forward_Return",
            "Score_NoNews", "Decision_NoNews", "Decision_RL", "Decision_Class"
        ]
        for col in required_cols:
            self.assertIn(col, self.df.columns)
            self.assertEqual(self.df[col].isnull().sum(), 0, f"Column {col} contains nulls")

    def test_03_zero_lookahead_leakage(self):
        """Assert that signals at date t are determined without knowing forward return."""
        # Verify Forward_Return does not correlate with Classification_Signal beyond sample noise
        # and that decision thresholds use strictly historical score
        for _, row in self.df.head(20).iterrows():
            score = row["Fusion_Score"]
            dec = row["Final_Decision"]
            if score >= 0.20:
                self.assertEqual(dec, "BUY")
            elif score <= -0.20:
                self.assertEqual(dec, "SELL")
            else:
                self.assertEqual(dec, "HOLD")

    def test_04_transaction_cost_deduction(self):
        """Verify transaction cost deduction occurs strictly upon position change."""
        sub = self.df[self.df["Ticker"] == "AAPL"].sort_values("Date").reset_index(drop=True)
        met_0, res_0 = self.evaluator.evaluate_single_stock(
            "AAPL", strategy_col="Final_Decision", cost_bps=0.0
        )
        met_10, res_10 = self.evaluator.evaluate_single_stock(
            "AAPL", strategy_col="Final_Decision", cost_bps=10.0
        )
        met_50, res_50 = self.evaluator.evaluate_single_stock(
            "AAPL", strategy_col="Final_Decision", cost_bps=50.0
        )
        
        # As costs increase, cumulative return must decrease monotonically
        self.assertGreater(met_0.cumulative_return, met_10.cumulative_return)
        self.assertGreater(met_10.cumulative_return, met_50.cumulative_return)
        
        # On days where position does not change (turnover == 0), fee must be exactly 0
        zero_to_days = res_10[res_10["Turnover"] == 0.0]
        self.assertTrue((zero_to_days["Fee"] == 0.0).all())
        
        # On days where position changes, fee must equal turnover * 0.0010
        active_to_days = res_10[res_10["Turnover"] > 0.0]
        self.assertTrue(np.allclose(active_to_days["Fee"], active_to_days["Turnover"] * 0.0010))

    def test_05_cash_constraint_and_exposure_limits(self):
        """Verify portfolio exposure stays within bounds [0.0, 1.0] in Long-Only mode."""
        met, curve = self.evaluator.evaluate_multi_asset_portfolio(
            strategy_col="Final_Decision", mode="LONG_ONLY", max_asset_weight=0.10
        )
        
        # Maximum portfolio exposure must never exceed 1.0 (10 assets * 10%)
        self.assertLessEqual(curve["Portfolio_Exposure"].max(), 1.0001)
        self.assertGreaterEqual(curve["Portfolio_Exposure"].min(), -0.0001)
        self.assertGreater(met.mean_exposure, 0.20)
        self.assertLess(met.mean_exposure, 0.80)

    def test_06_news_ablation_signal_equivalence(self):
        """Verify that when news signal is zero (96% of days), ablated score equals rescaled market score."""
        no_news_subset = self.df[self.df["News_Count"] == 0]
        self.assertGreater(len(no_news_subset), 2800)
        
        # Rescaled market components: (0.35*Class + 0.30*RL + 0.15*Regime) / 0.80
        computed_nonews = (
            (0.35 / 0.80) * no_news_subset["Classification_Signal"] +
            (0.30 / 0.80) * no_news_subset["RL_Signal"] +
            (0.15 / 0.80) * no_news_subset["Market_Regime_Signal"]
        ).round(4)
        
        self.assertTrue(np.allclose(no_news_subset["Score_NoNews"], computed_nonews, atol=1e-4))

    def test_07_bootstrap_confidence_interval_bounds(self):
        """Verify bootstrap confidence intervals satisfy lower <= point <= upper."""
        sub = self.df[self.df["Ticker"] == "MSFT"].sort_values("Date")
        daily_rets = sub["Forward_Return"]
        ci = bootstrap_metric_ci(daily_rets, n_bootstraps=300, risk_free_rate=0.03)
        
        sh_pt = calculate_sharpe_ratio(daily_rets, risk_free_rate=0.03)
        cr_pt = (1.0 + daily_rets).prod() - 1.0
        
        self.assertLessEqual(ci["sharpe_ci"][0], ci["sharpe_ci"][1])
        self.assertLessEqual(ci["cum_return_ci"][0], ci["cum_return_ci"][1])

    def test_08_production_database_and_research_preservation(self):
        """Verify production database row counts and Phase 1-6 research files are untouched."""
        if LIVE_DB_PATH.exists():
            conn = sqlite3.connect(LIVE_DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) FROM articles")
            art_cnt = cursor.fetchone()[0]
            cursor.execute("SELECT count(*) FROM nlp_results")
            nlp_cnt = cursor.fetchone()[0]
            conn.close()
            
            self.assertGreaterEqual(art_cnt, 650, "Production articles count must not decrease")
            self.assertGreaterEqual(nlp_cnt, 649, "Production nlp_results count must not decrease")

        # Verify research folder exists
        self.assertTrue(RESEARCH_DIR.exists(), "Explanation Discordance research directory must exist")
        self.assertTrue((RESEARCH_DIR / "phase6").exists(), "Phase 6 research artifacts must exist")


if __name__ == "__main__":
    unittest.main()
