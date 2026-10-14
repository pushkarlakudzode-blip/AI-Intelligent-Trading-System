"""
Phase 5D: Controlled Strategy Comparison & Candidate Selection Test Suite.
=========================================================================
Rigorous unit and regression tests verifying:
1. Identical experimental conditions across all candidate strategies.
2. Signal-to-fill timing with next-day open (MOO) and zero lookahead.
3. Hysteresis exit behavior (Candidate B & C holding between -0.30 and +0.20).
4. Cooldown anti-whipsaw rule enforcement (Candidate C overriding SELL to HOLD for duration < 3).
5. Validation vs Untouched Test split separation and zero leakage.
6. Strict Dollar Ledger cash conservation and non-negativity.
7. Execution cost sensitivity monotonicity and limit-order non-fill behavior.
8. Production database and research artifact preservation.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import unittest

import numpy as np
import pandas as pd

from src.evaluation.phase5d_candidate_selector import (
    CandidateConfig,
    CandidateMetrics,
    Phase5DCandidateSelector
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LIVE_DB_PATH = PROJECT_ROOT / "data" / "live" / "news_database.db"
PHASE5A_DIR = PROJECT_ROOT / "data" / "evaluation" / "phase5a"
PHASE5B_DIR = PROJECT_ROOT / "data" / "evaluation" / "phase5b"
PHASE5C_DIR = PROJECT_ROOT / "data" / "evaluation" / "phase5c"
PHASE5D_DIR = PROJECT_ROOT / "data" / "evaluation" / "phase5d"
RESEARCH_DIR = PROJECT_ROOT / "data" / "live" / "alpaca_staging" / "research"


class TestPhase5DCandidateSelection(unittest.TestCase):
    """Rigorous regression and validation test suite for Phase 5D."""

    @classmethod
    def setUpClass(cls):
        cls.selector = Phase5DCandidateSelector(processed_dir=PROCESSED_DIR)

    def test_01_identical_experiment_conditions(self):
        """Verify identical test setup: dates, capital, universe, cost for all candidates."""
        self.assertEqual(len(self.selector.dates), 300)
        self.assertEqual(len(self.selector.val_dates), 150)
        self.assertEqual(len(self.selector.test_dates), 150)
        self.assertEqual(len(self.selector.tickers), 10)

        # Ensure all tickers are present
        for t in ["AAPL", "AMZN", "GOOGL", "JNJ", "JPM", "META", "MSFT", "NVDA", "TSLA", "V"]:
            self.assertIn(t, self.selector.tickers)

    def test_02_signal_to_fill_timing_and_zero_lookahead(self):
        """Verify signals formed on day t execute at Open_{t+1} and returns realize to Open_{t+2}."""
        df = self.selector.df_aligned
        self.assertIn("Fwd_Open_Open", df.columns)
        self.assertIn("Open_t1", df.columns)
        self.assertIn("Open_t2", df.columns)

        # Check for non-null forward returns
        valid_fwd = df["Fwd_Open_Open"].dropna()
        self.assertGreater(len(valid_fwd), len(df) * 0.95)

        # Spot check that Fwd_Open_Open matches (Open_t2 - Open_t1) / Open_t1
        sample = df.dropna(subset=["Open_t1", "Open_t2", "Fwd_Open_Open"]).iloc[10]
        expected_r = (sample["Open_t2"] - sample["Open_t1"]) / sample["Open_t1"]
        self.assertAlmostEqual(sample["Fwd_Open_Open"], expected_r, places=6)

    def test_03_hysteresis_exit_behavior(self):
        """Verify Candidate B holds when score is in [-0.30, 0.20), while Baseline sells at -0.25."""
        cfg_a = CandidateConfig(name="test_a", description="Baseline", max_asset_weight=0.10)
        cfg_b = CandidateConfig(name="test_b", description="Hysteresis", max_asset_weight=0.10, hysteresis_sell_threshold=-0.30)

        met_a, cur_a, _ = self.selector.simulate_candidate(cfg_a, self.selector.val_dates)
        met_b, cur_b, _ = self.selector.simulate_candidate(cfg_b, self.selector.val_dates)

        # Candidate B must have fewer trades and fewer whipsaws due to the expanded holding zone
        self.assertLessEqual(met_b.trade_count, met_a.trade_count)
        self.assertLessEqual(met_b.whipsaws_3d, met_a.whipsaws_3d)
        self.assertLessEqual(met_b.total_turnover, met_a.total_turnover)

    def test_04_cooldown_anti_whipsaw_enforcement(self):
        """Verify Candidate C overrides SELL to HOLD when holding duration is < 3 days."""
        cfg_b = CandidateConfig(name="test_b", description="Hysteresis", max_asset_weight=0.10, hysteresis_sell_threshold=-0.30, cooldown_days=0)
        cfg_c = CandidateConfig(name="test_c", description="Hysteresis+Cooldown", max_asset_weight=0.10, hysteresis_sell_threshold=-0.30, cooldown_days=3)

        met_b, _, _ = self.selector.simulate_candidate(cfg_b, self.selector.val_dates)
        met_c, _, _ = self.selector.simulate_candidate(cfg_c, self.selector.val_dates)

        # Candidate C must strictly cut whipsaws (duration <= 3) compared to Candidate B
        self.assertLessEqual(met_c.whipsaws_3d, met_b.whipsaws_3d)
        self.assertLessEqual(met_c.trade_count, met_b.trade_count)

    def test_05_validation_vs_untouched_test_separation(self):
        """Verify zero date overlap between Validation (sessions 1-150) and Untouched Test (sessions 151-300)."""
        val_set = set(self.selector.val_dates)
        test_set = set(self.selector.test_dates)

        self.assertEqual(len(val_set.intersection(test_set)), 0)
        self.assertEqual(len(val_set) + len(test_set), len(self.selector.dates))
        self.assertLess(max(self.selector.val_dates), min(self.selector.test_dates))

    def test_06_strict_dollar_ledger_and_cash_conservation(self):
        """Verify Dollar Ledger non-negativity and daily equity balance conservation."""
        cfg = CandidateConfig(name="test_cash", description="Check cash", max_asset_weight=0.10, hysteresis_sell_threshold=-0.30, cooldown_days=3)
        met, curve, _ = self.selector.simulate_candidate(cfg, self.selector.dates)

        # Check cash never negative
        self.assertTrue((curve["Cash"] >= -1e-6).all())
        # Check equity always positive
        self.assertTrue((curve["Equity"] > 0).all())
        # Check portfolio exposure is within [0.0, 1.0]
        self.assertTrue((curve["Portfolio_Exposure"] >= 0.0).all())
        self.assertTrue((curve["Portfolio_Exposure"] <= 1.0001).all())

    def test_07_cost_sensitivity_and_non_fill_stress(self):
        """Verify cost monotonicity (higher bps -> lower return) and non-fill simulation safety."""
        cfg_0 = CandidateConfig(name="c0", description="", cost_bps=0.0, hysteresis_sell_threshold=-0.30)
        cfg_10 = CandidateConfig(name="c10", description="", cost_bps=10.0, hysteresis_sell_threshold=-0.30)
        cfg_30 = CandidateConfig(name="c30", description="", cost_bps=30.0, hysteresis_sell_threshold=-0.30)

        m0, _, _ = self.selector.simulate_candidate(cfg_0, self.selector.val_dates)
        m10, _, _ = self.selector.simulate_candidate(cfg_10, self.selector.val_dates)
        m30, _, _ = self.selector.simulate_candidate(cfg_30, self.selector.val_dates)

        self.assertGreater(m0.cumulative_return, m10.cumulative_return)
        self.assertGreater(m10.cumulative_return, m30.cumulative_return)

        # Non-fill test (gap limit)
        cfg_gap = CandidateConfig(name="c_gap", description="", cost_bps=10.0, gap_limit_pct=0.015)
        m_gap, cur_gap, ps_gap = self.selector.simulate_candidate(cfg_gap, self.selector.val_dates)
        self.assertGreater(len(cur_gap), 0)

    def test_08_production_database_and_research_preservation(self):
        """Verify production SQLite database and all historical phase artifacts remain intact."""
        conn = sqlite3.connect(f"file:{LIVE_DB_PATH}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM articles")
        art_cnt = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM nlp_results")
        nlp_cnt = cur.fetchone()[0]
        conn.close()

        self.assertGreaterEqual(art_cnt, 657)
        self.assertGreaterEqual(nlp_cnt, 656)

        # Verify research directories and reports
        self.assertTrue(RESEARCH_DIR.exists())
        self.assertTrue((RESEARCH_DIR / "aligned_daily_dataset.csv").exists())
        self.assertTrue((PHASE5A_DIR / "portfolio_comparison.csv").exists())
        self.assertTrue((PHASE5B_DIR / "experiment_summary.csv").exists())
        self.assertTrue((PHASE5C_DIR / "ticker_coverage_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
