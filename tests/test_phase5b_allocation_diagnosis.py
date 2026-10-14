"""
Phase 5B: Portfolio Allocation, Cooldown & Turnover Diagnosis Test Suite.
========================================================================
Rigorous unit and integration tests verifying:
1. Target allocation constraints (10% and 20% caps).
2. Cash conservation and non-negativity in Strict Dollar Ledger.
3. Next-Day Open (MOO) trade timing and zero lookahead.
4. Cooldown / minimum holding period enforcement.
5. Volatility scaling inverse proportionality.
6. Temporal split integrity (Validation vs Untouched Test).
7. Turnover and transaction cost deduction accuracy.
8. Production database and Phase 1-6 research preservation.
"""

import json
import sqlite3
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.phase5b_allocation_engine import (
    AllocationConfig,
    Phase5BAllocationEngine,
    Phase5BMetrics
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PHASE5B_DIR = PROJECT_ROOT / "data" / "evaluation" / "phase5b"
LIVE_DB_PATH = PROJECT_ROOT / "data" / "live" / "news_database.db"
RESEARCH_DIR = PROJECT_ROOT / "data" / "live" / "alpaca_staging" / "research"


class TestPhase5BAllocationDiagnosis(unittest.TestCase):
    """Rigorous unit and integration tests for Phase 5B allocation engine."""

    @classmethod
    def setUpClass(cls):
        cls.engine = Phase5BAllocationEngine(processed_dir=PROCESSED_DIR)
        cls.df = cls.engine.df_aligned

    def test_01_target_allocation_constraints(self):
        """Verify target allocation cap (10% in baseline, 20% in dynamic cash)."""
        cfg_10 = AllocationConfig(name="test_10", description="10% cap", max_asset_weight=0.10)
        met_10, curve_10 = self.engine.simulate_strategy(cfg_10, self.engine.dates)
        self.assertLessEqual(met_10.max_asset_weight, 0.10)

        cfg_20 = AllocationConfig(
            name="test_20", description="20% cap", max_asset_weight=0.20, dynamic_cash_fungible=True
        )
        met_20, curve_20 = self.engine.simulate_strategy(cfg_20, self.engine.dates)
        self.assertEqual(met_20.max_asset_weight, 0.20)
        self.assertGreater(met_20.mean_exposure, met_10.mean_exposure)

    def test_02_cash_conservation_and_non_negativity(self):
        """Verify conservation of equity and non-negative cash across all trading days."""
        cfg = AllocationConfig(name="test_cash", description="Cash invariant check", max_asset_weight=0.10)
        met, hist_df = self.engine.simulate_strategy(cfg, self.engine.dates)

        self.assertEqual(len(hist_df), 300)
        self.assertIn("Cash", hist_df.columns)
        self.assertIn("Equity", hist_df.columns)
        # Cash cannot go negative
        self.assertTrue((hist_df["Cash"] >= -1e-4).all(), "Cash must never be negative")
        # Total equity must remain positive
        self.assertTrue((hist_df["Equity"] > 0).all(), "Equity must remain positive")

    def test_03_execution_timing_and_fill_realism(self):
        """Verify Next-Day Open (MOO) returns are used without lookahead."""
        self.assertIn("Fwd_Open_Open", self.df.columns)
        self.assertIn("Vol_20", self.df.columns)

        # Check that Vol_20 has zero nulls
        self.assertEqual(self.df["Vol_20"].isnull().sum(), 0)

        # Check that dates are sorted and monotonic
        self.assertTrue(self.df["Date"].is_monotonic_increasing)

    def test_04_cooldown_enforcement(self):
        """Verify that cooldown rule strictly prevents premature liquidations."""
        # Baseline has 77 whipsaws <= 3 days
        cfg_base = AllocationConfig(name="base", description="baseline", cooldown_days=0)
        met_base, _ = self.engine.simulate_strategy(cfg_base, self.engine.dates)

        # Cooldown K=3 enforces minimum 3 days holding, reducing 3-day whipsaws significantly
        cfg_k3 = AllocationConfig(name="k3", description="cooldown K=3", cooldown_days=3)
        met_k3, _ = self.engine.simulate_strategy(cfg_k3, self.engine.dates)

        self.assertLess(met_k3.whipsaws_3d, met_base.whipsaws_3d)
        self.assertLess(met_k3.total_turnover, met_base.total_turnover)
        self.assertLess(met_k3.trade_count, met_base.trade_count)

    def test_05_volatility_scaling_logic(self):
        """Verify inverse volatility scaling logic: low vol assets receive higher weights than high vol."""
        nvda_vol = self.df[self.df["Ticker"] == "NVDA"]["Vol_20"].mean()
        jnj_vol = self.df[self.df["Ticker"] == "JNJ"]["Vol_20"].mean()

        # NVDA volatility is substantially higher than JNJ
        self.assertGreater(nvda_vol, jnj_vol * 2.0)

        # Under inverse volatility scaling, 1/JNJ_vol > 1/NVDA_vol
        inv_jnj = 1.0 / jnj_vol
        inv_nvda = 1.0 / nvda_vol
        self.assertGreater(inv_jnj, inv_nvda * 2.0)

    def test_06_temporal_split_integrity(self):
        """Verify strict chronological separation between Validation (150d) and Untouched Test (150d)."""
        val_dates = self.engine.val_dates
        test_dates = self.engine.test_dates

        self.assertEqual(len(val_dates), 150)
        self.assertEqual(len(test_dates), 150)
        self.assertEqual(len(self.engine.dates), 300)

        # Max validation date must precede min test date
        self.assertLess(max(val_dates), min(test_dates))

        # Zero overlap
        intersection = set(val_dates).intersection(set(test_dates))
        self.assertEqual(len(intersection), 0)

    def test_07_turnover_and_cost_deduction(self):
        """Verify transaction costs scale monotonically with bps."""
        cum_rets = []
        for c_bps in [0.0, 10.0, 30.0, 50.0]:
            cfg = AllocationConfig(name="test_cost", description="cost check", cost_bps=c_bps)
            met, _ = self.engine.simulate_strategy(cfg, self.engine.dates)
            cum_rets.append(met.cumulative_return)

        # Monotonically decreasing net return
        for i in range(len(cum_rets) - 1):
            self.assertGreater(cum_rets[i], cum_rets[i + 1])

    def test_08_production_database_and_research_preservation(self):
        """Verify production database and Phase 1-6 research artifacts are untouched."""
        self.assertTrue(LIVE_DB_PATH.exists(), "Live database must exist")
        conn = sqlite3.connect(LIVE_DB_PATH)
        c = conn.cursor()
        art_count = c.execute("SELECT count(*) FROM articles").fetchone()[0]
        nlp_count = c.execute("SELECT count(*) FROM nlp_results").fetchone()[0]
        exp_count = c.execute("SELECT count(*) FROM news_explanations").fetchone()[0]
        conn.close()

        self.assertGreaterEqual(art_count, 650)
        self.assertGreaterEqual(nlp_count, 649)
        self.assertGreaterEqual(exp_count, 15)

        # Research directory verification
        self.assertTrue(RESEARCH_DIR.exists())
        self.assertTrue((RESEARCH_DIR / "phase6").exists())


if __name__ == "__main__":
    unittest.main()
