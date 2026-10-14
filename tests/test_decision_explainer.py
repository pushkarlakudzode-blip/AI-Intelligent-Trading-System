"""
Unit & Integration Test Suite for Human-Readable AI Trading Decision Explanations.
==================================================================================

Covers:
1. Schema conformity and dataclass serialization.
2. BUY decision explanation generation with high conviction.
3. HOLD decision explanation generation with model discordance.
4. SELL decision explanation generation with downside protection rationale.
5. Positive news handling: supportive vs pivotal news causal evaluation.
6. Negative news handling: supportive vs contradictory news context.
7. Absent news handling: quiet market vs disconnected feed discrimination.
8. Direct company news vs broad market context discrimination.
9. Missing model outputs and NaN resilience.
10. Historical timestamp adherence (zero lookahead, MOO Open_{t+1} fills).
11. Provider abstraction and graceful fallback on LLM failure.
12. Empirical verification on real historical dataset (AAPL BUY, HOLD, SELL, News).
13. Strict safety assertion: Zero mutations to production database or historical research.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import sys
import unittest
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.analytics.decision_explainer import (
    DecisionExplainer,
    DecisionExplanation,
    GroundedAnalyticalDecisionExplainer,
    BaseDecisionExplanationProvider,
    SUPPORTED_TICKERS,
    COMPANY_NAMES,
    FUSION_WEIGHTS,
    BUY_THRESHOLD,
    SELL_THRESHOLD,
    LIVE_DB_PATH
)

PROD_DB_PATH = PROJECT_ROOT / "data" / "live" / "news_database.db"
PROCESSED_FUSION_PATH = PROJECT_ROOT / "data" / "processed" / "multi_company_decision_fusion.csv"


def calculate_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file for safety / non-mutation assertions."""
    if not path.exists():
        return ""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class TestDecisionExplainer(unittest.TestCase):
    """Comprehensive test suite for Decision Explainer engine and provider integration."""

    @classmethod
    def setUpClass(cls):
        # Capture baseline SHA-256 hashes of critical production artifacts
        cls.prod_db_hash_before = calculate_file_hash(PROD_DB_PATH)
        cls.fusion_csv_hash_before = calculate_file_hash(PROCESSED_FUSION_PATH)

    @classmethod
    def tearDownClass(cls):
        # Safety assertion: verify zero modification to production assets
        if cls.prod_db_hash_before:
            prod_db_hash_after = calculate_file_hash(PROD_DB_PATH)
            assert cls.prod_db_hash_before == prod_db_hash_after, (
                "FATAL: Production database was modified during testing!"
            )
        if cls.fusion_csv_hash_before:
            fusion_csv_hash_after = calculate_file_hash(PROCESSED_FUSION_PATH)
            assert cls.fusion_csv_hash_before == fusion_csv_hash_after, (
                "FATAL: Processed decision fusion dataset was modified during testing!"
            )

    def setUp(self):
        self.explainer = DecisionExplainer()
        self.analytical_engine = GroundedAnalyticalDecisionExplainer()

    # --------------------------------------------------------------------------
    # 1. Schema Conformity and Serialization
    # --------------------------------------------------------------------------
    def test_schema_conformity_and_serialization(self):
        """Verify that DecisionExplanation adheres to required schema and serializes cleanly."""
        mock_row = {
            "Date": "2024-01-03",
            "Ticker": "AAPL",
            "Classification_P_UP": 0.58,
            "Classification_Signal": 0.16,
            "Classification_Contribution": 0.056,
            "RL_Action": 2,
            "RL_Signal": 1.0,
            "RL_Contribution": 0.300,
            "Market_Regime": 0,
            "Market_Regime_Label": "Bull Trend",
            "Market_Regime_Signal": 0.5,
            "Market_Regime_Contribution": 0.075,
            "News_Count": 0,
            "NLP_Signal": 0.0,
            "NLP_Contribution": 0.0,
            "Dominant_Sentiment": "Neutral",
            "Fusion_Score": 0.431,
            "Final_Decision": "BUY"
        }

        explanation = self.explainer.explain_decision(mock_row)

        self.assertIsInstance(explanation, DecisionExplanation)
        self.assertEqual(explanation.ticker, "AAPL")
        self.assertEqual(explanation.decision, "BUY")
        self.assertGreater(explanation.fusion_score, 0.20)

        # Check serialization
        d = explanation.to_dict()
        self.assertIsInstance(d, dict)
        required_keys = [
            "ticker", "company_name", "date", "decision", "fusion_score",
            "decision_summary", "primary_reasons", "supporting_evidence",
            "contradicting_evidence", "news_impact", "model_agreement",
            "what_could_change", "uncertainties", "data_freshness",
            "explanation_confidence", "limitations", "fusion_breakdown",
            "provider_used"
        ]
        for k in required_keys:
            self.assertIn(k, d, f"Missing required key in explanation: {k}")

        # JSON encodability
        json_str = json.dumps(d)
        self.assertTrue(len(json_str) > 100)

    # --------------------------------------------------------------------------
    # 2. BUY Decision Explanation
    # --------------------------------------------------------------------------
    def test_buy_decision_explanation(self):
        """Verify plain-English explanation for high-conviction BUY signal."""
        buy_row = {
            "Date": "2024-01-03",
            "Ticker": "AAPL",
            "Classification_P_UP": 0.62,
            "Classification_Signal": 0.24,
            "Classification_Contribution": 0.084,
            "RL_Action": 2,
            "RL_Signal": 1.0,
            "RL_Contribution": 0.300,
            "Market_Regime": 0,
            "Market_Regime_Label": "Bull Trend",
            "Market_Regime_Signal": 0.5,
            "Market_Regime_Contribution": 0.075,
            "News_Count": 0,
            "NLP_Signal": 0.0,
            "NLP_Contribution": 0.0,
            "Dominant_Sentiment": "Neutral",
            "Fusion_Score": 0.459,
            "Final_Decision": "BUY"
        }
        market_feat = {
            "Close": 185.0,
            "SMA_5": 182.0,
            "Volatility_5": 0.18,
            "Daily_Return": 0.012
        }

        exp = self.explainer.explain_decision(buy_row, market_features=market_feat)

        self.assertEqual(exp.decision, "BUY")
        self.assertIn("BUY", exp.decision_summary)
        self.assertTrue(any("classification model indicates upward probability" in r for r in exp.primary_reasons))
        self.assertTrue(any("reinforcement-learning agent favors buying" in r for r in exp.primary_reasons))
        self.assertTrue(any("short-term price momentum is positive" in r.lower() for r in exp.primary_reasons))
        self.assertEqual(exp.model_agreement["reinforcement_learning_stance"], "Supports BUY (Action BUY)")
        self.assertIn(exp.explanation_confidence, ["HIGH", "MODERATE"])

    # --------------------------------------------------------------------------
    # 3. HOLD Decision Explanation with Conflicting Models
    # --------------------------------------------------------------------------
    def test_hold_decision_with_conflicting_models(self):
        """Verify that model discordance (Class BUY vs RL SELL) triggers explicit conflict explanation."""
        hold_row = {
            "Date": "2024-02-15",
            "Ticker": "MSFT",
            "Classification_P_UP": 0.59,
            "Classification_Signal": 0.18,
            "Classification_Contribution": 0.063,
            "RL_Action": 0,  # SELL
            "RL_Signal": -1.0,
            "RL_Contribution": -0.300,
            "Market_Regime": 1,
            "Market_Regime_Label": "Neutral / Sideways",
            "Market_Regime_Signal": 0.0,
            "Market_Regime_Contribution": 0.0,
            "News_Count": 0,
            "NLP_Signal": 0.0,
            "NLP_Contribution": 0.0,
            "Dominant_Sentiment": "Neutral",
            "Fusion_Score": -0.237,  # Will be adjusted or test actual HOLD score
            "Final_Decision": "HOLD"
        }
        # Set a balanced score near zero
        hold_row["Classification_Contribution"] = 0.15
        hold_row["RL_Contribution"] = -0.15
        hold_row["Fusion_Score"] = 0.00
        hold_row["Final_Decision"] = "HOLD"

        exp = self.explainer.explain_decision(hold_row)

        self.assertEqual(exp.decision, "HOLD")
        self.assertIn("HOLD", exp.decision_summary)
        # Should detect conflict
        self.assertTrue(exp.model_agreement["has_internal_conflict"])
        self.assertIn("do not agree", exp.model_agreement["consensus_statement"].lower())
        self.assertTrue(len(exp.contradicting_evidence) > 0)

    # --------------------------------------------------------------------------
    # 4. SELL Decision Explanation with Downside Protection
    # --------------------------------------------------------------------------
    def test_sell_decision_explanation(self):
        """Verify plain-English explanation for SELL signal focusing on risk mitigation."""
        sell_row = {
            "Date": "2023-10-20",
            "Ticker": "AAPL",
            "Classification_P_UP": 0.41,
            "Classification_Signal": -0.18,
            "Classification_Contribution": -0.063,
            "RL_Action": 0,
            "RL_Signal": -1.0,
            "RL_Contribution": -0.300,
            "Market_Regime": 2,
            "Market_Regime_Label": "High Volatility Drawdown",
            "Market_Regime_Signal": -0.5,
            "Market_Regime_Contribution": -0.075,
            "News_Count": 0,
            "NLP_Signal": 0.0,
            "NLP_Contribution": 0.0,
            "Dominant_Sentiment": "Neutral",
            "Fusion_Score": -0.438,
            "Final_Decision": "SELL"
        }
        market_feat = {
            "Close": 172.0,
            "SMA_5": 176.0,
            "Volatility_5": 0.38,
            "Daily_Return": -0.024
        }

        exp = self.explainer.explain_decision(sell_row, market_features=market_feat)

        self.assertEqual(exp.decision, "SELL")
        self.assertIn("SELL", exp.decision_summary)
        self.assertTrue(any("downward probability" in r.lower() for r in exp.primary_reasons))
        self.assertTrue(any("reinforcement-learning agent favors selling" in r.lower() for r in exp.primary_reasons))
        self.assertTrue(any("unfavorable" in r.lower() or "drawdown" in r.lower() for r in exp.primary_reasons))
        self.assertIn(exp.explanation_confidence, ["HIGH", "MODERATE"])

    # --------------------------------------------------------------------------
    # 5. Positive News Handling: Supportive vs Pivotal
    # --------------------------------------------------------------------------
    def test_positive_news_supportive_vs_pivotal(self):
        """Verify differentiation between supportive news and pivotal news that flipped the decision."""
        # Case A: Supportive news (Technical was already BUY)
        row_supportive = {
            "Date": "2024-03-01",
            "Ticker": "NVDA",
            "Classification_P_UP": 0.65,
            "Classification_Signal": 0.30,
            "Classification_Contribution": 0.105,
            "RL_Action": 2,
            "RL_Signal": 1.0,
            "RL_Contribution": 0.300,
            "Market_Regime": 0,
            "Market_Regime_Label": "Bull Trend",
            "Market_Regime_Signal": 0.5,
            "Market_Regime_Contribution": 0.075,
            "News_Count": 1,
            "NLP_Signal": 0.8,
            "NLP_Contribution": 0.160,
            "Dominant_Sentiment": "Positive",
            "Fusion_Score": 0.640,
            "Final_Decision": "BUY"
        }
        news_item = [{
            "Headline": "NVIDIA Reports Record Quarterly Revenue Driven by Accelerating Data Center Demand",
            "Summary": "NVIDIA announced blowout results with surging demand for its AI accelerators.",
            "Sentiment_Score": 0.8,
            "Sentiment_Label": "Positive",
            "Date": "2024-03-01",
            "Ticker": "NVDA"
        }]

        exp_sup = self.explainer.explain_decision(row_supportive, matching_news=news_item)
        self.assertTrue(exp_sup.news_impact["has_news"])
        self.assertFalse(exp_sup.news_impact["pivotal_influence"])
        self.assertEqual(exp_sup.news_impact["decision_without_news"], "BUY")
        self.assertIn("Supportive context", exp_sup.news_impact["influence_on_decision"])

        # Case B: Pivotal news (Technical was HOLD, News flipped to BUY)
        row_pivotal = {
            "Date": "2024-03-05",
            "Ticker": "NVDA",
            "Classification_P_UP": 0.53,
            "Classification_Signal": 0.06,
            "Classification_Contribution": 0.021,  # 0.35 * 0.06
            "RL_Action": 1,  # HOLD
            "RL_Signal": 0.0,
            "RL_Contribution": 0.000,
            "Market_Regime": 1,
            "Market_Regime_Label": "Neutral",
            "Market_Regime_Signal": 0.0,
            "Market_Regime_Contribution": 0.000,
            "News_Count": 1,
            "NLP_Signal": 0.95,
            "NLP_Contribution": 0.190,  # 0.20 * 0.95
            "Dominant_Sentiment": "Positive",
            "Fusion_Score": 0.211,  # Crosses 0.20!
            "Final_Decision": "BUY"
        }
        exp_piv = self.explainer.explain_decision(row_pivotal, matching_news=news_item)
        self.assertTrue(exp_piv.news_impact["has_news"])
        self.assertTrue(exp_piv.news_impact["pivotal_influence"])
        self.assertEqual(exp_piv.news_impact["decision_without_news"], "HOLD")
        self.assertIn("Pivotal factor", exp_piv.news_impact["influence_on_decision"])

    # --------------------------------------------------------------------------
    # 6. Negative News Handling: Contradictory vs Supportive
    # --------------------------------------------------------------------------
    def test_negative_news_contradictory_context(self):
        """Verify that negative news in a BUY signal is correctly marked as contradictory context."""
        row = {
            "Date": "2024-04-10",
            "Ticker": "AMZN",
            "Classification_P_UP": 0.70,
            "Classification_Signal": 0.40,
            "Classification_Contribution": 0.140,
            "RL_Action": 2,
            "RL_Signal": 1.0,
            "RL_Contribution": 0.300,
            "Market_Regime": 0,
            "Market_Regime_Label": "Bull Trend",
            "Market_Regime_Signal": 0.5,
            "Market_Regime_Contribution": 0.075,
            "News_Count": 1,
            "NLP_Signal": -0.60,
            "NLP_Contribution": -0.120,
            "Dominant_Sentiment": "Negative",
            "Fusion_Score": 0.395,
            "Final_Decision": "BUY"
        }
        news_item = [{
            "Headline": "Amazon Faces Regulatory Scrutiny Over Cloud Market Practices in Europe",
            "Summary": "EU regulators have initiated an inquiry into cloud computing market concentration.",
            "Sentiment_Score": -0.60,
            "Sentiment_Label": "Negative",
            "Date": "2024-04-10",
            "Ticker": "AMZN"
        }]

        exp = self.explainer.explain_decision(row, matching_news=news_item)
        self.assertEqual(exp.decision, "BUY")
        self.assertTrue(exp.news_impact["has_news"])
        self.assertIn("Contradicting context", exp.news_impact["influence_on_decision"])
        self.assertTrue(any("news sentiment is negative" in c.lower() for c in exp.contradicting_evidence))

    # --------------------------------------------------------------------------
    # 7. Absent News Handling: Quiet Market vs Feed Disconnected
    # --------------------------------------------------------------------------
    def test_absent_news_quiet_market_vs_feed_disconnected(self):
        """Verify that zero news explicitly differentiates quiet market from disconnected feed."""
        row = {
            "Date": "2024-05-01",
            "Ticker": "GOOGL",
            "Classification_P_UP": 0.52,
            "Classification_Signal": 0.04,
            "Classification_Contribution": 0.014,
            "RL_Action": 1,
            "RL_Signal": 0.0,
            "RL_Contribution": 0.0,
            "Market_Regime": 1,
            "Market_Regime_Label": "Neutral",
            "Market_Regime_Signal": 0.0,
            "Market_Regime_Contribution": 0.0,
            "News_Count": 0,
            "NLP_Signal": 0.0,
            "NLP_Contribution": 0.0,
            "Dominant_Sentiment": "Neutral",
            "Fusion_Score": 0.014,
            "Final_Decision": "HOLD"
        }

        # Case A: Stream connected (Quiet market)
        status_conn = {"connected": True, "articles_received": 12}
        exp_quiet = self.explainer.explain_decision(row, stream_status=status_conn)
        self.assertFalse(exp_quiet.news_impact["has_news"])
        self.assertEqual(exp_quiet.news_impact["feed_status"], "QUIET_MARKET")
        self.assertIn("No relevant news was available", exp_quiet.news_impact["explanation_text"])

        # Case B: Stream disconnected (Feed outage)
        status_disconn = {"connected": False, "articles_received": 0}
        exp_down = self.explainer.explain_decision(row, stream_status=status_disconn)
        self.assertFalse(exp_down.news_impact["has_news"])
        self.assertEqual(exp_down.news_impact["feed_status"], "FEED_DISCONNECTED")
        self.assertIn("offline/disconnected", exp_down.news_impact["explanation_text"])

    # --------------------------------------------------------------------------
    # 8. Direct Company Coverage vs Broad Market Context
    # --------------------------------------------------------------------------
    def test_direct_company_coverage_discrimination(self):
        """Verify discrimination between company-specific news and broad market news."""
        row = {
            "Date": "2024-05-10",
            "Ticker": "TSLA",
            "Classification_P_UP": 0.55,
            "Classification_Signal": 0.10,
            "Classification_Contribution": 0.035,
            "RL_Action": 1,
            "RL_Signal": 0.0,
            "RL_Contribution": 0.0,
            "Market_Regime": 1,
            "Market_Regime_Label": "Neutral",
            "Market_Regime_Signal": 0.0,
            "Market_Regime_Contribution": 0.0,
            "News_Count": 1,
            "NLP_Signal": 0.30,
            "NLP_Contribution": 0.060,
            "Dominant_Sentiment": "Positive",
            "Fusion_Score": 0.095,
            "Final_Decision": "HOLD"
        }

        # Direct news: mentions TSLA / Tesla
        direct_news = [{
            "Headline": "Tesla Delivers First Batch of Cybertrucks in North American Markets",
            "Summary": "Tesla commenced customer handovers for its newly launched vehicle model.",
            "Sentiment_Score": 0.30,
            "Sentiment_Label": "Positive"
        }]
        exp_direct = self.explainer.explain_decision(row, matching_news=direct_news)
        self.assertTrue(exp_direct.news_impact["is_company_specific"])
        self.assertEqual(exp_direct.news_impact["relationship_type"], "Direct Company Coverage")

        # Macro news: general economic headline
        macro_news = [{
            "Headline": "Federal Reserve Holds Benchmark Interest Rates Steady Amid Inflation Data",
            "Summary": "The central bank opted to maintain its target funds rate during the policy meeting.",
            "Sentiment_Score": 0.30,
            "Sentiment_Label": "Positive"
        }]
        exp_macro = self.explainer.explain_decision(row, matching_news=macro_news)
        self.assertFalse(exp_macro.news_impact["is_company_specific"])
        self.assertEqual(exp_macro.news_impact["relationship_type"], "Industry / Macro Context")

    # --------------------------------------------------------------------------
    # 9. Missing Model Outputs and NaN Resilience
    # --------------------------------------------------------------------------
    def test_missing_model_outputs_and_nan_resilience(self):
        """Verify robust handling of NaN values or missing fields without throwing unhandled exceptions."""
        sparse_row = {
            "Date": "2024-06-01",
            "Ticker": "JPM",
            "Final_Decision": "BUY",
            "Fusion_Score": 0.25,
            "Classification_P_UP": np.nan,  # NaN
            "RL_Action": None,             # None
            "News_Count": 0
        }

        exp = self.explainer.explain_decision(sparse_row, market_features=None)
        self.assertIsInstance(exp, DecisionExplanation)
        self.assertEqual(exp.decision, "BUY")
        self.assertIsNotNone(exp.decision_summary)
        self.assertTrue(len(exp.primary_reasons) >= 1)
        self.assertEqual(exp.explanation_confidence, "LOW")

    # --------------------------------------------------------------------------
    # 10. Historical Timestamp Adherence (Zero Lookahead)
    # --------------------------------------------------------------------------
    def test_historical_timestamp_adherence(self):
        """Verify execution timing transparency: MOO fill on Open_{t+1} documented clearly."""
        row = {
            "Date": "2024-01-10",
            "Ticker": "V",
            "Final_Decision": "BUY",
            "Fusion_Score": 0.31,
            "Classification_P_UP": 0.60,
            "Classification_Signal": 0.20,
            "Classification_Contribution": 0.070,
            "RL_Action": 2,
            "RL_Signal": 1.0,
            "RL_Contribution": 0.300,
            "Market_Regime": 0,
            "Market_Regime_Label": "Bull Trend",
            "Market_Regime_Signal": 0.5,
            "Market_Regime_Contribution": 0.075,
            "News_Count": 0
        }

        exp = self.explainer.explain_decision(row)
        timing_str = exp.data_freshness.get("execution_timing", "")
        self.assertIn("Next-Day Market-on-Open", timing_str)
        self.assertIn("Open_{t+1}", timing_str)

    # --------------------------------------------------------------------------
    # 11. Graceful Offline Fallback on LLM Failure
    # --------------------------------------------------------------------------
    def test_graceful_offline_fallback(self):
        """Verify that DecisionExplainer transparently falls back to analytical engine if primary fails."""
        mock_failing_provider = MagicMock(spec=BaseDecisionExplanationProvider)
        mock_failing_provider.is_available.return_value = True
        mock_failing_provider.explain_decision.side_effect = RuntimeError("LLM API quota exceeded or timeout")

        resilient_explainer = DecisionExplainer(provider=mock_failing_provider)

        row = {
            "Date": "2024-01-03",
            "Ticker": "AAPL",
            "Final_Decision": "BUY",
            "Fusion_Score": 0.3442,
            "Classification_P_UP": 0.58,
            "Classification_Signal": 0.16,
            "Classification_Contribution": 0.056,
            "RL_Action": 2,
            "RL_Signal": 1.0,
            "RL_Contribution": 0.300,
            "Market_Regime": 0,
            "Market_Regime_Label": "Bull Trend",
            "Market_Regime_Signal": 0.5,
            "Market_Regime_Contribution": 0.075,
            "News_Count": 0
        }

        # Should NOT raise RuntimeError; must fall back to GroundedAnalyticalEngine
        exp = resilient_explainer.explain_decision(row)
        self.assertIsInstance(exp, DecisionExplanation)
        self.assertEqual(exp.decision, "BUY")
        self.assertEqual(exp.provider_used, "GroundedAnalyticalEngine")

    # --------------------------------------------------------------------------
    # 12. Empirical Verification on Real Historical Dataset
    # --------------------------------------------------------------------------
    def test_empirical_real_historical_dataset(self):
        """Test explanation generation on real historical dataset rows."""
        if not PROCESSED_FUSION_PATH.exists():
            self.skipTest("Historical decision fusion dataset not found.")

        # Test real BUY: AAPL 2024-01-03
        exp_buy = self.explainer.explain_ticker_date("AAPL", "2024-01-03")
        self.assertEqual(exp_buy.ticker, "AAPL")
        self.assertEqual(exp_buy.decision, "BUY")
        self.assertGreater(exp_buy.fusion_score, 0.20)
        self.assertTrue(len(exp_buy.primary_reasons) >= 2)

        # Test real HOLD: AAPL 2023-10-23
        exp_hold = self.explainer.explain_ticker_date("AAPL", "2023-10-23")
        self.assertEqual(exp_hold.ticker, "AAPL")
        self.assertEqual(exp_hold.decision, "HOLD")
        self.assertTrue(abs(exp_hold.fusion_score) <= 0.20)

        # Test real SELL: AAPL 2023-10-20
        exp_sell = self.explainer.explain_ticker_date("AAPL", "2023-10-20")
        self.assertEqual(exp_sell.ticker, "AAPL")
        self.assertEqual(exp_sell.decision, "SELL")
        self.assertLess(exp_sell.fusion_score, -0.20)

        # Test real News Day: AAPL 2023-11-02
        exp_news = self.explainer.explain_ticker_date("AAPL", "2023-11-02")
        self.assertEqual(exp_news.ticker, "AAPL")
        self.assertTrue(exp_news.news_impact["has_news"])
        self.assertIsNotNone(exp_news.news_impact["headline"])
        self.assertIn("Apple", exp_news.news_impact["headline"])


if __name__ == "__main__":
    unittest.main()
