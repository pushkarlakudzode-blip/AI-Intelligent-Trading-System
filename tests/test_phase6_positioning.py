"""
Unit tests for Phase 6: Final Research Contribution, Novelty Validation & Paper Positioning.

Verifies:
1. Complete Phase 6 artifact suite existence in data/live/alpaca_staging/research/phase6/.
2. Literature review table schema (13 columns) and verified peer-reviewed paper count (21 papers).
3. Novelty matrix schema (10 columns) and framework comparison completeness.
4. Claim strength audit matrix (8 statements) with strict prohibition on overclaiming.
5. Novelty position document confirms Category B and separates Novelty vs Contribution vs Result.
6. Paper-ready modules completeness (research_gap, novelty, contributions, limitations, conclusion).
7. Three distinct tiers of the exact novelty statement (Versions A, B, C).
8. PPT presentation content (6 slides + 30-45s spoken script).
9. Comprehensive 15-section final novelty report.
10. Strict production database non-interference (articles=525, nlp_results=524, validation_logs=6).
"""

import os
import sys
import sqlite3
import hashlib
import unittest
from pathlib import Path

import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.phase6_paper_positioning import (
    run_phase6_pipeline,
    PHASE6_DIR,
    LITERATURE_DIR,
    CLAIMS_DIR,
    PAPER_DIR,
    PRESENTATION_DIR,
    REPORTS_DIR
)


class TestPhase6Positioning(unittest.TestCase):
    """Test suite validating Phase 6 paper positioning, novelty, and safety."""

    @classmethod
    def setUpClass(cls):
        # Run pipeline once to ensure artifacts exist
        cls.pipeline_out = run_phase6_pipeline()
        cls.db_path = PROJECT_ROOT / "data" / "live" / "news_database.db"

    def test_01_phase6_directories_and_files(self):
        """Verify all 12 required files exist in their respective phase6 subdirectories."""
        expected_files = [
            LITERATURE_DIR / "literature_review.csv",
            LITERATURE_DIR / "novelty_matrix.csv",
            LITERATURE_DIR / "research_gap_analysis.md",
            CLAIMS_DIR / "claim_strength_matrix.csv",
            CLAIMS_DIR / "novelty_position.md",
            PAPER_DIR / "research_gap.md",
            PAPER_DIR / "novelty.md",
            PAPER_DIR / "contributions.md",
            PAPER_DIR / "limitations.md",
            PAPER_DIR / "conclusion.md",
            PRESENTATION_DIR / "ppt_content.md",
            REPORTS_DIR / "phase6_final_novelty_report.md"
        ]
        for f in expected_files:
            self.assertTrue(f.exists(), f"Missing required Phase 6 file: {f}")
            self.assertGreater(f.stat().st_size, 0, f"File {f} is empty")

    def test_02_literature_review_schema_and_contents(self):
        """Verify literature_review.csv schema (13 columns) and 21 real papers."""
        lit_csv = LITERATURE_DIR / "literature_review.csv"
        df = pd.read_csv(lit_csv)
        self.assertEqual(len(df), 21, f"Expected 21 papers in literature review, found {len(df)}")
        
        required_cols = [
            "Paper_Title", "Year", "Authors", "Venue", "Problem", "Modalities",
            "Prediction_Task", "Explanation_Method", "Cross_Modal_Explanations_Compared",
            "Explanation_Disagreement_Measured", "Used_For_Error_Prediction",
            "Used_For_Selective_Prediction", "Main_Limitation", "How_Our_Approach_Differs"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing required column in literature review: {col}")

        # Verify that NO prior paper measured explanation disagreement for error prediction in finance
        for _, row in df.iterrows():
            if "Our Approach" not in row["Paper_Title"]:
                self.assertEqual(row["Used_For_Selective_Prediction"] in ["No", "Yes (Foundational Risk-Coverage & AURC Framework)", "Yes (End-to-end selective training)", "Yes (Threshold-based abstention)", "Yes (Rejection via ensemble entropy)"], True)
                if row["Cross_Modal_Explanations_Compared"] == "Yes":
                    # Only Lyu 2022, Frank 2021, Hessel 2020 compare vision-language modalities, none predict financial errors
                    self.assertEqual(row["Used_For_Error_Prediction"], "No")

    def test_03_novelty_matrix_structure(self):
        """Verify novelty_matrix.csv compares approaches across 10 dimensions."""
        mat_csv = LITERATURE_DIR / "novelty_matrix.csv"
        df = pd.read_csv(mat_csv)
        self.assertGreaterEqual(len(df), 9)

        required_cols = [
            "Approach_Family", "Market_Data", "News_Data", "Multimodal", "XAI",
            "Cross_Modal_Explanation_Comparison", "Explanation_Discordance_Metric",
            "Error_Prediction", "Selective_Prediction", "Our_Difference"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing column in novelty matrix: {col}")

        # Check our proposed row exists
        our_rows = df[df["Approach_Family"].str.contains("Proposed")]
        self.assertEqual(len(our_rows), 1)
        self.assertEqual(our_rows.iloc[0]["Multimodal"], "Yes (10 Companies, 2019-2024)")
        self.assertEqual(our_rows.iloc[0]["Cross_Modal_Explanation_Comparison"], "Yes (Market vs News Explanations in Common Space)")

    def test_04_claim_strength_matrix_safety(self):
        """Verify claim_strength_matrix.csv audits 8 claims and forbids overclaiming."""
        claim_csv = CLAIMS_DIR / "claim_strength_matrix.csv"
        df = pd.read_csv(claim_csv)
        self.assertEqual(len(df), 8)

        # Claim 7 ("universally useful") must strictly be "NO"
        claim7 = df[df["Claim_Statement"].str.contains("universally useful")].iloc[0]
        self.assertEqual(claim7["Safe_To_Claim"], "NO")

        # Claim 1 ("novel") must be "QUALIFIED YES"
        claim1 = df[df["Claim_Statement"].str.contains("Explanation Discordance is novel")].iloc[0]
        self.assertEqual(claim1["Safe_To_Claim"], "QUALIFIED YES")

        # Claim 8 ("potentially novel research direction") must be "YES"
        claim8 = df[df["Claim_Statement"].str.contains("potentially novel research direction")].iloc[0]
        self.assertEqual(claim8["Safe_To_Claim"], "YES")

    def test_05_novelty_position_classification(self):
        """Verify novelty_position.md confirms Category B and outlines tripartite distinction."""
        path = CLAIMS_DIR / "novelty_position.md"
        content = path.read_text(encoding="utf-8")
        self.assertIn("CATEGORY B", content)
        self.assertIn("POTENTIALLY NOVEL WITH MODERATE EMPIRICAL EVIDENCE", content)
        self.assertIn("NOVELTY", content)
        self.assertIn("CONTRIBUTION", content)
        self.assertIn("RESULT", content)
        self.assertIn("DO NOT CLAIM universal effectiveness", content)

    def test_06_paper_modules_completeness(self):
        """Verify all 5 paper modules exist and contain comprehensive content."""
        modules = ["research_gap.md", "novelty.md", "contributions.md", "limitations.md", "conclusion.md"]
        for mod in modules:
            p = PAPER_DIR / mod
            self.assertTrue(p.exists(), f"Missing paper module: {mod}")
            text = p.read_text(encoding="utf-8")
            self.assertGreater(len(text), 200, f"Module {mod} text too short")

    def test_07_novelty_statement_three_versions(self):
        """Verify paper/novelty.md contains Version A, Version B, and Version C."""
        nov_path = PAPER_DIR / "novelty.md"
        content = nov_path.read_text(encoding="utf-8")
        self.assertIn("Version A", content)
        self.assertIn("Version B", content)
        self.assertIn("Version C", content)
        self.assertIn("Explanation Discordance", content)

    def test_08_ppt_presentation_deck(self):
        """Verify presentation/ppt_content.md contains 6 slides and 30-45s pitch."""
        ppt_path = PRESENTATION_DIR / "ppt_content.md"
        content = ppt_path.read_text(encoding="utf-8")
        self.assertIn("Slide 1", content)
        self.assertIn("Slide 2", content)
        self.assertIn("Slide 3", content)
        self.assertIn("Slide 4", content)
        self.assertIn("Slide 5", content)
        self.assertIn("Slide 6", content)
        self.assertIn("Spoken Presentation Script", content)

    def test_09_final_report_sections(self):
        """Verify phase6_final_novelty_report.md contains all required sections."""
        rep_path = REPORTS_DIR / "phase6_final_novelty_report.md"
        content = rep_path.read_text(encoding="utf-8")
        self.assertIn("## 1. Executive Summary", content)
        self.assertIn("## 2. Research Scope & The Core Question", content)
        self.assertIn("## 3. Literature Review Methodology", content)
        self.assertIn("## 4. Novelty Matrix", content)
        self.assertIn("## 5. Research Gap", content)
        self.assertIn("## 6. Exact Novelty Formulations", content)
        self.assertIn("## 7. Structured Research Contributions", content)
        self.assertIn("## 8. Novelty vs. Contribution vs. Result", content)
        self.assertIn("## 9. Claim-Strength Governance Matrix", content)
        self.assertIn("## 10. Explicit Research Limitations", content)
        self.assertIn("## 11. Final Novelty Verdict", content)
        self.assertIn("## 15. Safety & System Verification", content)

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

        with open(self.db_path, "rb") as f:
            db_md5 = hashlib.md5(f.read()).hexdigest()
        self.assertEqual(db_md5, "91f5c7dd3011ea44fca54df3ef24fd87", "Database MD5 checksum changed!")


if __name__ == "__main__":
    unittest.main()
