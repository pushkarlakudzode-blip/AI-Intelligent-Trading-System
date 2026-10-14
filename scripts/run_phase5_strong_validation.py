"""
CLI Runner for Phase 5: Strong Validation, Ablation & Statistical Robustness.

Executes:
1. Strict time-ordered evaluation across 4 expanding chronological folds.
2. Nested temporal validation (Train -> Inner Val -> Outer Evaluation).
3. Feature ablation study (Full vs No-Discordance, No-Confidence, No-Disagreement, Discordance-only).
4. Permutation test (N = 1,000 permutations, seed = 42) for conditional incremental value.
5. Leave-One-Company-Out (LOCO) cross-validation across all 10 companies.
6. Sector robustness analysis (GICS 5-sector and Broad 4-sector groups).
7. Placebo control test comparing real Explanation Discordance against randomized placebo.
8. Multicollinearity & redundancy diagnostics (Pearson, Spearman, VIF).
9. Effect size estimation (Cohen's d, median, 95% CIs, % folds/companies improving).
10. Multiple-comparison control via Benjamini-Hochberg False Discovery Rate (FDR).
11. Final Evidence Classification (Category A, B, C, or D).
12. Publication-quality figure generation (6 PNG figures).
13. Generation of comprehensive 18-section research report.
"""

import sys
import logging
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.phase5_strong_validation import run_phase5_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("run_phase5")


def main():
    print("=" * 80)
    print("STARTING PHASE 5: STRONG VALIDATION, ABLATION & STATISTICAL ROBUSTNESS")
    print("ISOLATED RESEARCH ONLY — DO NOT MODIFY PRODUCTION")
    print("=" * 80)

    result = run_phase5_pipeline()

    summary_df = result["summary_df"]
    assigned_category = summary_df[summary_df["Metric_Name"] == "Assigned_Category"]["Value"].iloc[0]

    print("=" * 80)
    print("PHASE 5 PIPELINE SUCCESSFULLY COMPLETED")
    print(f"Final Evidence Classification: {assigned_category}")
    print(f"Report Generated: {result['report_path']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
