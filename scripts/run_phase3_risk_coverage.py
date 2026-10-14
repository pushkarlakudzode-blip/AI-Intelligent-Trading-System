"""
CLI Runner for Phase 3: Risk-Coverage & Selective Prediction Evaluation.

Runs:
1. Risk-Coverage curve generation for Confidence, Disagreement, Discordance, Oracle, Random.
2. AURC computation and baseline improvement reporting.
3. Paired bootstrap resampling (1,000 resamples) for statistical hypothesis tests.
4. Robustness checks across 10 individual companies and calendar years (2023 vs 2024).
5. Decile error analysis and multivariate logistic regression.
6. Publication figure generation under results/figures/.
7. Generation of phase3_risk_coverage_report.md.
"""

import sys
import logging
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.phase3_risk_coverage import run_phase3_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("run_phase3")


def main():
    print("=" * 80)
    print("STARTING PHASE 3: RISK-COVERAGE & SELECTIVE PREDICTION EVALUATION")
    print("STAGING ONLY — DO NOT MODIFY PRODUCTION")
    print("=" * 80)
    
    result = run_phase3_pipeline()
    
    print("=" * 80)
    print("PHASE 3 PIPELINE SUCCESSFULLY COMPLETED")
    print(f"Evaluated Test Observations: {result['n_test']:,}")
    print(f"Confidence AURC:   {result['aurc_confidence']:.4f}")
    print(f"Disagreement AURC: {result['aurc_disagreement']:.4f}")
    print(f"Random AURC:       {result['aurc_random']:.4f}")
    print(f"Discordance AURC:  {result['aurc_discordance']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
