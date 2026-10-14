"""
CLI Runner for Phase 4: Complementary Signal & Error-Prediction Analysis.

Runs:
1. 7 error-prediction logistic models (Models A to G).
2. Paired bootstrap significance testing (1,000 resamples).
3. Non-linear effect analysis (quadratic, deciles, tertiles).
4. Interaction effect analysis (Confidence x Discordance, Disagreement x Discordance).
5. Temporal robustness (walk-forward expanding windows).
6. Company-level robustness across all 10 companies.
7. Exploratory composite learned error-risk selective prediction on 2024 test data.
8. Publication figure generation under results/figures/.
9. Generation of phase4_complementary_signal_report.md.
"""

import sys
import logging
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.phase4_complementary_analysis import run_phase4_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("run_phase4")


def main():
    print("=" * 80)
    print("STARTING PHASE 4: COMPLEMENTARY SIGNAL & ERROR-PREDICTION ANALYSIS")
    print("STAGING ONLY — DO NOT MODIFY PRODUCTION")
    print("=" * 80)
    
    result = run_phase4_pipeline()
    
    print("=" * 80)
    print("PHASE 4 PIPELINE SUCCESSFULLY COMPLETED")
    print(f"Evaluated Test Observations: {result['n_test']:,}")
    print(f"Baseline Error Model (Conf + Dis) ROC-AUC: {result['baseline_auc']:.4f}")
    print(f"Combined Error Model (+ Discordance) ROC-AUC: {result['combined_auc']:.4f}")
    print(f"Bootstrap Mean Delta ROC-AUC: {result['delta_auc_boot']:+.5f} (p = {result['p_val_boot']:.4f})")
    print("=" * 80)


if __name__ == "__main__":
    main()
