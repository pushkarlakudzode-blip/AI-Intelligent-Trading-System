"""
CLI Entrypoint for Phase 2 Explanation Discordance Research Experiment.

Runs:
1. Model training for Group A (Market), Group B (News), Group C (Multimodal).
2. Held-out test set evaluation & metric reporting.
3. TreeSHAP attribution extraction for Multimodal model.
4. Common-space representation & Explanation Discordance calculation.
5. Descriptive analysis, baseline comparisons, and selective prediction preparation.
6. Verification that production SQLite database remains untouched.
"""

import sys
import logging
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.phase2_discordance_experiment import run_phase2_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("run_phase2")


def main():
    print("=" * 80)
    print("STARTING PHASE 2: TRAINING, SHAP ATTRIBUTION & EXPLANATION DISCORDANCE")
    print("STAGING ONLY — DO NOT MODIFY PRODUCTION")
    print("=" * 80)
    
    result = run_phase2_pipeline()
    
    print("=" * 80)
    print("PHASE 2 PIPELINE SUCCESSFULLY COMPLETED")
    print(f"Evaluated Test Observations: {result['n_test']:,}")
    print(f"Mean Explanation Discordance: {result['mean_discordance']:.4f}")
    print("Metrics:")
    for m in result["metrics"]:
        print(f"  {m['Model']}: Accuracy={m['Accuracy']}, ROC-AUC={m['ROC_AUC']}, F1={m['F1']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
