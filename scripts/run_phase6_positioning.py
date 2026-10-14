"""
CLI Runner for Phase 6: Final Research Contribution, Novelty Validation & Paper Positioning.

Generates the complete Phase 6 artifact suite:
1. phase6/literature/literature_review.csv (21 peer-reviewed papers compared along 13 dimensions)
2. phase6/literature/novelty_matrix.csv (Framework comparison matrix)
3. phase6/literature/research_gap_analysis.md (Formal academic research gap analysis)
4. phase6/claims/claim_strength_matrix.csv (Audit of 8 core claims)
5. phase6/claims/novelty_position.md (Novelty vs Contribution vs Result analysis & prohibited claims)
6. phase6/paper/research_gap.md (Paper-ready research gap)
7. phase6/paper/novelty.md (Paper-ready 3-tier novelty statements)
8. phase6/paper/contributions.md (Paper-ready 5 structured contributions)
9. phase6/paper/limitations.md (Paper-ready transparent limitations)
10. phase6/paper/conclusion.md (Paper-ready conclusion and future directions)
11. phase6/presentation/ppt_content.md (6 presentation slides + 30-45s spoken script)
12. phase6/reports/phase6_final_novelty_report.md (Comprehensive 15-section academic report)
"""

import sys
import logging
from pathlib import Path

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.phase6_paper_positioning import run_phase6_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("run_phase6")


def main():
    print("=" * 80)
    print("STARTING PHASE 6: FINAL RESEARCH CONTRIBUTION, NOVELTY & PAPER POSITIONING")
    print("ISOLATED RESEARCH ONLY — DO NOT MODIFY PRODUCTION")
    print("=" * 80)

    result = run_phase6_pipeline()

    print("=" * 80)
    print("PHASE 6 ARTIFACT GENERATION COMPLETED SUCCESSFULLY")
    print(f"Literature Review CSV:   {result['literature_review_csv']}")
    print(f"Novelty Matrix CSV:       {result['novelty_matrix_csv']}")
    print(f"Research Gap MD:          {result['research_gap_analysis_md']}")
    print(f"Claim Strength CSV:       {result['claim_strength_matrix_csv']}")
    print(f"Novelty Position MD:      {result['novelty_position_md']}")
    print(f"Presentation MD:          {result['ppt_content_md']}")
    print(f"Final Report MD:          {result['final_report_md']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
