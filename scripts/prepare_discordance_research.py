"""
CLI Runner for Explanation Discordance Research Data Preparation.

Executes the alignment of 2019–2024 market data with Alpaca staging news signals,
performs temporal leakage audits, generates coverage reports, prepares research
datasets, and configures baseline experiment definitions in:
data/live/alpaca_staging/research/
"""

import sys
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research.prepare_discordance_research import run_research_preparation_pipeline


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    result = run_research_preparation_pipeline()

    if result.get("status") == "SUCCESS":
        print("\nResearch data preparation completed successfully!")
        print(f"Report: {result['report_path']}")
        sys.exit(0)
    else:
        print("\nResearch data preparation failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
