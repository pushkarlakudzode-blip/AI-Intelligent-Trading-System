"""
Runner script for Phase 4: Rigorous Out-of-Sample Evaluation.
Executes multi-asset and per-stock backtesting, ablation tests, bootstrap intervals,
and writes structured evaluation artifacts to data/evaluation/phase4/.
"""

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.evaluation.out_of_sample_evaluator import OutOfSampleEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_phase4")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "data" / "evaluation" / "phase4"


def main():
    logger.info("=" * 80)
    logger.info("PHASE 4: RIGOROUS OUT-OF-SAMPLE EVALUATION ENGINE")
    logger.info("=" * 80)

    evaluator = OutOfSampleEvaluator()
    results = evaluator.run_full_suite(cost_bps=10.0, output_dir=EVAL_DIR)

    # Load generated artifacts for summary display
    df_portfolio = pd.read_csv(EVAL_DIR / "portfolio_comparison.csv")
    df_stocks = pd.read_csv(EVAL_DIR / "per_stock_comparison.csv")
    with open(EVAL_DIR / "news_ablation_significance.json", "r", encoding="utf-8") as f:
        ablation = json.load(f)
    df_cost = pd.read_csv(EVAL_DIR / "cost_sensitivity.csv")

    print("\n" + "=" * 95)
    print("TABLE 1: 10-ASSET MULTI-ASSET PORTFOLIO OUT-OF-SAMPLE PERFORMANCE (10 BPS FRICTION)")
    print("=" * 95)
    cols_show = [
        "strategy_name", "cumulative_return", "annualized_return", "annualized_volatility",
        "sharpe_ratio", "sortino_ratio", "max_drawdown", "mean_exposure", "total_turnover", "beta", "alpha_ann"
    ]
    df_p_disp = df_portfolio[cols_show].copy()
    df_p_disp["cumulative_return"] = df_p_disp["cumulative_return"].map(lambda x: f"{x*100:+.2f}%")
    df_p_disp["annualized_return"] = df_p_disp["annualized_return"].map(lambda x: f"{x*100:+.2f}%")
    df_p_disp["annualized_volatility"] = df_p_disp["annualized_volatility"].map(lambda x: f"{x*100:.2f}%")
    df_p_disp["sharpe_ratio"] = df_p_disp["sharpe_ratio"].map(lambda x: f"{x:.2f}")
    df_p_disp["sortino_ratio"] = df_p_disp["sortino_ratio"].map(lambda x: f"{x:.2f}")
    df_p_disp["max_drawdown"] = df_p_disp["max_drawdown"].map(lambda x: f"{x*100:.2f}%")
    df_p_disp["mean_exposure"] = df_p_disp["mean_exposure"].map(lambda x: f"{x*100:.1f}%")
    df_p_disp["total_turnover"] = df_p_disp["total_turnover"].map(lambda x: f"{x*100:.0f}%")
    df_p_disp["beta"] = df_p_disp["beta"].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "1.00")
    df_p_disp["alpha_ann"] = df_p_disp["alpha_ann"].map(lambda x: f"{x*100:+.2f}%" if pd.notnull(x) else "0.00%")
    print(df_p_disp.to_string(index=False))

    print("\n" + "=" * 95)
    print("TABLE 2: NEWS ABLATION HYPOTHESIS TEST (FULL FUSION VS. NO-NEWS FUSION)")
    print("=" * 95)
    print(f"Mean Daily Return Difference (Full - NoNews):   {ablation['mean_daily_return_difference']:+.6f} ({ablation['annualized_return_difference']*100:+.2f}% ann)")
    print(f"Paired t-test t-statistic:                      {ablation['paired_ttest_t_stat']:.4f}")
    print(f"Paired t-test p-value:                          {ablation['paired_ttest_p_value']:.4f} (Significant: {ablation['statistically_significant_at_05']})")
    print(f"Wilcoxon signed-rank test statistic:            {ablation['wilcoxon_stat']:.1f}")
    print(f"Wilcoxon signed-rank p-value:                   {ablation['wilcoxon_p_value']:.4f}")

    print("\n" + "=" * 95)
    print("TABLE 3: TRANSACTION COST & SLIPPAGE SENSITIVITY (FULL FUSION PORTFOLIO)")
    print("=" * 95)
    df_c_disp = df_cost.copy()
    df_c_disp["cumulative_return"] = df_c_disp["cumulative_return"].map(lambda x: f"{x*100:+.2f}%")
    df_c_disp["annualized_return"] = df_c_disp["annualized_return"].map(lambda x: f"{x*100:+.2f}%")
    df_c_disp["annualized_volatility"] = df_c_disp["annualized_volatility"].map(lambda x: f"{x*100:.2f}%")
    df_c_disp["sharpe_ratio"] = df_c_disp["sharpe_ratio"].map(lambda x: f"{x:.2f}")
    df_c_disp["max_drawdown"] = df_c_disp["max_drawdown"].map(lambda x: f"{x*100:.2f}%")
    df_c_disp["total_turnover"] = df_c_disp["total_turnover"].map(lambda x: f"{x*100:.0f}%")
    print(df_c_disp.to_string(index=False))

    print("\n" + "=" * 95)
    print("TABLE 4: PER-EQUITY PERFORMANCE (DECISION FUSION FULL VS. BUY & HOLD)")
    print("=" * 95)
    fus_stocks = df_stocks[df_stocks["strategy_name"] == "Decision_Fusion_Full"].set_index("ticker")
    bh_stocks = df_stocks[df_stocks["strategy_name"] == "Buy_and_Hold"].set_index("ticker")
    
    comp_rows = []
    for t in fus_stocks.index:
        f_row = fus_stocks.loc[t]
        b_row = bh_stocks.loc[t]
        comp_rows.append({
            "Ticker": t,
            "Fusion_CumRet": f"{f_row['cumulative_return']*100:+.2f}%",
            "Fusion_Sharpe": f"{f_row['sharpe_ratio']:.2f}",
            "Fusion_MDD": f"{f_row['max_drawdown']*100:.2f}%",
            "Fusion_Exp": f"{f_row['mean_exposure']*100:.1f}%",
            "B&H_CumRet": f"{b_row['cumulative_return']*100:+.2f}%",
            "B&H_Sharpe": f"{b_row['sharpe_ratio']:.2f}",
            "B&H_MDD": f"{b_row['max_drawdown']*100:.2f}%",
            "Diff_Return": f"{(f_row['cumulative_return'] - b_row['cumulative_return'])*100:+.2f}%"
        })
    print(pd.DataFrame(comp_rows).to_string(index=False))
    logger.info("Evaluation runner finished successfully.")


if __name__ == "__main__":
    main()
