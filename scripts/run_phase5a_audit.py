"""
Phase 5A: Independent Backtest Methodology Audit and Execution Realism Runner.
Audits:
1. Information availability & news timestamp lookahead.
2. Execution realism: Next-Day Open (MOO) vs Same-Day Close (MOC) vs Intraday Open-to-Close.
3. Accounting models: Strict Dollar Ledger vs Weight Compounding.
4. Exposure disambiguation: Multi-asset portfolio weight contribution vs standalone single-stock days invested.
5. Standardized CAPM Jensen's alpha.
6. Exports all audit tables and structured findings to data/evaluation/phase5a/.
"""

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.out_of_sample_evaluator import (
    OutOfSampleEvaluator,
    PerformanceMetrics,
    bootstrap_metric_ci,
    COMPANY_NAMES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase5a_audit")

AUDIT_DIR = PROJECT_ROOT / "data" / "evaluation" / "phase5a"


def audit_news_timestamps(processed_dir: Path) -> Dict[str, Any]:
    """Audit news timestamps against market hours to identify lookahead risk."""
    news_nlp_path = processed_dir / "multi_company_news_nlp_results.csv"
    if not news_nlp_path.exists():
        return {}
    df = pd.read_csv(news_nlp_path)
    df["ts"] = pd.to_datetime(df["Timestamp"])
    
    t_open = pd.to_datetime("09:30:00").time()
    t_close = pd.to_datetime("16:00:00").time()
    
    before_open = df[df["ts"].dt.time < t_open]
    during_market = df[(df["ts"].dt.time >= t_open) & (df["ts"].dt.time < t_close)]
    after_close = df[df["ts"].dt.time >= t_close]
    
    return {
        "total_news_items": len(df),
        "before_market_open_count": len(before_open),
        "before_market_open_pct": round(len(before_open) / len(df) * 100, 2),
        "during_market_count": len(during_market),
        "during_market_pct": round(len(during_market) / len(df) * 100, 2),
        "after_market_close_count": len(after_close),
        "after_market_close_pct": round(len(after_close) / len(df) * 100, 2),
        "lookahead_risk_if_same_day_close": f"{len(after_close)} articles ({len(after_close)/len(df)*100:.1f}%)"
    }


def main():
    logger.info("=" * 85)
    logger.info("PHASE 5A: INDEPENDENT BACKTEST METHODOLOGY AUDIT & EXECUTION REALISM")
    logger.info("=" * 85)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    evaluator = OutOfSampleEvaluator()
    
    # 1. Audit News Timestamps
    ts_audit = audit_news_timestamps(PROJECT_ROOT / "data" / "processed")
    with open(AUDIT_DIR / "news_timestamp_audit.json", "w", encoding="utf-8") as f:
        json.dump(ts_audit, f, indent=2)
        
    print("\n" + "=" * 95)
    print("AUDIT SECTION 1: INFORMATION AVAILABILITY & NEWS TIMESTAMP DISTRIBUTION")
    print("=" * 95)
    print(f"Total News Items Evaluated:             {ts_audit['total_news_items']}")
    print(f"Before Market Open (< 09:30 EST):       {ts_audit['before_market_open_count']} ({ts_audit['before_market_open_pct']}%)")
    print(f"During Market Hours (09:30 - 16:00 EST): {ts_audit['during_market_count']} ({ts_audit['during_market_pct']}%)")
    print(f"After Market Close (>= 16:00 EST):      {ts_audit['after_market_close_count']} ({ts_audit['after_market_close_pct']}%)")
    print(f"-> Lookahead Risk in Phase 4 MOC Fill:  {ts_audit['lookahead_risk_if_same_day_close']}")

    # 2. Execution Realism Comparison across Fill Models
    print("\n" + "=" * 95)
    print("AUDIT SECTION 2: EXECUTION REALISM ACROSS FILL MODELS (10-ASSET PORTFOLIO)")
    print("=" * 95)
    
    exec_modes = [
        ("NEXT_OPEN", "Model B: Next-Day Open-to-Open (MOO Fill, Zero Lookahead)"),
        ("SAME_CLOSE", "Model A: Same-Day Close-to-Close (MOC Fill, Upper Bound)"),
        ("INTRADAY_OPEN_CLOSE", "Model C: Next-Day Open-to-Close (Intraday Fill)")
    ]
    
    exec_results = []
    for mode_code, mode_label in exec_modes:
        # Full Fusion
        met_full, _ = evaluator.evaluate_multi_asset_portfolio(
            strategy_col="Final_Decision", strategy_name="Decision_Fusion_Full",
            execution_mode=mode_code, accounting_model="DOLLAR_LEDGER", cost_bps=10.0
        )
        # No News
        met_nonews, _ = evaluator.evaluate_multi_asset_portfolio(
            strategy_col="Decision_NoNews", strategy_name="Decision_Fusion_NoNews",
            execution_mode=mode_code, accounting_model="DOLLAR_LEDGER", cost_bps=10.0
        )
        # Buy & Hold
        met_bh, _ = evaluator.evaluate_multi_asset_portfolio(
            strategy_col=None, strategy_name="Buy_and_Hold",
            execution_mode=mode_code, accounting_model="DOLLAR_LEDGER", cost_bps=10.0
        )
        # RL Only
        met_rl, _ = evaluator.evaluate_multi_asset_portfolio(
            strategy_col="Decision_RL", strategy_name="RL_Only",
            execution_mode=mode_code, accounting_model="DOLLAR_LEDGER", cost_bps=10.0
        )
        
        exec_results.append({
            "Execution_Model": mode_label,
            "Fusion_CumRet": f"{met_full.cumulative_return*100:+.2f}%",
            "Fusion_Sharpe": f"{met_full.sharpe_ratio:.2f}",
            "Fusion_MaxDD": f"{met_full.max_drawdown*100:.2f}%",
            "NoNews_CumRet": f"{met_nonews.cumulative_return*100:+.2f}%",
            "NoNews_Sharpe": f"{met_nonews.sharpe_ratio:.2f}",
            "RL_CumRet": f"{met_rl.cumulative_return*100:+.2f}%",
            "RL_Sharpe": f"{met_rl.sharpe_ratio:.2f}",
            "B&H_CumRet": f"{met_bh.cumulative_return*100:+.2f}%",
            "B&H_Sharpe": f"{met_bh.sharpe_ratio:.2f}"
        })
        
    df_exec = pd.DataFrame(exec_results)
    df_exec.to_csv(AUDIT_DIR / "execution_realism_comparison.csv", index=False)
    print(df_exec.to_string(index=False))

    # 3. Accounting Model Comparison (Strict Dollar Ledger vs Weight Compounding)
    print("\n" + "=" * 95)
    print("AUDIT SECTION 3: ACCOUNTING MODEL RECONCILIATION (STRICT DOLLAR LEDGER VS WEIGHT COMPOUNDING)")
    print("=" * 95)
    
    acct_results = []
    for acct_code, acct_label in [("DOLLAR_LEDGER", "Strict Dollar Ledger (Cash Account)"), ("WEIGHT_COMPOUNDING", "Weight Compounding (Phase 4 Legacy)")]:
        met_f, _ = evaluator.evaluate_multi_asset_portfolio(
            strategy_col="Final_Decision", strategy_name="Decision_Fusion_Full",
            execution_mode="NEXT_OPEN", accounting_model=acct_code, cost_bps=10.0
        )
        met_b, _ = evaluator.evaluate_multi_asset_portfolio(
            strategy_col=None, strategy_name="Buy_and_Hold",
            execution_mode="NEXT_OPEN", accounting_model=acct_code, cost_bps=10.0
        )
        acct_results.append({
            "Accounting_Model": acct_label,
            "Fusion_CumRet": f"{met_f.cumulative_return*100:+.2f}%",
            "Fusion_Sharpe": f"{met_f.sharpe_ratio:.2f}",
            "Fusion_MDD": f"{met_f.max_drawdown*100:.2f}%",
            "Fusion_Turnover": f"{met_f.total_turnover*100:.0f}%",
            "B&H_CumRet": f"{met_b.cumulative_return*100:+.2f}%",
            "B&H_Sharpe": f"{met_b.sharpe_ratio:.2f}"
        })
    df_acct = pd.DataFrame(acct_results)
    df_acct.to_csv(AUDIT_DIR / "accounting_model_comparison.csv", index=False)
    print(df_acct.to_string(index=False))

    # 4. Exposure & Target Weight Disambiguation Matrix
    print("\n" + "=" * 95)
    print("AUDIT SECTION 4: EXPOSURE DISAMBIGUATION (PORTFOLIO ALLOCATION CONTRIBUTION VS SINGLE-STOCK DAYS)")
    print("=" * 95)
    
    # Calculate per-stock portfolio weight contribution under strict dollar ledger
    dates = sorted(evaluator.df_aligned["Date"].unique())
    tickers = evaluator.tickers
    cost_rate = 0.0010
    
    cash = 100000.0
    stock_dollars = {t: 0.0 for t in tickers}
    stock_weights_hist = {t: [] for t in tickers}
    
    for d in dates:
        day_df = evaluator.df_aligned[evaluator.df_aligned["Date"] == d].set_index("Ticker")
        total_eq_start = cash + sum(stock_dollars.values())
        for t in tickers:
            dec = day_df.loc[t, "Final_Decision"] if t in day_df.index else "HOLD"
            curr_val = stock_dollars[t]
            if dec == "BUY":
                target_val = 0.10 * total_eq_start
                delta = target_val - curr_val
                if delta > 0:
                    buy_amt = min(delta, cash)
                    cash -= (buy_amt + buy_amt * cost_rate)
                    stock_dollars[t] += buy_amt
            elif dec == "SELL":
                if curr_val > 0:
                    cash += (curr_val - curr_val * cost_rate)
                    stock_dollars[t] = 0.0
        for t in tickers:
            stock_weights_hist[t].append(stock_dollars[t] / total_eq_start if total_eq_start > 0 else 0.0)
        for t in tickers:
            if stock_dollars[t] > 0 and t in day_df.index:
                r = day_df.loc[t, "Fwd_Open_Open"]
                if pd.notnull(r):
                    stock_dollars[t] *= (1.0 + r)

    exposure_rows = []
    for t in tickers:
        # Single stock standalone evaluation
        met_single, _ = evaluator.evaluate_single_stock(
            ticker=t, strategy_col="Final_Decision", execution_mode="NEXT_OPEN", cost_bps=10.0
        )
        mean_p_weight = np.mean(stock_weights_hist[t])
        peak_p_weight = np.max(stock_weights_hist[t])
        days_inv = met_single.standalone_days_invested or 0.0
        
        exposure_rows.append({
            "Ticker": t,
            "Company_Name": COMPANY_NAMES.get(t, t),
            "Max_Target_Allocation": "10.0%",
            "Mean_Portfolio_Weight": f"{mean_p_weight*100:5.2f}%",
            "Peak_Portfolio_Weight": f"{peak_p_weight*100:5.2f}%",
            "Standalone_Days_Invested": f"{days_inv*100:5.1f}%",
            "Single_Stock_CumRet": f"{met_single.cumulative_return*100:+.2f}%",
            "Single_Stock_Sharpe": f"{met_single.sharpe_ratio:.2f}"
        })
        
    df_exp = pd.DataFrame(exposure_rows)
    df_exp.to_csv(AUDIT_DIR / "exposure_disambiguation.csv", index=False)
    print(df_exp.to_string(index=False))

    # 5. Full Standardized Evaluation Suite under NEXT_OPEN and Strict Dollar Ledger
    logger.info("Executing standardized Phase 5A evaluation suite under NEXT_OPEN fill...")
    suite_results = evaluator.run_full_suite(
        execution_mode="NEXT_OPEN",
        accounting_model="DOLLAR_LEDGER",
        cost_bps=10.0,
        risk_free_rate=0.03,
        output_dir=AUDIT_DIR
    )

    print("\n" + "=" * 95)
    print("AUDIT SECTION 5: STANDARDIZED PHASE 5A MULTI-ASSET PORTFOLIO RESULTS (NEXT_OPEN FILL)")
    print("=" * 95)
    df_port = pd.read_csv(AUDIT_DIR / "portfolio_comparison.csv")
    cols_show = [
        "strategy_name", "cumulative_return", "annualized_return", "annualized_volatility",
        "sharpe_ratio", "sortino_ratio", "max_drawdown", "mean_exposure", "total_turnover", "beta", "capm_alpha"
    ]
    df_disp = df_port[cols_show].copy()
    df_disp["cumulative_return"] = df_disp["cumulative_return"].map(lambda x: f"{x*100:+.2f}%")
    df_disp["annualized_return"] = df_disp["annualized_return"].map(lambda x: f"{x*100:+.2f}%")
    df_disp["annualized_volatility"] = df_disp["annualized_volatility"].map(lambda x: f"{x*100:.2f}%")
    df_disp["sharpe_ratio"] = df_disp["sharpe_ratio"].map(lambda x: f"{x:.2f}")
    df_disp["sortino_ratio"] = df_disp["sortino_ratio"].map(lambda x: f"{x:.2f}")
    df_disp["max_drawdown"] = df_disp["max_drawdown"].map(lambda x: f"{x*100:.2f}%")
    df_disp["mean_exposure"] = df_disp["mean_exposure"].map(lambda x: f"{x*100:.1f}%")
    df_disp["total_turnover"] = df_disp["total_turnover"].map(lambda x: f"{x*100:.0f}%")
    df_disp["beta"] = df_disp["beta"].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "1.00")
    df_disp["capm_alpha"] = df_disp["capm_alpha"].map(lambda x: f"{x*100:+.2f}%" if pd.notnull(x) else "0.00%")
    print(df_disp.to_string(index=False))

    logger.info(f"Phase 5A audit completed successfully. Artifacts saved in {AUDIT_DIR}")


if __name__ == "__main__":
    main()
