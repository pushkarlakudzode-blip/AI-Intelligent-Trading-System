"""
Phase 5B: Portfolio Allocation, Cooldown & Turnover Diagnosis Runner.
===================================================================
Executes all Phase 5B allocation experiments, audits root causes of cash drag
and whipsaw turnover, verifies out-of-sample generalization across splits,
and exports structured artifacts to data/evaluation/phase5b/.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.evaluation.phase5b_allocation_engine import Phase5BAllocationEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_phase5b")

OUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "phase5b"


def main():
    logger.info("Initializing Phase 5B Allocation Engine...")
    engine = Phase5BAllocationEngine()

    logger.info("Running baseline behavior diagnosis and experimental simulations...")
    res = engine.run_all_experiments(cost_bps=10.0, output_dir=OUT_DIR)
    diag = res["diagnosis"]

    print("\n" + "=" * 105)
    print("PHASE 5B: PORTFOLIO ALLOCATION, COOLDOWN & TURNOVER DIAGNOSIS REPORT")
    print("=" * 105)

    # 1. Section 1: Turnover & Cash-Trap Diagnosis
    print("\n" + "=" * 105)
    print("DIAGNOSTIC SECTION 1: ROOT-CAUSE AUDIT OF CASH DRAG & TURNOVER")
    print("=" * 105)
    print(f"Total Trading Sessions Evaluated:         {diag['total_trading_sessions']} sessions")
    print(f"Mean Actively Held Stocks:                {diag['mean_actively_held_stocks']:.2f} / 10 equities")
    print(f"Median Actively Held Stocks:              {diag['median_actively_held_stocks']} equities")
    print(f"Min / Max Held Stocks:                    {diag['min_actively_held_stocks']} / {diag['max_actively_held_stocks']} equities")
    print(f"Total Portfolio Round-Trips:              {diag['total_round_trips']} trades")
    print(f"Total Whipsaws (Duration <= 3 Days):      {diag['total_whipsaws_le_3d']} trades ({diag['portfolio_whipsaw_percentage']}%)")
    print("\nActive Stocks Distribution (Days per stock count):")
    for cnt, num_days in sorted(diag["active_stocks_distribution"].items()):
        pct = (num_days / diag["total_trading_sessions"]) * 100
        bar = "#" * int(pct // 2)
        print(f"  {cnt} stocks held: {num_days:3d} days ({pct:5.1f}%) | {bar}")

    print("\nPer-Stock Round-Trip & Whipsaw Breakdown:")
    df_per_stock = pd.DataFrame(diag["per_stock_diagnosis"])
    cols_display = [
        "ticker", "company_name", "buy_days", "sell_days", "hold_days",
        "round_trips", "avg_holding_duration_days", "median_holding_duration_days",
        "whipsaws_le_3d", "whipsaw_ratio"
    ]
    print(df_per_stock[cols_display].to_string(index=False))

    # 2. Section 2: Temporal Split Comparison (Validation vs Test)
    print("\n" + "=" * 105)
    print("DIAGNOSTIC SECTION 2: OUT-OF-SAMPLE GENERALIZATION (VALIDATION 150d vs UNTOUCHED TEST 150d)")
    print("=" * 105)
    df_splits = pd.read_csv(OUT_DIR / "split_comparison.csv")
    
    split_summary = []
    for exp_name in df_splits["experiment_name"].unique():
        sub = df_splits[df_splits["experiment_name"] == exp_name].set_index("split")
        val_r = sub.loc["VALIDATION", "cumulative_return"] * 100
        val_sh = sub.loc["VALIDATION", "sharpe_ratio"]
        val_to = sub.loc["VALIDATION", "total_turnover"] * 100
        val_cash = sub.loc["VALIDATION", "mean_cash"] * 100

        tst_r = sub.loc["UNTOUCHED_TEST", "cumulative_return"] * 100
        tst_sh = sub.loc["UNTOUCHED_TEST", "sharpe_ratio"]
        tst_to = sub.loc["UNTOUCHED_TEST", "total_turnover"] * 100
        tst_cash = sub.loc["UNTOUCHED_TEST", "mean_cash"] * 100

        full_r = sub.loc["FULL_300D", "cumulative_return"] * 100
        full_sh = sub.loc["FULL_300D", "sharpe_ratio"]
        full_to = sub.loc["FULL_300D", "total_turnover"] * 100
        full_cash = sub.loc["FULL_300D", "mean_cash"] * 100

        split_summary.append({
            "Experiment": exp_name.replace("Exp_", ""),
            "Val_Ret": f"{val_r:+6.2f}%",
            "Val_Sh": f"{val_sh:4.2f}",
            "Val_TO": f"{val_to:5.0f}%",
            "Test_Ret": f"{tst_r:+6.2f}%",
            "Test_Sh": f"{tst_sh:4.2f}",
            "Test_TO": f"{tst_to:5.0f}%",
            "Full_Ret": f"{full_r:+6.2f}%",
            "Full_Sh": f"{full_sh:4.2f}",
            "Full_TO": f"{full_to:5.0f}%",
            "Full_Cash": f"{full_cash:4.1f}%"
        })
    df_split_comp = pd.DataFrame(split_summary)
    print(df_split_comp.to_string(index=False))

    # 3. Section 3: Standardized Full 300-Day Experiment Table
    print("\n" + "=" * 105)
    print("DIAGNOSTIC SECTION 3: STANDARDIZED FULL 300-DAY EXPERIMENT COMPARISON (10 bps Friction)")
    print("=" * 105)
    df_full = pd.read_csv(OUT_DIR / "experiment_summary.csv")
    full_table = []
    for _, r in df_full.iterrows():
        full_table.append({
            "Experiment": r["experiment_name"].replace("Exp_", ""),
            "Cum_Ret": f"{r['cumulative_return']*100:+6.2f}%",
            "Ann_Ret": f"{r['annualized_return']*100:5.2f}%",
            "Vol": f"{r['annualized_volatility']*100:4.1f}%",
            "Sharpe": f"{r['sharpe_ratio']:4.2f}",
            "Sortino": f"{r['sortino_ratio']:4.2f}",
            "MaxDD": f"{r['max_drawdown']*100:5.2f}%",
            "Cash": f"{r['mean_cash']*100:4.1f}%",
            "Turnover": f"{r['total_turnover']*100:5.0f}%",
            "Trades": int(r["trade_count"]),
            "Whipsaws": int(r["whipsaws_3d"]),
            "Beta": f"{r['beta']:4.2f}" if pd.notnull(r["beta"]) else "1.00",
            "CAPM_Alpha": f"{r['capm_alpha']*100:+5.2f}%" if pd.notnull(r["capm_alpha"]) else "+0.00%"
        })
    print(pd.DataFrame(full_table).to_string(index=False))

    # 4. Section 4: Friction Sensitivity Table
    print("\n" + "=" * 105)
    print("DIAGNOSTIC SECTION 4: TRANSACTION FRICTION SENSITIVITY (0 to 50 bps)")
    print("=" * 105)
    df_fric = pd.read_csv(OUT_DIR / "cost_sensitivity_comparison.csv")
    piv_ret = df_fric.pivot(index="experiment_name", columns="cost_bps", values="cumulative_return")
    piv_sh = df_fric.pivot(index="experiment_name", columns="cost_bps", values="sharpe_ratio")
    
    fric_display = []
    for exp in piv_ret.index:
        fric_display.append({
            "Experiment": exp.replace("Exp_", ""),
            "0bps_Ret": f"{piv_ret.loc[exp, 0.0]*100:+6.2f}%",
            "0bps_Sh": f"{piv_sh.loc[exp, 0.0]:4.2f}",
            "10bps_Ret": f"{piv_ret.loc[exp, 10.0]*100:+6.2f}%",
            "10bps_Sh": f"{piv_sh.loc[exp, 10.0]:4.2f}",
            "20bps_Ret": f"{piv_ret.loc[exp, 20.0]*100:+6.2f}%",
            "20bps_Sh": f"{piv_sh.loc[exp, 20.0]:4.2f}",
            "50bps_Ret": f"{piv_ret.loc[exp, 50.0]*100:+6.2f}%",
            "50bps_Sh": f"{piv_sh.loc[exp, 50.0]:4.2f}"
        })
    print(pd.DataFrame(fric_display).to_string(index=False))

    print("\n" + "=" * 105)
    logger.info(f"Phase 5B runner completed successfully. Artifacts saved in {OUT_DIR}")


if __name__ == "__main__":
    main()
