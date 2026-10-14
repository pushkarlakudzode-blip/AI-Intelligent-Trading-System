"""
Phase 5D: Controlled Strategy Comparison and Candidate Selection Engine.
=======================================================================
Implements institutional, controlled comparison between:
- Candidate A: Original Decision Fusion Strategy (Raw score thresholds: Buy >= +0.20, Sell <= -0.20)
- Candidate B: Hysteresis Exit Threshold of -0.30 (Buy >= +0.20, Sell <= -0.30, Hold in between)
- Candidate C: Hysteresis Exit Threshold of -0.30 PLUS 3-Trading-Day Cooldown (Anti-Whipsaw)
- Benchmark: Passive Equal-Weight Buy & Hold (10% per asset, zero active turnover)

Architectural Mandates:
1. Identical chronological data splits (Validation 150d vs Untouched Test 150d vs Full 300d).
2. Strict next-day Market-On-Open (MOO) execution timing (Open_{t+1} to Open_{t+2}, zero lookahead).
3. Candidate selection restricted strictly to the Validation split; out-of-sample verification on Untouched Test.
4. Comprehensive tracking of returns, Sharpe, Sortino, Max Drawdown, turnover, trades, exposure, cash drag, whipsaws.
5. Per-stock standalone and within-portfolio performance across all 10 portfolio equities.
6. Execution cost sensitivity (0, 5, 10, 20, 30 bps) and limit-order non-fill stress testing.
7. Isolated news incremental ablation (paired t-test and Wilcoxon signed-rank test).
8. Read-only execution with all results saved to data/evaluation/phase5d/.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.out_of_sample_evaluator import (
    COMPANY_NAMES,
    SUPPORTED_TICKERS,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio
)

logger = logging.getLogger("phase5d_selector")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PHASE5D_DIR = PROJECT_ROOT / "data" / "evaluation" / "phase5d"


@dataclass
class CandidateConfig:
    name: str
    description: str
    max_asset_weight: float = 0.10
    hysteresis_sell_threshold: Optional[float] = None
    cooldown_days: int = 0
    buy_threshold: float = 0.20
    default_sell_threshold: float = -0.20
    cost_bps: float = 10.0
    execution_mode: str = "NEXT_OPEN"
    is_benchmark: bool = False
    use_news: bool = True
    non_fill_rate: float = 0.0
    gap_limit_pct: Optional[float] = None
    seed: int = 42


@dataclass
class CandidateMetrics:
    candidate_name: str
    split: str
    total_sessions: int
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    mean_exposure: float
    mean_cash: float
    total_turnover: float
    trade_count: int
    whipsaws_3d: int
    win_rate: float
    cost_bps: float
    non_fill_rate: float
    beta: Optional[float] = None
    capm_alpha: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Phase5DCandidateSelector:
    """
    Controlled evaluation, sensitivity testing, and candidate selection engine.
    """

    def __init__(self, processed_dir: Optional[Path] = None):
        self.processed_dir = processed_dir or PROCESSED_DIR
        self.tickers = sorted(SUPPORTED_TICKERS)
        self.company_names = COMPANY_NAMES
        self.df_aligned: Optional[pd.DataFrame] = None
        self.dates: List[pd.Timestamp] = []
        self.val_dates: List[pd.Timestamp] = []
        self.test_dates: List[pd.Timestamp] = []
        self._load_and_align_datasets()

    def _load_and_align_datasets(self) -> None:
        """Load decision fusion and market features, enforcing next-day MOO return alignment."""
        fusion_path = self.processed_dir / "multi_company_decision_fusion.csv"
        feat_path = self.processed_dir / "multi_company_features.csv"

        if not fusion_path.exists():
            raise FileNotFoundError(f"Missing fusion dataset: {fusion_path}")
        if not feat_path.exists():
            raise FileNotFoundError(f"Missing feature dataset: {feat_path}")

        df_fus = pd.read_csv(fusion_path)
        df_fus["Date"] = pd.to_datetime(df_fus["Date"])

        df_feat = pd.read_csv(feat_path)
        df_feat["Date"] = pd.to_datetime(df_feat["Date"])
        df_feat = df_feat.sort_values(["Ticker", "Date"]).reset_index(drop=True)

        # Forward open-to-open returns (MOO fill on Open_{t+1}, return to Open_{t+2})
        df_feat["Open_t1"] = df_feat.groupby("Ticker")["Open"].shift(-1)
        df_feat["Open_t2"] = df_feat.groupby("Ticker")["Open"].shift(-2)
        df_feat["Fwd_Open_Open"] = (df_feat["Open_t2"] - df_feat["Open_t1"]) / df_feat["Open_t1"]

        # Calculate overnight opening gap: |Open_{t+1} - Close_t| / Close_t
        df_feat["Open_Gap_Pct"] = (df_feat["Open_t1"] - df_feat["Close"]).abs() / df_feat["Close"]

        # Merge datasets
        df = pd.merge(
            df_fus,
            df_feat[["Date", "Ticker", "Fwd_Open_Open", "Open_Gap_Pct", "Open", "Close", "Open_t1", "Open_t2"]],
            on=["Date", "Ticker"],
            how="inner"
        )

        # Compute Ablated Fusion Score (without news component):
        # Normalized weights: Class = 0.35/0.80 = 0.4375, RL = 0.30/0.80 = 0.3750, Regime = 0.15/0.80 = 0.1875
        df["Score_NoNews"] = (
            0.4375 * df["Classification_Signal"] +
            0.3750 * df["RL_Signal"] +
            0.1875 * df["Market_Regime_Signal"]
        ).round(4)
        df["Decision_NoNews"] = np.where(
            df["Score_NoNews"] >= 0.20, "BUY",
            np.where(df["Score_NoNews"] <= -0.20, "SELL", "HOLD")
        )

        df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
        self.df_aligned = df

        self.dates = sorted(df["Date"].unique())
        n_dates = len(self.dates)
        split_idx = n_dates // 2
        self.val_dates = self.dates[:split_idx]
        self.test_dates = self.dates[split_idx:]

        logger.info(
            f"Phase 5D Selector ready: {len(df):,} observations across {n_dates} dates "
            f"(Validation: {len(self.val_dates)}d, Untouched Test: {len(self.test_dates)}d)."
        )

    def simulate_candidate(
        self,
        config: CandidateConfig,
        target_dates: List[pd.Timestamp],
        initial_capital: float = 100000.0
    ) -> Tuple[CandidateMetrics, pd.DataFrame, Dict[str, Any]]:
        """
        Simulate trading under Strict Dollar Ledger accounting, realistic MOO fills,
        anti-whipsaw rules, transaction costs, and optional limit-order non-fill constraints.
        """
        cost_rate = config.cost_bps / 10000.0
        cash = initial_capital
        stock_dollars = {t: 0.0 for t in self.tickers}
        hold_duration = {t: 0 for t in self.tickers}
        trades = 0
        turnover_dollars_total = 0.0
        whipsaws_3d = 0
        history = []

        # Per-stock tracking
        per_stock_stats = {
            t: {
                "ticker": t,
                "company_name": self.company_names.get(t, t),
                "buys": 0,
                "sells": 0,
                "holds": 0,
                "turnover_dollars": 0.0,
                "unfilled_orders": 0,
                "whipsaws_3d": 0,
                "pnl_dollars": 0.0
            }
            for t in self.tickers
        }

        # Deterministic RNG for stochastic non-fill simulation
        rng = np.random.RandomState(config.seed)
        sub_df = self.df_aligned

        for d in target_dates:
            day_df = sub_df[sub_df["Date"] == d].set_index("Ticker")
            total_eq_start = cash + sum(stock_dollars.values())
            day_turnover = 0.0

            # 1. Determine Desired Action
            desired_actions = {}
            for t in self.tickers:
                if config.is_benchmark:
                    desired_actions[t] = "BUY"
                    continue

                if t not in day_df.index:
                    desired_actions[t] = "HOLD"
                    continue

                row = day_df.loc[t]
                score = row["Score_NoNews"] if not config.use_news else row["Fusion_Score"]

                # Apply Hysteresis
                if config.hysteresis_sell_threshold is not None:
                    if score >= config.buy_threshold:
                        dec = "BUY"
                    elif score <= config.hysteresis_sell_threshold:
                        dec = "SELL"
                    else:
                        dec = "HOLD"
                else:
                    if not config.use_news:
                        dec = row["Decision_NoNews"]
                    else:
                        dec = row["Final_Decision"]

                # Apply Cooldown Rule (Overrides SELL to HOLD if held < cooldown_days)
                if dec == "SELL" and stock_dollars[t] > 0 and hold_duration[t] < config.cooldown_days:
                    dec = "HOLD"

                desired_actions[t] = dec

            # 2. Check Limit-Order Fill Viability
            executed_actions = {}
            for t in self.tickers:
                act = desired_actions[t]
                if act == "HOLD" or config.is_benchmark:
                    executed_actions[t] = act
                    continue

                # Check gap limit non-fill
                is_unfilled = False
                if config.gap_limit_pct is not None and t in day_df.index:
                    gap = day_df.loc[t, "Open_Gap_Pct"]
                    if pd.notnull(gap) and gap > config.gap_limit_pct:
                        is_unfilled = True

                # Check stochastic non-fill
                if config.non_fill_rate > 0.0:
                    if rng.uniform(0.0, 1.0) < config.non_fill_rate:
                        is_unfilled = True

                if is_unfilled:
                    per_stock_stats[t]["unfilled_orders"] += 1
                    # Unfilled order keeps previous position
                    executed_actions[t] = "HOLD"
                else:
                    executed_actions[t] = act

            # 3. Determine Target Portfolio Allocation Weights
            active_assets = [
                t for t, a in executed_actions.items()
                if a == "BUY" or (a == "HOLD" and stock_dollars[t] > 0)
            ]

            target_weights = {t: 0.0 for t in self.tickers}
            if config.is_benchmark:
                for t in self.tickers:
                    target_weights[t] = 0.10
            else:
                for t in active_assets:
                    target_weights[t] = min(0.10, config.max_asset_weight)

            # 4. Execute SELLs first to replenish cash
            for t in self.tickers:
                if executed_actions[t] == "SELL":
                    curr_val = stock_dollars[t]
                    if curr_val > 0:
                        fee = curr_val * cost_rate
                        cash += (curr_val - fee)
                        day_turnover += curr_val
                        turnover_dollars_total += curr_val
                        per_stock_stats[t]["sells"] += 1
                        per_stock_stats[t]["turnover_dollars"] += curr_val
                        stock_dollars[t] = 0.0
                        if hold_duration[t] <= 3:
                            whipsaws_3d += 1
                            per_stock_stats[t]["whipsaws_3d"] += 1
                        hold_duration[t] = 0
                        trades += 1
                elif executed_actions[t] == "HOLD":
                    if stock_dollars[t] > 0:
                        hold_duration[t] += 1
                        per_stock_stats[t]["holds"] += 1

            # 5. Execute BUYs / Target Rebalances using available cash
            for t in self.tickers:
                if executed_actions[t] == "BUY":
                    curr_val = stock_dollars[t]
                    target_val = target_weights[t] * total_eq_start
                    delta = target_val - curr_val
                    if delta > 10.0:
                        buy_amt = min(delta, cash)
                        if buy_amt > 10.0:
                            fee = buy_amt * cost_rate
                            cash -= (buy_amt + fee)
                            stock_dollars[t] += buy_amt
                            day_turnover += buy_amt
                            turnover_dollars_total += buy_amt
                            per_stock_stats[t]["buys"] += 1
                            per_stock_stats[t]["turnover_dollars"] += buy_amt
                            if curr_val == 0:
                                hold_duration[t] = 1
                            else:
                                hold_duration[t] += 1
                            trades += 1
                    else:
                        if stock_dollars[t] > 0:
                            hold_duration[t] += 1
                            per_stock_stats[t]["holds"] += 1

            # 6. Holding Return from MOO Fill (Open_{t+1} to Open_{t+2})
            for t in self.tickers:
                if stock_dollars[t] > 0 and t in day_df.index:
                    r = day_df.loc[t, "Fwd_Open_Open"]
                    if pd.notnull(r):
                        pnl = stock_dollars[t] * r
                        per_stock_stats[t]["pnl_dollars"] += pnl
                        stock_dollars[t] *= (1.0 + r)

            total_eq_end = cash + sum(stock_dollars.values())
            day_ret = (total_eq_end / total_eq_start) - 1.0 if total_eq_start > 0 else 0.0

            history.append({
                "Date": d,
                "Equity": total_eq_end,
                "Cash": cash,
                "Net_Daily_Return": day_ret,
                "Portfolio_Exposure": sum(stock_dollars.values()) / total_eq_end if total_eq_end > 0 else 0.0,
                "Turnover": day_turnover / total_eq_start if total_eq_start > 0 else 0.0
            })

        res_df = pd.DataFrame(history)
        n_sessions = len(target_dates)
        cum_ret = float((res_df["Equity"].iloc[-1] / initial_capital) - 1.0)
        ann_ret = float((1.0 + cum_ret) ** (252.0 / n_sessions) - 1.0)
        ann_vol = float(res_df["Net_Daily_Return"].std() * np.sqrt(252.0))
        sharpe = float(calculate_sharpe_ratio(res_df["Net_Daily_Return"], risk_free_rate=0.03))
        sortino = float(calculate_sortino_ratio(res_df["Net_Daily_Return"], risk_free_rate=0.03))
        mdd, _ = calculate_max_drawdown(res_df["Equity"])
        mean_exp = float(res_df["Portfolio_Exposure"].mean())
        mean_cash = float((res_df["Cash"] / res_df["Equity"]).mean())
        tot_turnover = float(turnover_dollars_total / initial_capital)
        win_rate = float((res_df["Net_Daily_Return"] > 0).mean())

        # Determine split label
        if n_sessions == len(self.val_dates) and target_dates[0] == self.val_dates[0]:
            split_label = "VALIDATION"
        elif n_sessions == len(self.test_dates) and target_dates[0] == self.test_dates[0]:
            split_label = "UNTOUCHED_TEST"
        else:
            split_label = "FULL_300D"

        metrics = CandidateMetrics(
            candidate_name=config.name,
            split=split_label,
            total_sessions=n_sessions,
            cumulative_return=cum_ret,
            annualized_return=ann_ret,
            annualized_volatility=ann_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=mdd,
            mean_exposure=mean_exp,
            mean_cash=mean_cash,
            total_turnover=tot_turnover,
            trade_count=trades,
            whipsaws_3d=whipsaws_3d,
            win_rate=win_rate,
            cost_bps=config.cost_bps,
            non_fill_rate=config.non_fill_rate
        )

        return metrics, res_df, per_stock_stats

    def evaluate_single_stock_standalone(
        self,
        ticker: str,
        config: CandidateConfig,
        target_dates: List[pd.Timestamp],
        initial_capital: float = 10000.0
    ) -> Dict[str, Any]:
        """
        Evaluate a single stock standalone under the candidate's exact trading rules.
        """
        sub = self.df_aligned[self.df_aligned["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
        sub = sub[sub["Date"].isin(target_dates)].reset_index(drop=True)
        n_sessions = len(sub)
        cost_rate = config.cost_bps / 10000.0

        pos = 0.0
        equity = initial_capital
        history = []
        trades = 0
        whipsaws_3d = 0
        hold_days = 0

        for i, row in sub.iterrows():
            prev_pos = pos
            if config.is_benchmark:
                act = "BUY"
            else:
                score = row["Score_NoNews"] if not config.use_news else row["Fusion_Score"]
                if config.hysteresis_sell_threshold is not None:
                    if score >= config.buy_threshold:
                        act = "BUY"
                    elif score <= config.hysteresis_sell_threshold:
                        act = "SELL"
                    else:
                        act = "HOLD"
                else:
                    act = row["Final_Decision"]

                if act == "SELL" and pos > 0 and hold_days < config.cooldown_days:
                    act = "HOLD"

            if act == "BUY":
                pos = 1.0
            elif act == "SELL":
                pos = 0.0

            # Friction
            turn = abs(pos - prev_pos)
            fee = equity * turn * cost_rate
            equity -= fee

            if turn > 0:
                trades += 1
                if prev_pos > 0 and pos == 0:
                    if hold_days <= 3:
                        whipsaws_3d += 1
                    hold_days = 0
                elif pos > 0 and prev_pos == 0:
                    hold_days = 1
            else:
                if pos > 0:
                    hold_days += 1

            fwd_r = row["Fwd_Open_Open"] if pd.notnull(row["Fwd_Open_Open"]) else 0.0
            day_ret = pos * fwd_r
            equity *= (1.0 + day_ret)

            history.append({"Date": row["Date"], "Equity": equity, "Return": day_ret})

        df_h = pd.DataFrame(history)
        cum_ret = float((df_h["Equity"].iloc[-1] / initial_capital) - 1.0)
        ann_vol = float(df_h["Return"].std() * np.sqrt(252.0))
        sharpe = float(calculate_sharpe_ratio(df_h["Return"], risk_free_rate=0.03))
        mdd, _ = calculate_max_drawdown(df_h["Equity"])

        return {
            "ticker": ticker,
            "company_name": self.company_names.get(ticker, ticker),
            "candidate_name": config.name,
            "cumulative_return": cum_ret,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": mdd,
            "trades": trades,
            "whipsaws_3d": whipsaws_3d
        }

    def run_full_phase5d_comparison(self, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Execute complete Phase 5D controlled comparison, validation selection,
        untouched test verification, cost/non-fill sensitivities, and news ablation.
        """
        out_dir = output_dir or PHASE5D_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        candidates = [
            CandidateConfig(
                name="Candidate_A_Baseline_Fusion",
                description="Original Decision Fusion strategy (Raw score: Buy >= 0.20, Sell <= -0.20, 10% cap).",
                max_asset_weight=0.10,
                cost_bps=10.0
            ),
            CandidateConfig(
                name="Candidate_B_Hysteresis_Exit_30",
                description="Hysteresis exit threshold at -0.30 (Buy >= 0.20, Sell <= -0.30, 10% cap).",
                max_asset_weight=0.10,
                hysteresis_sell_threshold=-0.30,
                cost_bps=10.0
            ),
            CandidateConfig(
                name="Candidate_C_Hysteresis_30_Cooldown_3d",
                description="Hysteresis exit threshold at -0.30 plus 3-trading-day cooldown (10% cap).",
                max_asset_weight=0.10,
                hysteresis_sell_threshold=-0.30,
                cooldown_days=3,
                cost_bps=10.0
            ),
            CandidateConfig(
                name="Benchmark_Passive_Buy_and_Hold",
                description="Passive Equal-Weight Buy & Hold benchmark (10% per equity, zero turnover).",
                max_asset_weight=0.10,
                cost_bps=10.0,
                is_benchmark=True
            )
        ]

        # 1. Primary Controlled Comparison across Splits (Val, Test, Full)
        splits = [
            ("VALIDATION", self.val_dates),
            ("UNTOUCHED_TEST", self.test_dates),
            ("FULL_300D", self.dates)
        ]

        all_metrics_list = []
        equity_curves = {}
        per_stock_portfolio_records = []

        for cfg in candidates:
            for s_name, s_dates in splits:
                met, curve, ps_dict = self.simulate_candidate(cfg, s_dates)
                all_metrics_list.append(met.to_dict())

                if s_name == "FULL_300D":
                    equity_curves[cfg.name] = curve
                    for t, s_info in ps_dict.items():
                        s_info["candidate_name"] = cfg.name
                        per_stock_portfolio_records.append(s_info)

        # Compute CAPM beta and alpha against Buy & Hold on Full Period
        bh_curve = equity_curves["Benchmark_Passive_Buy_and_Hold"]["Net_Daily_Return"].values
        rf_daily = 0.03 / 252.0

        for m in all_metrics_list:
            if m["split"] == "FULL_300D":
                c_name = m["candidate_name"]
                if c_name == "Benchmark_Passive_Buy_and_Hold":
                    m["beta"] = 1.0
                    m["capm_alpha"] = 0.0
                else:
                    c_curve = equity_curves[c_name]["Net_Daily_Return"].values
                    cov_mat = np.cov(c_curve, bh_curve)
                    beta = float(cov_mat[0, 1] / cov_mat[1, 1]) if cov_mat[1, 1] > 0 else 1.0
                    alpha = float(((np.mean(c_curve) - rf_daily) - beta * (np.mean(bh_curve) - rf_daily)) * 252.0)
                    m["beta"] = beta
                    m["capm_alpha"] = alpha

        df_split_comp = pd.DataFrame(all_metrics_list)
        df_split_comp.to_csv(out_dir / "phase5d_split_comparison.csv", index=False)

        df_summary = df_split_comp[df_split_comp["split"] == "FULL_300D"].reset_index(drop=True)
        df_summary.to_csv(out_dir / "phase5d_candidate_summary.csv", index=False)

        # 2. Per-Stock Standalone Evaluation across Candidates
        per_stock_standalone_rows = []
        for cfg in candidates:
            for t in self.tickers:
                row_res = self.evaluate_single_stock_standalone(t, cfg, self.dates)
                per_stock_standalone_rows.append(row_res)

        df_ps_standalone = pd.DataFrame(per_stock_standalone_rows)
        df_ps_standalone.to_csv(out_dir / "phase5d_per_stock_breakdown.csv", index=False)

        # 3. Execution Cost Sensitivity (0, 5, 10, 20, 30 bps) on Full Period
        cost_levels = [0.0, 5.0, 10.0, 20.0, 30.0]
        cost_rows = []
        for c_bps in cost_levels:
            for cfg in candidates[:3]: # Candidates A, B, C
                c_mod = CandidateConfig(
                    name=cfg.name,
                    description=cfg.description,
                    max_asset_weight=cfg.max_asset_weight,
                    hysteresis_sell_threshold=cfg.hysteresis_sell_threshold,
                    cooldown_days=cfg.cooldown_days,
                    cost_bps=c_bps,
                    is_benchmark=cfg.is_benchmark
                )
                met, _, _ = self.simulate_candidate(c_mod, self.dates)
                cost_rows.append({
                    "cost_bps": c_bps,
                    "candidate_name": cfg.name,
                    "cumulative_return": met.cumulative_return,
                    "annualized_return": met.annualized_return,
                    "sharpe_ratio": met.sharpe_ratio,
                    "max_drawdown": met.max_drawdown,
                    "total_turnover": met.total_turnover
                })

        df_cost = pd.DataFrame(cost_rows)
        df_cost.to_csv(out_dir / "phase5d_cost_sensitivity.csv", index=False)

        # 4. Limit-Order Non-Fill Stress Scenarios (0%, 5%, 10%, 15%, 20% and Gap > 1.5%)
        non_fill_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
        stress_rows = []
        for nf in non_fill_levels:
            for cfg in candidates[:3]:
                c_mod = CandidateConfig(
                    name=cfg.name,
                    description=cfg.description,
                    max_asset_weight=cfg.max_asset_weight,
                    hysteresis_sell_threshold=cfg.hysteresis_sell_threshold,
                    cooldown_days=cfg.cooldown_days,
                    cost_bps=10.0,
                    non_fill_rate=nf
                )
                met, _, _ = self.simulate_candidate(c_mod, self.dates)
                stress_rows.append({
                    "scenario": f"Stochastic Non-Fill {int(nf*100)}%",
                    "non_fill_rate": nf,
                    "gap_limit_pct": None,
                    "candidate_name": cfg.name,
                    "cumulative_return": met.cumulative_return,
                    "annualized_return": met.annualized_return,
                    "sharpe_ratio": met.sharpe_ratio,
                    "max_drawdown": met.max_drawdown,
                    "total_turnover": met.total_turnover
                })

        # Gap-based non-fill (> 1.5% opening gap)
        for cfg in candidates[:3]:
            c_gap = CandidateConfig(
                name=cfg.name,
                description=cfg.description,
                max_asset_weight=cfg.max_asset_weight,
                hysteresis_sell_threshold=cfg.hysteresis_sell_threshold,
                cooldown_days=cfg.cooldown_days,
                cost_bps=10.0,
                gap_limit_pct=0.015
            )
            met, _, _ = self.simulate_candidate(c_gap, self.dates)
            stress_rows.append({
                "scenario": "Gap Limit Non-Fill (>1.5% Gap)",
                "non_fill_rate": 0.0,
                "gap_limit_pct": 0.015,
                "candidate_name": cfg.name,
                "cumulative_return": met.cumulative_return,
                "annualized_return": met.annualized_return,
                "sharpe_ratio": met.sharpe_ratio,
                "max_drawdown": met.max_drawdown,
                "total_turnover": met.total_turnover
            })

        df_stress = pd.DataFrame(stress_rows)
        df_stress.to_csv(out_dir / "phase5d_non_fill_stress.csv", index=False)

        # 5. News Incremental Value Ablation
        ablation_results = {}
        for cfg in candidates[:3]:
            c_news = CandidateConfig(
                name=cfg.name, description=cfg.description,
                hysteresis_sell_threshold=cfg.hysteresis_sell_threshold,
                cooldown_days=cfg.cooldown_days, cost_bps=10.0, use_news=True
            )
            c_nonews = CandidateConfig(
                name=cfg.name, description=cfg.description,
                hysteresis_sell_threshold=cfg.hysteresis_sell_threshold,
                cooldown_days=cfg.cooldown_days, cost_bps=10.0, use_news=False
            )

            # Test on Validation, Test, and Full
            cand_ablation = {}
            for s_name, s_dates in splits:
                m_news, cur_news, _ = self.simulate_candidate(c_news, s_dates)
                m_nonews, cur_nonews, _ = self.simulate_candidate(c_nonews, s_dates)

                r_news = cur_news["Net_Daily_Return"]
                r_nonews = cur_nonews["Net_Daily_Return"]
                diff_series = r_news - r_nonews

                t_stat, p_val = stats.ttest_rel(r_news, r_nonews)
                w_stat, w_pval = stats.wilcoxon(diff_series) if (diff_series != 0).any() else (0.0, 1.0)

                diff_cum = m_news.cumulative_return - m_nonews.cumulative_return
                diff_ann = (r_news.mean() - r_nonews.mean()) * 252.0

                cand_ablation[s_name] = {
                    "cum_return_with_news": m_news.cumulative_return,
                    "cum_return_without_news": m_nonews.cumulative_return,
                    "return_difference_cumulative": diff_cum,
                    "return_difference_annualized": diff_ann,
                    "sharpe_with_news": m_news.sharpe_ratio,
                    "sharpe_without_news": m_nonews.sharpe_ratio,
                    "paired_t_statistic": round(float(t_stat), 4),
                    "paired_t_pvalue": round(float(p_val), 4),
                    "wilcoxon_statistic": round(float(w_stat), 2),
                    "wilcoxon_pvalue": round(float(w_pval), 4),
                    "statistically_significant_05": bool(p_val < 0.05)
                }

            ablation_results[cfg.name] = cand_ablation

        with open(out_dir / "phase5d_news_ablation_significance.json", "w", encoding="utf-8") as f:
            json.dump(ablation_results, f, indent=2)

        # 6. Candidate Selection Decision Matrix
        # Based strictly on Validation split (Sessions 1-150)
        val_records = [m for m in all_metrics_list if m["split"] == "VALIDATION" and m["candidate_name"] != "Benchmark_Passive_Buy_and_Hold"]
        val_records.sort(key=lambda x: x["sharpe_ratio"], reverse=True)
        selected_candidate = val_records[0]["candidate_name"]

        logger.info(f"Phase 5D Candidate Selection complete. Selected on Validation: {selected_candidate}")

        return {
            "selected_candidate_on_validation": selected_candidate,
            "validation_rankings": val_records,
            "full_summary": df_summary.to_dict(orient="records"),
            "news_ablation": ablation_results
        }
