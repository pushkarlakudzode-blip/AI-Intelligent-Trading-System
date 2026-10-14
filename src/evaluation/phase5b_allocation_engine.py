"""
Phase 5B: Portfolio Allocation, Cooldown & Turnover Diagnosis Engine.
===================================================================
Provides modular, reproducible evaluation of portfolio sizing, anti-whipsaw rules,
volatility scaling, and cash management under realistic Next-Day MOO execution.
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

logger = logging.getLogger("phase5b_engine")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PHASE5B_DIR = PROJECT_ROOT / "data" / "evaluation" / "phase5b"


@dataclass
class AllocationConfig:
    name: str
    description: str
    max_asset_weight: float = 0.10
    dynamic_cash_fungible: bool = False
    target_portfolio_exposure: float = 0.90
    volatility_scaled: bool = False
    vol_lookback_days: int = 20
    cooldown_days: int = 0
    hysteresis_sell_threshold: Optional[float] = None
    buy_threshold: float = 0.20
    default_sell_threshold: float = -0.20
    cost_bps: float = 10.0
    execution_mode: str = "NEXT_OPEN"
    is_benchmark: bool = False


@dataclass
class Phase5BMetrics:
    experiment_name: str
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
    beta: Optional[float] = None
    capm_alpha: Optional[float] = None
    max_asset_weight: float = 0.10
    cooldown_days: int = 0
    volatility_scaled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Phase5BAllocationEngine:
    """
    Dedicated evaluation and diagnosis engine for portfolio allocation,
    turnover reduction, and execution realism.
    """

    def __init__(self, processed_dir: Optional[Path] = None):
        self.processed_dir = processed_dir or PROCESSED_DIR
        self.tickers = sorted(SUPPORTED_TICKERS)
        self.company_names = COMPANY_NAMES
        self.df_aligned: Optional[pd.DataFrame] = None
        self.dates: List[pd.Timestamp] = []
        self.val_dates: List[pd.Timestamp] = []
        self.test_dates: List[pd.Timestamp] = []
        self._load_and_prepare_data()

    def _load_and_prepare_data(self) -> None:
        """Load decision fusion and feature datasets with rolling backward-looking volatility."""
        fusion_path = self.processed_dir / "multi_company_decision_fusion.csv"
        feat_path = self.processed_dir / "multi_company_features.csv"

        if not fusion_path.exists():
            raise FileNotFoundError(f"Missing fusion file: {fusion_path}")
        if not feat_path.exists():
            raise FileNotFoundError(f"Missing features file: {feat_path}")

        df_fus = pd.read_csv(fusion_path)
        df_fus["Date"] = pd.to_datetime(df_fus["Date"])

        df_feat = pd.read_csv(feat_path)
        df_feat["Date"] = pd.to_datetime(df_feat["Date"])
        df_feat = df_feat.sort_values(["Ticker", "Date"]).reset_index(drop=True)

        # Forward next-day open-to-open returns (strict MOO, zero lookahead)
        df_feat["Open_t1"] = df_feat.groupby("Ticker")["Open"].shift(-1)
        df_feat["Open_t2"] = df_feat.groupby("Ticker")["Open"].shift(-2)
        df_feat["Fwd_Open_Open"] = (df_feat["Open_t2"] - df_feat["Open_t1"]) / df_feat["Open_t1"]

        # Backward-looking rolling 20-day annualized volatility from daily returns (zero lookahead)
        df_feat["Vol_20"] = (
            df_feat.groupby("Ticker")["Daily_Return"].rolling(20).std().reset_index(0, drop=True)
            * np.sqrt(252.0)
        )

        # Merge with decision fusion test set
        df = pd.merge(
            df_fus,
            df_feat[["Date", "Ticker", "Fwd_Open_Open", "Vol_20"]],
            on=["Date", "Ticker"],
            how="inner"
        )
        df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
        self.df_aligned = df

        self.dates = sorted(df["Date"].unique())
        n_dates = len(self.dates)
        split_idx = n_dates // 2
        self.val_dates = self.dates[:split_idx]
        self.test_dates = self.dates[split_idx:]

        logger.info(
            f"Phase 5B Engine initialized: {len(df)} rows across {n_dates} dates "
            f"(Validation: {len(self.val_dates)}d, Untouched Test: {len(self.test_dates)}d)."
        )

    def diagnose_baseline_behavior(self) -> Dict[str, Any]:
        """
        Produce a comprehensive mathematical diagnosis of the baseline strategy:
        - Distribution of actively held stocks (why cash remains 52.9%)
        - Whipsaw frequency and holding duration per stock (why turnover is 2,932%)
        - Idle capacity per asset
        """
        df = self.df_aligned
        holding = {t: 0 for t in self.tickers}
        held_counts = []
        hold_durations: Dict[str, List[int]] = {t: [] for t in self.tickers}
        current_dur = {t: 0 for t in self.tickers}
        whipsaws_3d = {t: 0 for t in self.tickers}
        ticker_buy_counts = {t: 0 for t in self.tickers}
        ticker_sell_counts = {t: 0 for t in self.tickers}
        ticker_hold_counts = {t: 0 for t in self.tickers}

        for d in self.dates:
            day_df = df[df["Date"] == d].set_index("Ticker")
            active_today = 0
            for t in self.tickers:
                dec = day_df.loc[t, "Final_Decision"] if t in day_df.index else "HOLD"
                if dec == "BUY":
                    ticker_buy_counts[t] += 1
                    if holding[t] == 0:
                        holding[t] = 1
                        current_dur[t] = 1
                    else:
                        current_dur[t] += 1
                elif dec == "SELL":
                    ticker_sell_counts[t] += 1
                    if holding[t] == 1:
                        dur = current_dur[t]
                        hold_durations[t].append(dur)
                        if dur <= 3:
                            whipsaws_3d[t] += 1
                        holding[t] = 0
                        current_dur[t] = 0
                else: # HOLD
                    ticker_hold_counts[t] += 1
                    if holding[t] == 1:
                        current_dur[t] += 1

                active_today += holding[t]
            held_counts.append(active_today)

        # Flush open positions at end of period
        for t in self.tickers:
            if holding[t] == 1 and current_dur[t] > 0:
                hold_durations[t].append(current_dur[t])

        per_stock_diag = []
        total_roundtrips = 0
        total_whipsaws = 0
        for t in self.tickers:
            durs = hold_durations[t]
            n_rt = len(durs)
            total_roundtrips += n_rt
            total_whipsaws += whipsaws_3d[t]
            avg_dur = float(np.mean(durs)) if durs else 0.0
            med_dur = float(np.median(durs)) if durs else 0.0
            per_stock_diag.append({
                "ticker": t,
                "company_name": self.company_names.get(t, t),
                "buy_days": ticker_buy_counts[t],
                "sell_days": ticker_sell_counts[t],
                "hold_days": ticker_hold_counts[t],
                "round_trips": n_rt,
                "avg_holding_duration_days": round(avg_dur, 2),
                "median_holding_duration_days": round(med_dur, 1),
                "whipsaws_le_3d": whipsaws_3d[t],
                "whipsaw_ratio": round(whipsaws_3d[t] / n_rt, 3) if n_rt > 0 else 0.0
            })

        held_s = pd.Series(held_counts)
        dist = held_s.value_counts().sort_index().to_dict()

        diagnosis = {
            "total_trading_sessions": len(self.dates),
            "mean_actively_held_stocks": round(float(np.mean(held_counts)), 2),
            "median_actively_held_stocks": int(np.median(held_counts)),
            "min_actively_held_stocks": int(np.min(held_counts)),
            "max_actively_held_stocks": int(np.max(held_counts)),
            "active_stocks_distribution": dist,
            "total_round_trips": total_roundtrips,
            "total_whipsaws_le_3d": total_whipsaws,
            "portfolio_whipsaw_percentage": round(total_whipsaws / total_roundtrips * 100.0, 1) if total_roundtrips > 0 else 0.0,
            "per_stock_diagnosis": per_stock_diag
        }
        return diagnosis

    def simulate_strategy(
        self,
        config: AllocationConfig,
        target_dates: List[pd.Timestamp],
        initial_capital: float = 100000.0
    ) -> Tuple[Phase5BMetrics, pd.DataFrame]:
        """
        Execute portfolio simulation with Strict Dollar Ledger cash accounting,
        cooldown checks, volatility-scaled weights, and friction tracking.
        """
        cost_rate = config.cost_bps / 10000.0
        cash = initial_capital
        stock_dollars = {t: 0.0 for t in self.tickers}
        hold_duration = {t: 0 for t in self.tickers}
        trades = 0
        turnover_dollars_total = 0.0
        whipsaws_3d = 0
        history = []

        sub_df = self.df_aligned

        for d in target_dates:
            day_df = sub_df[sub_df["Date"] == d].set_index("Ticker")
            total_eq_start = cash + sum(stock_dollars.values())
            day_turnover = 0.0

            # 1. Determine Desired Signals
            desired_actions = {}
            for t in self.tickers:
                if config.is_benchmark:
                    desired_actions[t] = "BUY"
                    continue

                if t not in day_df.index:
                    desired_actions[t] = "HOLD"
                    continue

                row = day_df.loc[t]
                raw_dec = row["Final_Decision"]
                score = row["Fusion_Score"]

                # Apply Hysteresis if configured
                if config.hysteresis_sell_threshold is not None:
                    if score >= config.buy_threshold:
                        dec = "BUY"
                    elif score <= config.hysteresis_sell_threshold:
                        dec = "SELL"
                    else:
                        dec = "HOLD"
                else:
                    dec = raw_dec

                # Apply Cooldown / Minimum Holding Period Rule
                if dec == "SELL" and stock_dollars[t] > 0 and hold_duration[t] < config.cooldown_days:
                    dec = "HOLD"

                desired_actions[t] = dec

            # 2. Determine Target Portfolio Allocation Weights
            active_assets = [
                t for t, a in desired_actions.items()
                if a == "BUY" or (a == "HOLD" and stock_dollars[t] > 0)
            ]

            target_weights = {t: 0.0 for t in self.tickers}
            if config.is_benchmark:
                # 10% equal weight across all 10 assets
                for t in self.tickers:
                    target_weights[t] = 0.10
            elif config.volatility_scaled:
                # Backward-looking inverse-volatility weights
                inv_vols = {}
                for t in self.tickers:
                    if t in day_df.index and pd.notnull(day_df.loc[t, "Vol_20"]) and day_df.loc[t, "Vol_20"] > 0:
                        inv_vols[t] = 1.0 / day_df.loc[t, "Vol_20"]
                    else:
                        inv_vols[t] = 1.0 / 0.25 # 25% default vol fallback
                
                if config.dynamic_cash_fungible:
                    # Dynamically scale active weights to target portfolio exposure (e.g. 90%)
                    active_inv_sum = sum(inv_vols[t] for t in active_assets)
                    if active_inv_sum > 0:
                        for t in active_assets:
                            w_raw = (inv_vols[t] / active_inv_sum) * config.target_portfolio_exposure
                            target_weights[t] = min(w_raw, config.max_asset_weight)
                else:
                    # Normalized around standard 10% average allocation
                    mean_inv = np.mean(list(inv_vols.values()))
                    for t in active_assets:
                        w_raw = (inv_vols[t] / mean_inv) * 0.10
                        target_weights[t] = min(w_raw, config.max_asset_weight)
            else:
                if config.dynamic_cash_fungible:
                    n_act = len(active_assets)
                    w_each = min(config.target_portfolio_exposure / n_act, config.max_asset_weight) if n_act > 0 else 0.0
                    for t in active_assets:
                        target_weights[t] = w_each
                else:
                    for t in active_assets:
                        target_weights[t] = min(0.10, config.max_asset_weight)

            # 3. Execute SELLs first to generate cash
            for t in self.tickers:
                if desired_actions[t] == "SELL":
                    curr_val = stock_dollars[t]
                    if curr_val > 0:
                        fee = curr_val * cost_rate
                        cash += (curr_val - fee)
                        day_turnover += curr_val
                        stock_dollars[t] = 0.0
                        if hold_duration[t] <= 3:
                            whipsaws_3d += 1
                        hold_duration[t] = 0
                        trades += 1
                elif desired_actions[t] == "HOLD":
                    if stock_dollars[t] > 0:
                        hold_duration[t] += 1

            # 4. Execute BUYs / Target Rebalances using cash
            for t in self.tickers:
                if desired_actions[t] == "BUY":
                    curr_val = stock_dollars[t]
                    target_val = target_weights[t] * total_eq_start
                    delta = target_val - curr_val
                    # Trade threshold to prevent sub-dollar noise rebalancing
                    if delta > 10.0:
                        buy_amt = min(delta, cash)
                        if buy_amt > 10.0:
                            fee = buy_amt * cost_rate
                            cash -= (buy_amt + fee)
                            stock_dollars[t] += buy_amt
                            day_turnover += buy_amt
                            if curr_val == 0:
                                hold_duration[t] = 1
                            else:
                                hold_duration[t] += 1
                            trades += 1
                    else:
                        if stock_dollars[t] > 0:
                            hold_duration[t] += 1

            # 5. Holding Return from MOO Fill (Open_{t+1} to Open_{t+2})
            for t in self.tickers:
                if stock_dollars[t] > 0 and t in day_df.index:
                    r = day_df.loc[t, "Fwd_Open_Open"]
                    if pd.notnull(r):
                        stock_dollars[t] *= (1.0 + r)

            total_eq_end = cash + sum(stock_dollars.values())
            day_ret = (total_eq_end / total_eq_start) - 1.0 if total_eq_start > 0 else 0.0
            turnover_dollars_total += day_turnover

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

        metrics = Phase5BMetrics(
            experiment_name=config.name,
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
            max_asset_weight=config.max_asset_weight,
            cooldown_days=config.cooldown_days,
            volatility_scaled=config.volatility_scaled
        )
        return metrics, res_df

    def run_all_experiments(
        self,
        cost_bps: float = 10.0,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Execute full suite of Phase 5B experiments across Validation, Test, and Full periods.
        """
        out_dir = output_dir or PHASE5B_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        configs = [
            AllocationConfig(
                name="Exp_A_Baseline_Fusion",
                description="Current baseline strategy with static 10% cap per equity and raw fusion signals.",
                max_asset_weight=0.10,
                cost_bps=cost_bps
            ),
            AllocationConfig(
                name="Exp_B1_Cooldown_K3",
                description="Baseline with 3-day minimum holding period to prevent whipsaws.",
                max_asset_weight=0.10,
                cooldown_days=3,
                cost_bps=cost_bps
            ),
            AllocationConfig(
                name="Exp_B2_Cooldown_K5",
                description="Baseline with 5-day minimum holding period.",
                max_asset_weight=0.10,
                cooldown_days=5,
                cost_bps=cost_bps
            ),
            AllocationConfig(
                name="Exp_B3_Hysteresis_Exit",
                description="Asymmetric exit threshold (Sell score <= -0.30 instead of -0.20) with 10% cap.",
                max_asset_weight=0.10,
                hysteresis_sell_threshold=-0.30,
                cost_bps=cost_bps
            ),
            AllocationConfig(
                name="Exp_B4_Combined_AntiWhipsaw",
                description="Combined anti-whipsaw: 3-day cooldown + mild hysteresis (Sell <= -0.25).",
                max_asset_weight=0.10,
                cooldown_days=3,
                hysteresis_sell_threshold=-0.25,
                cost_bps=cost_bps
            ),
            AllocationConfig(
                name="Exp_C1_Volatility_Scaled_10Cap",
                description="Inverse-volatility position sizing strictly capped at 10% per equity.",
                max_asset_weight=0.10,
                volatility_scaled=True,
                cost_bps=cost_bps
            ),
            AllocationConfig(
                name="Exp_C2_VolScaled_DynamicCash_20Cap",
                description="Inverse-volatility sizing with fungible cash (expanded 20% cap) to eliminate cash drag.",
                max_asset_weight=0.20,
                dynamic_cash_fungible=True,
                volatility_scaled=True,
                cost_bps=cost_bps
            ),
            AllocationConfig(
                name="Exp_B_plus_C_Optimized",
                description="Integrated: 3-day cooldown + inverse-volatility + dynamic cash (20% cap).",
                max_asset_weight=0.20,
                dynamic_cash_fungible=True,
                volatility_scaled=True,
                cooldown_days=3,
                cost_bps=cost_bps
            ),
            AllocationConfig(
                name="Exp_D_Passive_Buy_and_Hold",
                description="Equal-weight buy-and-hold benchmark (10% per asset, zero active turnover).",
                max_asset_weight=0.10,
                cost_bps=cost_bps,
                is_benchmark=True
            )
        ]

        # 1. Baseline Diagnosis
        diag = self.diagnose_baseline_behavior()
        with open(out_dir / "turnover_diagnosis.json", "w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2)

        # 2. Run Simulations across Splits
        all_metrics_list = []
        equity_curves = {}

        splits = [
            ("VALIDATION", self.val_dates),
            ("UNTOUCHED_TEST", self.test_dates),
            ("FULL_300D", self.dates)
        ]

        for cfg in configs:
            for s_name, s_dates in splits:
                met, curve = self.simulate_strategy(cfg, s_dates)
                all_metrics_list.append(met.to_dict())
                if s_name == "FULL_300D":
                    equity_curves[cfg.name] = curve

        # Calculate CAPM beta and alpha against Buy & Hold on Full Period
        bh_curve = equity_curves["Exp_D_Passive_Buy_and_Hold"]["Net_Daily_Return"].values
        rf_daily = 0.03 / 252.0

        for m in all_metrics_list:
            if m["split"] == "FULL_300D":
                s_name = m["experiment_name"]
                if s_name == "Exp_D_Passive_Buy_and_Hold":
                    m["beta"] = 1.0
                    m["capm_alpha"] = 0.0
                else:
                    s_curve = equity_curves[s_name]["Net_Daily_Return"].values
                    cov_mat = np.cov(s_curve, bh_curve)
                    beta = float(cov_mat[0, 1] / cov_mat[1, 1]) if cov_mat[1, 1] > 0 else 1.0
                    alpha = float(((np.mean(s_curve) - rf_daily) - beta * (np.mean(bh_curve) - rf_daily)) * 252.0)
                    m["beta"] = beta
                    m["capm_alpha"] = alpha

        # 3. Friction Sensitivity on Full Period (0, 5, 10, 20, 50 bps)
        friction_levels = [0.0, 5.0, 10.0, 20.0, 50.0]
        friction_rows = []
        for c_bps in friction_levels:
            for cfg in [configs[0], configs[1], configs[3], configs[5], configs[6], configs[7], configs[8]]:
                c_copy = AllocationConfig(
                    name=cfg.name,
                    description=cfg.description,
                    max_asset_weight=cfg.max_asset_weight,
                    dynamic_cash_fungible=cfg.dynamic_cash_fungible,
                    target_portfolio_exposure=cfg.target_portfolio_exposure,
                    volatility_scaled=cfg.volatility_scaled,
                    cooldown_days=cfg.cooldown_days,
                    hysteresis_sell_threshold=cfg.hysteresis_sell_threshold,
                    cost_bps=c_bps,
                    is_benchmark=cfg.is_benchmark
                )
                met, _ = self.simulate_strategy(c_copy, self.dates)
                friction_rows.append({
                    "cost_bps": c_bps,
                    "experiment_name": cfg.name,
                    "cumulative_return": met.cumulative_return,
                    "annualized_return": met.annualized_return,
                    "sharpe_ratio": met.sharpe_ratio,
                    "max_drawdown": met.max_drawdown,
                    "total_turnover": met.total_turnover
                })

        # Save artifacts
        df_all = pd.DataFrame(all_metrics_list)
        df_all.to_csv(out_dir / "split_comparison.csv", index=False)

        df_full = df_all[df_all["split"] == "FULL_300D"].reset_index(drop=True)
        df_full.to_csv(out_dir / "experiment_summary.csv", index=False)

        df_fric = pd.DataFrame(friction_rows)
        df_fric.to_csv(out_dir / "cost_sensitivity_comparison.csv", index=False)

        df_diag_per_stock = pd.DataFrame(diag["per_stock_diagnosis"])
        df_diag_per_stock.to_csv(out_dir / "per_stock_experiment_breakdown.csv", index=False)

        logger.info(f"Phase 5B experiments complete. Artifacts written to {out_dir}")
        return {
            "full_summary": df_full.to_dict(orient="records"),
            "diagnosis": diag,
            "total_simulations": len(all_metrics_list)
        }
