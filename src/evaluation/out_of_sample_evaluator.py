"""
Phase 4 Out-of-Sample Evaluation Module
=======================================
Provides institutional-grade backtesting, portfolio simulation, transaction cost modeling,
and statistical uncertainty estimation for the AI-Based Intelligent Trading System.

Design Principles:
1. Strict Chronological Ordering: All test inputs derived on or before date t.
2. Comprehensive Baselines: Buy & Hold, Market-Only, RL-Only, Classification-Only, News-Ablated.
3. Realistic Market Microstructure: Configurable commissions, bid-ask slippage, cash constraints,
   and portfolio exposure bounds.
4. Statistical Rigor: Stationary block bootstrap confidence intervals, paired hypothesis tests,
   and slice-based equity decomposition.
5. Zero System Mutation: Operates as a read-only analytical harness without modifying
   production weights, database tables, or historical research.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("out_of_sample_evaluator")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation" / "phase4"

# 10 Tracked Portfolio Companies
SUPPORTED_TICKERS = [
    "AAPL", "AMZN", "GOOGL", "JNJ", "JPM", "META", "MSFT", "NVDA", "TSLA", "V"
]

COMPANY_NAMES = {
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase",
    "META": "Meta Platforms",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "V": "Visa"
}

SECTOR_MAPPINGS = {
    "AAPL": "Technology",
    "AMZN": "Consumer Cyclical",
    "GOOGL": "Communication Services",
    "JNJ": "Healthcare",
    "JPM": "Financial Services",
    "META": "Communication Services",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "TSLA": "Consumer Cyclical",
    "V": "Financial Services"
}


@dataclass
class PerformanceMetrics:
    """Standardized institutional metrics record."""
    strategy_name: str
    ticker: str  # Ticker symbol or 'PORTFOLIO_10'
    mode: str    # 'LONG_ONLY' or 'LONG_SHORT'
    cost_bps: float
    total_sessions: int
    initial_capital: float
    final_equity: float
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    mean_exposure: float
    total_turnover: float
    trade_count: int
    win_rate: float
    profit_factor: float
    execution_mode: str = "NEXT_OPEN"  # 'NEXT_OPEN', 'SAME_CLOSE', 'INTRADAY_OPEN_CLOSE'
    accounting_model: str = "DOLLAR_LEDGER"  # 'DOLLAR_LEDGER' or 'WEIGHT_COMPOUNDING'
    standalone_days_invested: Optional[float] = None
    portfolio_weight_contribution: Optional[float] = None
    capm_alpha: Optional[float] = None
    alpha_ann: Optional[float] = None
    beta: Optional[float] = None
    information_ratio: Optional[float] = None
    sharpe_ci_lower: Optional[float] = None
    sharpe_ci_upper: Optional[float] = None
    return_ci_lower: Optional[float] = None
    return_ci_upper: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_max_drawdown(equity_series: pd.Series) -> Tuple[float, int]:
    """Calculate maximum peak-to-trough drawdown and duration in days."""
    cum_max = equity_series.cummax()
    drawdowns = (equity_series - cum_max) / cum_max
    max_dd = float(drawdowns.min())
    
    # Calculate duration
    duration = 0
    max_duration = 0
    for dd in drawdowns:
        if dd < 0:
            duration += 1
            if duration > max_duration:
                max_duration = duration
        else:
            duration = 0
            
    return max_dd, max_duration


def calculate_sharpe_ratio(
    return_series: pd.Series, risk_free_rate: float = 0.03
) -> float:
    """Calculate annualized Sharpe Ratio using 252 trading days."""
    if len(return_series) == 0:
        return 0.0
    daily_rf = risk_free_rate / 252.0
    excess_daily = return_series - daily_rf
    std = float(return_series.std())
    if std <= 1e-8 or np.isnan(std):
        return 0.0
    return float(np.sqrt(252.0) * excess_daily.mean() / std)


def calculate_sortino_ratio(
    return_series: pd.Series, risk_free_rate: float = 0.03
) -> float:
    """Calculate annualized Sortino Ratio focusing on downside risk."""
    if len(return_series) == 0:
        return 0.0
    daily_rf = risk_free_rate / 252.0
    excess_daily = return_series - daily_rf
    downside = return_series[return_series < daily_rf] - daily_rf
    downside_std = float(np.sqrt(np.mean(downside**2))) if len(downside) > 0 else 0.0
    if downside_std <= 1e-8 or np.isnan(downside_std):
        return 0.0
    return float(np.sqrt(252.0) * excess_daily.mean() / downside_std)


def bootstrap_metric_ci(
    return_series: pd.Series,
    n_bootstraps: int = 1000,
    block_size: int = 5,
    risk_free_rate: float = 0.03,
    alpha: float = 0.05
) -> Dict[str, Tuple[float, float]]:
    """
    Compute stationary block-bootstrap confidence intervals for Sharpe Ratio and Cumulative Return.
    Block bootstrapping preserves temporal autocorrelation in financial time series.
    """
    n = len(return_series)
    if n < block_size * 2:
        return {
            "sharpe_ci": (0.0, 0.0),
            "cum_return_ci": (0.0, 0.0)
        }
        
    rets = return_series.values
    sharpe_dist = []
    cum_ret_dist = []
    
    rng = np.random.default_rng(seed=42)
    n_blocks = int(np.ceil(n / block_size))
    
    for _ in range(n_bootstraps):
        start_indices = rng.integers(0, n - block_size + 1, size=n_blocks)
        boot_sample = np.concatenate([rets[i:i + block_size] for i in start_indices])[:n]
        
        # Calculate bootstrap Sharpe
        s_std = np.std(boot_sample)
        if s_std > 1e-8:
            s_sh = np.sqrt(252.0) * (np.mean(boot_sample) - (risk_free_rate / 252.0)) / s_std
        else:
            s_sh = 0.0
        sharpe_dist.append(s_sh)
        
        # Calculate bootstrap Cumulative Return
        s_cum = np.prod(1.0 + boot_sample) - 1.0
        cum_ret_dist.append(s_cum)
        
    low_pct = (alpha / 2.0) * 100.0
    high_pct = (1.0 - alpha / 2.0) * 100.0
    
    sh_low, sh_high = np.percentile(sharpe_dist, [low_pct, high_pct])
    cr_low, cr_high = np.percentile(cum_ret_dist, [low_pct, high_pct])
    
    return {
        "sharpe_ci": (float(sh_low), float(sh_high)),
        "cum_return_ci": (float(cr_low), float(cr_high))
    }


class OutOfSampleEvaluator:
    """
    Comprehensive Out-of-Sample Strategy Evaluator for the Multi-Modal System.
    """

    def __init__(self, processed_dir: Optional[Path] = None):
        self.processed_dir = processed_dir or PROCESSED_DIR
        self.df_aligned: Optional[pd.DataFrame] = None
        self.tickers = SUPPORTED_TICKERS
        self.company_names = COMPANY_NAMES
        self._load_and_align_data()

    def _load_and_align_data(self) -> None:
        """Load and strictly align all upstream component outputs."""
        fusion_path = self.processed_dir / "multi_company_decision_fusion.csv"
        rl_path = self.processed_dir / "multi_company_rl_results.csv"
        feat_path = self.processed_dir / "multi_company_features.csv"

        if not fusion_path.exists():
            raise FileNotFoundError(f"Missing fusion dataset: {fusion_path}")
        if not rl_path.exists():
            raise FileNotFoundError(f"Missing RL dataset: {rl_path}")

        df_fus = pd.read_csv(fusion_path)
        df_rl = pd.read_csv(rl_path)

        # Merge on (Date, Ticker)
        df = pd.merge(
            df_fus,
            df_rl[["Date", "Ticker", "Forward_Return"]],
            on=["Date", "Ticker"],
            how="inner"
        )

        # Merge with raw features if available to capture Open/Close execution prices
        if feat_path.exists():
            df_feat = pd.read_csv(feat_path)
            df_feat["Open_t1"] = df_feat.groupby("Ticker")["Open"].shift(-1)
            df_feat["Open_t2"] = df_feat.groupby("Ticker")["Open"].shift(-2)
            df_feat["Close_t1"] = df_feat.groupby("Ticker")["Close"].shift(-1)
            
            # Model A: Close-to-Close (Same-Bar Close / MOC Fill, theoretical upper bound)
            df_feat["Fwd_Close_Close"] = (df_feat["Close_t1"] - df_feat["Close"]) / df_feat["Close"]
            
            # Model B: Next-Day Open-to-Open (MOO Fill, strict zero lookahead)
            df_feat["Fwd_Open_Open"] = (df_feat["Open_t2"] - df_feat["Open_t1"]) / df_feat["Open_t1"]
            
            # Model C: Next-Day Open-to-Close (Intraday execution)
            df_feat["Fwd_Open_Close"] = (df_feat["Close_t1"] - df_feat["Open_t1"]) / df_feat["Open_t1"]
            
            df = pd.merge(
                df,
                df_feat[["Date", "Ticker", "Fwd_Close_Close", "Fwd_Open_Open", "Fwd_Open_Close"]],
                on=["Date", "Ticker"],
                how="left"
            )
        else:
            df["Fwd_Close_Close"] = df["Forward_Return"]
            df["Fwd_Open_Open"] = df["Forward_Return"]
            df["Fwd_Open_Close"] = df["Forward_Return"]

        # Precompute Ablated Signals:
        # 1. Ablated Fusion without news component:
        # Scaled weights: Class = 0.35/0.80 = 0.4375, RL = 0.30/0.80 = 0.3750, Regime = 0.15/0.80 = 0.1875
        df["Score_NoNews"] = (
            0.4375 * df["Classification_Signal"] +
            0.3750 * df["RL_Signal"] +
            0.1875 * df["Market_Regime_Signal"]
        ).round(4)
        df["Decision_NoNews"] = np.where(
            df["Score_NoNews"] >= 0.20, "BUY",
            np.where(df["Score_NoNews"] <= -0.20, "SELL", "HOLD")
        )

        # 2. RL Only signal:
        df["Decision_RL"] = np.where(
            df["RL_Action"] == 2, "BUY",
            np.where(df["RL_Action"] == 0, "SELL", "HOLD")
        )

        # 3. Classification Only signal:
        df["Decision_Class"] = np.where(
            df["Classification_Signal"] >= 0.20, "BUY",
            np.where(df["Classification_Signal"] <= -0.20, "SELL", "HOLD")
        )

        # Sort chronologically
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
        self.df_aligned = df
        logger.info(
            f"Successfully aligned {len(df):,} observations across {df['Ticker'].nunique()} tickers."
        )

    def evaluate_single_stock(
        self,
        ticker: str,
        strategy_col: Optional[str] = "Final_Decision",
        strategy_name: Optional[str] = None,
        execution_mode: str = "NEXT_OPEN",
        accounting_model: str = "DOLLAR_LEDGER",
        mode: str = "LONG_ONLY",
        cost_bps: float = 10.0,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.03
    ) -> Tuple[PerformanceMetrics, pd.DataFrame]:
        """
        Simulate trading on an individual stock under realistic transaction costs.
        
        Parameters:
        - strategy_col: Column indicating decision ('Final_Decision', 'Decision_NoNews', 'Decision_RL', 'Decision_Class')
                        or None for Buy & Hold.
        - strategy_name: Optional explicit name label for the strategy.
        - execution_mode: 'NEXT_OPEN' (MOO fill, zero lookahead), 'SAME_CLOSE' (MOC fill), 'INTRADAY_OPEN_CLOSE'
        - accounting_model: 'DOLLAR_LEDGER' or 'WEIGHT_COMPOUNDING'
        - mode: 'LONG_ONLY' (BUY=1.0, HOLD=keep, SELL=0.0) or 'LONG_SHORT' (BUY=1.0, HOLD=keep, SELL=-1.0).
        - cost_bps: Basis points charged per unit of position turnover (e.g. 10 bps = 0.0010).
        """
        sub = self.df_aligned[self.df_aligned["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
        n_sessions = len(sub)
        cost_rate = cost_bps / 10000.0
        
        # Select appropriate forward return column
        if execution_mode == "NEXT_OPEN":
            ret_col = "Fwd_Open_Open"
        elif execution_mode == "INTRADAY_OPEN_CLOSE":
            ret_col = "Fwd_Open_Close"
        else:
            ret_col = "Fwd_Close_Close"
            
        pos = 0.0
        equity = initial_capital
        history = []
        
        trades = 0
        profit_trades = 0
        gross_profits = 0.0
        gross_losses = 0.0
        trade_entry_equity = 0.0
        
        for i, row in sub.iterrows():
            prev_pos = pos
            if strategy_col is None:
                target_pos = 1.0
            else:
                dec = row[strategy_col]
                if dec == "BUY":
                    target_pos = 1.0
                elif dec == "SELL":
                    target_pos = -1.0 if mode == "LONG_SHORT" else 0.0
                else:  # HOLD
                    target_pos = prev_pos
                    
            turnover = abs(target_pos - prev_pos)
            fee = turnover * cost_rate
            
            f_ret = row[ret_col] if ret_col in row and pd.notnull(row[ret_col]) else row["Forward_Return"]
            net_daily_ret = target_pos * f_ret - fee
            
            prev_equity = equity
            equity = equity * (1.0 + net_daily_ret)
            
            if turnover > 0:
                trades += 1
                if prev_pos != 0.0:
                    trade_pnl = prev_equity - trade_entry_equity
                    if trade_pnl > 0:
                        profit_trades += 1
                        gross_profits += trade_pnl
                    else:
                        gross_losses += abs(trade_pnl)
                trade_entry_equity = prev_equity
                
            pos = target_pos
            history.append({
                "Date": row["Date"],
                "Ticker": ticker,
                "Position": pos,
                "Turnover": turnover,
                "Fee": fee,
                "Forward_Return": f_ret,
                "Net_Daily_Return": net_daily_ret,
                "Equity": equity
            })
            
        res_df = pd.DataFrame(history)
        cum_ret = (equity / initial_capital) - 1.0
        ann_ret = (1.0 + cum_ret) ** (252.0 / n_sessions) - 1.0 if n_sessions > 0 else 0.0
        ann_vol = float(res_df["Net_Daily_Return"].std() * np.sqrt(252.0))
        sharpe = calculate_sharpe_ratio(res_df["Net_Daily_Return"], risk_free_rate)
        sortino = calculate_sortino_ratio(res_df["Net_Daily_Return"], risk_free_rate)
        mdd, mdd_dur = calculate_max_drawdown(res_df["Equity"])
        calmar = abs(ann_ret / mdd) if abs(mdd) > 1e-4 else 0.0
        mean_exp = float(res_df["Position"].abs().mean())
        tot_turnover = float(res_df["Turnover"].sum())
        
        win_rate = (profit_trades / trades) if trades > 0 else 0.0
        profit_factor = (gross_profits / gross_losses) if gross_losses > 0 else (10.0 if gross_profits > 0 else 1.0)
        
        # Standalone days invested (% of trading sessions in market)
        standalone_days_inv = float((res_df["Position"] != 0.0).mean())
        
        boot_ci = bootstrap_metric_ci(res_df["Net_Daily_Return"], risk_free_rate=risk_free_rate)
        
        strat_label = strategy_name or (strategy_col if strategy_col is not None else "Buy_and_Hold")
        metrics = PerformanceMetrics(
            strategy_name=strat_label,
            ticker=ticker,
            mode=mode,
            cost_bps=cost_bps,
            total_sessions=n_sessions,
            initial_capital=initial_capital,
            final_equity=equity,
            cumulative_return=cum_ret,
            annualized_return=ann_ret,
            annualized_volatility=ann_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=mdd,
            max_drawdown_duration_days=mdd_dur,
            mean_exposure=mean_exp,
            total_turnover=tot_turnover,
            trade_count=trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            execution_mode=execution_mode,
            accounting_model=accounting_model,
            standalone_days_invested=standalone_days_inv,
            sharpe_ci_lower=boot_ci["sharpe_ci"][0],
            sharpe_ci_upper=boot_ci["sharpe_ci"][1],
            return_ci_lower=boot_ci["cum_return_ci"][0],
            return_ci_upper=boot_ci["cum_return_ci"][1]
        )
        return metrics, res_df

    def evaluate_multi_asset_portfolio(
        self,
        strategy_col: Optional[str] = "Final_Decision",
        strategy_name: Optional[str] = None,
        execution_mode: str = "NEXT_OPEN",
        accounting_model: str = "DOLLAR_LEDGER",
        mode: str = "LONG_ONLY",
        cost_bps: float = 10.0,
        initial_capital: float = 100000.0,
        max_asset_weight: float = 0.10,
        risk_free_rate: float = 0.03
    ) -> Tuple[PerformanceMetrics, pd.DataFrame]:
        """
        Simulate an actively managed 10-asset portfolio with cash constraints and position sizing.
        Supports both Strict Dollar Ledger (Phase 5A) and Weight Compounding (Phase 4).
        """
        dates = sorted(self.df_aligned["Date"].unique())
        cost_rate = cost_bps / 10000.0
        
        # Select return column
        if execution_mode == "NEXT_OPEN":
            ret_col = "Fwd_Open_Open"
        elif execution_mode == "INTRADAY_OPEN_CLOSE":
            ret_col = "Fwd_Open_Close"
        else:
            ret_col = "Fwd_Close_Close"
            
        history = []
        trades = 0
        total_turnover_sum = 0.0
        
        if accounting_model == "DOLLAR_LEDGER":
            # Strict Dollar Accounting with Explicit Cash Ledger
            cash = initial_capital
            stock_dollars = {t: 0.0 for t in self.tickers}
            stock_weights_hist = {t: [] for t in self.tickers}
            
            for d in dates:
                day_df = self.df_aligned[self.df_aligned["Date"] == d].set_index("Ticker")
                total_eq_start = cash + sum(stock_dollars.values())
                turnover_dollars = 0.0
                
                for t in self.tickers:
                    if strategy_col is None:
                        dec = "BUY"
                    else:
                        dec = day_df.loc[t, strategy_col] if t in day_df.index else "HOLD"
                    curr_val = stock_dollars[t]
                    
                    if dec == "BUY":
                        target_val = max_asset_weight * total_eq_start
                        delta = target_val - curr_val
                        if delta > 0:
                            buy_amt = min(delta, cash)
                            fee = buy_amt * cost_rate
                            cash -= (buy_amt + fee)
                            stock_dollars[t] += buy_amt
                            turnover_dollars += buy_amt
                            trades += 1
                    elif dec == "SELL":
                        if curr_val > 0:
                            fee = curr_val * cost_rate
                            cash += (curr_val - fee)
                            turnover_dollars += curr_val
                            stock_dollars[t] = 0.0
                            trades += 1
                    # If HOLD: maintain position in stock without artificial rebalance drag
                    
                for t in self.tickers:
                    stock_weights_hist[t].append(
                        stock_dollars[t] / total_eq_start if total_eq_start > 0 else 0.0
                    )
                    
                # Holding return for the session
                for t in self.tickers:
                    if stock_dollars[t] > 0 and t in day_df.index:
                        r = day_df.loc[t, ret_col] if ret_col in day_df.columns else day_df.loc[t, "Forward_Return"]
                        if pd.notnull(r):
                            stock_dollars[t] *= (1.0 + r)
                            
                total_eq_end = cash + sum(stock_dollars.values())
                day_ret = (total_eq_end / total_eq_start) - 1.0 if total_eq_start > 0 else 0.0
                turnover_fraction = turnover_dollars / total_eq_start if total_eq_start > 0 else 0.0
                total_turnover_sum += turnover_fraction
                
                history.append({
                    "Date": d,
                    "Equity": total_eq_end,
                    "Cash": cash,
                    "Net_Daily_Return": day_ret,
                    "Fee": turnover_dollars * cost_rate,
                    "Turnover": turnover_fraction,
                    "Portfolio_Exposure": sum(stock_dollars.values()) / total_eq_end if total_eq_end > 0 else 0.0
                })
            equity = total_eq_end
        else:
            # Legacy Weight-Compounding Accounting
            equity = initial_capital
            current_weights = {t: 0.0 for t in self.tickers}
            for d in dates:
                day_df = self.df_aligned[self.df_aligned["Date"] == d].set_index("Ticker")
                new_weights = {}
                turnover_sum = 0.0
                for t in self.tickers:
                    prev_w = current_weights[t]
                    if strategy_col is None:
                        target_w = max_asset_weight
                    elif t in day_df.index:
                        dec = day_df.loc[t, strategy_col]
                        if dec == "BUY":
                            target_w = max_asset_weight
                        elif dec == "SELL":
                            target_w = -max_asset_weight if mode == "LONG_SHORT" else 0.0
                        else:
                            target_w = prev_w
                    else:
                        target_w = prev_w
                    turnover_sum += abs(target_w - prev_w)
                    new_weights[t] = target_w
                fee = turnover_sum * cost_rate
                total_turnover_sum += turnover_sum
                if turnover_sum > 0:
                    trades += int(sum(abs(new_weights[t] - current_weights[t]) > 1e-4 for t in self.tickers))
                gross_ret = 0.0
                for t in self.tickers:
                    if t in day_df.index:
                        f_ret = day_df.loc[t, ret_col] if ret_col in day_df.columns else day_df.loc[t, "Forward_Return"]
                        if pd.notnull(f_ret):
                            gross_ret += new_weights[t] * float(f_ret)
                net_ret = gross_ret - fee
                equity *= (1.0 + net_ret)
                current_weights = new_weights
                history.append({
                    "Date": d,
                    "Equity": equity,
                    "Net_Daily_Return": net_ret,
                    "Gross_Daily_Return": gross_ret,
                    "Fee": fee,
                    "Turnover": turnover_sum,
                    "Portfolio_Exposure": sum(abs(w) for w in new_weights.values())
                })
                
        res_df = pd.DataFrame(history)
        n_sessions = len(dates)
        cum_ret = (equity / initial_capital) - 1.0
        ann_ret = (1.0 + cum_ret) ** (252.0 / n_sessions) - 1.0 if n_sessions > 0 else 0.0
        ann_vol = float(res_df["Net_Daily_Return"].std() * np.sqrt(252.0))
        sharpe = calculate_sharpe_ratio(res_df["Net_Daily_Return"], risk_free_rate)
        sortino = calculate_sortino_ratio(res_df["Net_Daily_Return"], risk_free_rate)
        mdd, mdd_dur = calculate_max_drawdown(res_df["Equity"])
        calmar = abs(ann_ret / mdd) if abs(mdd) > 1e-4 else 0.0
        mean_exp = float(res_df["Portfolio_Exposure"].mean())
        
        boot_ci = bootstrap_metric_ci(res_df["Net_Daily_Return"], risk_free_rate=risk_free_rate)
        
        strat_label = strategy_name or (strategy_col if strategy_col is not None else "Buy_and_Hold")
        metrics = PerformanceMetrics(
            strategy_name=strat_label,
            ticker="PORTFOLIO_10",
            mode=mode,
            cost_bps=cost_bps,
            total_sessions=n_sessions,
            initial_capital=initial_capital,
            final_equity=equity,
            cumulative_return=cum_ret,
            annualized_return=ann_ret,
            annualized_volatility=ann_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=mdd,
            max_drawdown_duration_days=mdd_dur,
            mean_exposure=mean_exp,
            total_turnover=total_turnover_sum,
            trade_count=trades,
            win_rate=float((res_df["Net_Daily_Return"] > 0).mean()),
            profit_factor=1.0,
            execution_mode=execution_mode,
            accounting_model=accounting_model,
            portfolio_weight_contribution=mean_exp,
            sharpe_ci_lower=boot_ci["sharpe_ci"][0],
            sharpe_ci_upper=boot_ci["sharpe_ci"][1],
            return_ci_lower=boot_ci["cum_return_ci"][0],
            return_ci_upper=boot_ci["cum_return_ci"][1]
        )
        return metrics, res_df

    def evaluate_cost_sensitivity(
        self,
        cost_bps_list: Optional[List[float]] = None,
        strategy_col: str = "Final_Decision",
        execution_mode: str = "NEXT_OPEN",
        accounting_model: str = "DOLLAR_LEDGER"
    ) -> pd.DataFrame:
        """Evaluate sensitivity of portfolio performance across different transaction cost levels."""
        costs = cost_bps_list or [0.0, 5.0, 10.0, 20.0, 50.0]
        rows = []
        for c_bps in costs:
            c_met, _ = self.evaluate_multi_asset_portfolio(
                strategy_col=strategy_col,
                execution_mode=execution_mode,
                accounting_model=accounting_model,
                cost_bps=c_bps
            )
            rows.append({
                "cost_bps": c_bps,
                "cumulative_return": c_met.cumulative_return,
                "annualized_return": c_met.annualized_return,
                "annualized_volatility": c_met.annualized_volatility,
                "sharpe_ratio": c_met.sharpe_ratio,
                "max_drawdown": c_met.max_drawdown,
                "total_turnover": c_met.total_turnover
            })
        return pd.DataFrame(rows)

    def run_full_suite(
        self,
        execution_mode: str = "NEXT_OPEN",
        accounting_model: str = "DOLLAR_LEDGER",
        cost_bps: float = 10.0,
        risk_free_rate: float = 0.03,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Execute comprehensive evaluation across all 10 equities and portfolio,
        comparing Full Fusion, Ablated (No News), RL Only, Class Only, and Buy & Hold.
        """
        out_dir = output_dir or EVALUATION_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        
        strategies = [
            ("Buy_and_Hold", None),
            ("Decision_Fusion_Full", "Final_Decision"),
            ("Decision_Fusion_NoNews", "Decision_NoNews"),
            ("RL_Only", "Decision_RL"),
            ("Classification_Only", "Decision_Class")
        ]
        
        all_metrics: List[Dict[str, Any]] = []
        equity_curves: Dict[str, pd.DataFrame] = {}
        
        # 1. Multi-Asset Portfolio Simulations
        portfolio_summaries = []
        for name, col in strategies:
            met, curve = self.evaluate_multi_asset_portfolio(
                strategy_col=col,
                strategy_name=name,
                execution_mode=execution_mode,
                accounting_model=accounting_model,
                cost_bps=cost_bps,
                risk_free_rate=risk_free_rate
            )
            all_metrics.append(met.to_dict())
            portfolio_summaries.append(met.to_dict())
            equity_curves[f"portfolio_{name}"] = curve
            
        # 2. Benchmark Relative Statistics (vs Buy and Hold) & CAPM Jensen's Alpha
        bh_curve = equity_curves["portfolio_Buy_and_Hold"]["Net_Daily_Return"].values
        rf_daily = risk_free_rate / 252.0
        
        for pm in portfolio_summaries:
            s_name = pm["strategy_name"]
            if s_name != "Buy_and_Hold":
                s_curve = equity_curves[f"portfolio_{s_name}"]["Net_Daily_Return"].values
                cov_mat = np.cov(s_curve, bh_curve)
                beta = cov_mat[0, 1] / cov_mat[1, 1] if cov_mat[1, 1] > 0 else 1.0
                
                # CAPM Jensen's Alpha: (E[R_s] - R_f) - beta * (E[R_m] - R_f)
                s_excess = np.mean(s_curve) - rf_daily
                m_excess = np.mean(bh_curve) - rf_daily
                capm_alpha = (s_excess - beta * m_excess) * 252.0
                alpha_ann = (np.mean(s_curve) - beta * np.mean(bh_curve)) * 252.0
                
                active_risk = np.std(s_curve - bh_curve) * np.sqrt(252.0)
                info_ratio = (np.mean(s_curve - bh_curve) * 252.0) / active_risk if active_risk > 0 else 0.0
                
                for m in all_metrics:
                    if m["strategy_name"] == s_name and m["ticker"] == "PORTFOLIO_10":
                        m["beta"] = float(beta)
                        m["capm_alpha"] = float(capm_alpha)
                        m["alpha_ann"] = float(alpha_ann)
                        m["information_ratio"] = float(info_ratio)
            else:
                for m in all_metrics:
                    if m["strategy_name"] == "Buy_and_Hold" and m["ticker"] == "PORTFOLIO_10":
                        m["beta"] = 1.0
                        m["capm_alpha"] = 0.0
                        m["alpha_ann"] = 0.0
                        m["information_ratio"] = 0.0

        # 3. Individual Stock Simulations across all 10 assets
        stock_metrics = []
        for ticker in self.tickers:
            for name, col in strategies:
                met, curve = self.evaluate_single_stock(
                    ticker=ticker,
                    strategy_col=col,
                    strategy_name=name,
                    execution_mode=execution_mode,
                    accounting_model=accounting_model,
                    mode="LONG_ONLY",
                    cost_bps=cost_bps,
                    risk_free_rate=risk_free_rate
                )
                all_metrics.append(met.to_dict())
                stock_metrics.append(met.to_dict())
                
        # 4. Statistical News Ablation Hypothesis Test (Full vs No News)
        full_rets = equity_curves["portfolio_Decision_Fusion_Full"]["Net_Daily_Return"].values
        nonews_rets = equity_curves["portfolio_Decision_Fusion_NoNews"]["Net_Daily_Return"].values
        diff_rets = full_rets - nonews_rets
        
        t_stat, p_val_ttest = stats.ttest_rel(full_rets, nonews_rets)
        non_zero_diff = diff_rets[diff_rets != 0]
        if len(non_zero_diff) > 0:
            wilc_stat, p_val_wilcoxon = stats.wilcoxon(non_zero_diff)
        else:
            wilc_stat, p_val_wilcoxon = 0.0, 1.0
            
        ablation_stats = {
            "mean_daily_return_difference": float(np.mean(diff_rets)),
            "annualized_return_difference": float(np.mean(diff_rets) * 252.0),
            "paired_ttest_t_stat": float(t_stat),
            "paired_ttest_p_value": float(p_val_ttest),
            "wilcoxon_stat": float(wilc_stat),
            "wilcoxon_p_value": float(p_val_wilcoxon),
            "statistically_significant_at_05": bool(p_val_ttest < 0.05)
        }
        
        # 5. Cost Sensitivity Analysis (0, 5, 10, 20, 50 bps)
        cost_sensitivity = []
        for c_bps in [0.0, 5.0, 10.0, 20.0, 50.0]:
            c_met, _ = self.evaluate_multi_asset_portfolio(
                strategy_col="Final_Decision", cost_bps=c_bps
            )
            cost_sensitivity.append({
                "cost_bps": c_bps,
                "cumulative_return": c_met.cumulative_return,
                "annualized_return": c_met.annualized_return,
                "annualized_volatility": c_met.annualized_volatility,
                "sharpe_ratio": c_met.sharpe_ratio,
                "max_drawdown": c_met.max_drawdown,
                "total_turnover": c_met.total_turnover
            })

        # 6. Save Data Artifacts to Isolated Directory
        df_metrics = pd.DataFrame(all_metrics)
        df_metrics.to_csv(out_dir / "all_strategy_metrics.csv", index=False)
        
        df_portfolio = df_metrics[df_metrics["ticker"] == "PORTFOLIO_10"]
        df_portfolio.to_csv(out_dir / "portfolio_comparison.csv", index=False)
        
        df_stocks = df_metrics[df_metrics["ticker"] != "PORTFOLIO_10"]
        df_stocks.to_csv(out_dir / "per_stock_comparison.csv", index=False)
        
        with open(out_dir / "news_ablation_significance.json", "w", encoding="utf-8") as f:
            json.dump(ablation_stats, f, indent=2)
            
        df_cost = pd.DataFrame(cost_sensitivity)
        df_cost.to_csv(out_dir / "cost_sensitivity.csv", index=False)
        
        logger.info(f"Phase 4 evaluation complete. Artifacts written to {out_dir}")
        return {
            "portfolio_metrics": df_portfolio.to_dict(orient="records"),
            "ablation_significance": ablation_stats,
            "cost_sensitivity": cost_sensitivity,
            "total_records": len(df_metrics)
        }
